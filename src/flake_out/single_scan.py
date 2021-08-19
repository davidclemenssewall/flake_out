# -*- coding: utf-8 -*-
"""
Class for loading, filtering, and saving single TLS scan position.

Created on Fri Aug  6 10:46:27 2021

@author: David Clemens-Sewall
"""

import os
import sys
import re
import copy
import json

import numpy as np
import pandas as pd
import open3d as o3d
import pdal
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.spatial import cKDTree
from datetime import datetime

class SingleScan:
    """
    Container for single lidar scan and methods for displaying it.
    
    ...
    
    Attributes
    ----------
    project_path : str
        Path to folder containing all Riscan projects.
    project_name : str
        Name of Riscan project.
    scan_name : str
        Typically ScanPos0XX where XX is the scan number.
    transform_dict : dict
        dict of vtkTransforms linked with this single scan.
    transform : vtkTransform
        pipelined, concatenated vtkTransform to apply to this scan.
    transformFilter : vtkTransformPolyDataFilter
        filter that applies the transform above
    currentFilter : varies, see init
        VTK pipeline object that returns
    polydata_raw : vtkPolyData
        Raw data read in from Riscan, we will add arrays to PointData. This
        polydata's PointData includes an array Classification. This is a uint8
        array with the classification of points defined as in the LAS 
        specification from ASPRS: 
        https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf 
        Plus additional catagories defined here
        0 : Created, Never Classified
        1 : Unclassified
        2 : Ground
        64: High Elevation (Classified by elevation_filter)
        65: Snowflake (Classified by returnindex filter)
        73: Manually Removed (not snowflake or surface point)
    dsa_raw : vtk.numpy_interface.dataset_adapter.Polydata
        dataset adaptor object for interacting with polydata_raw
    man_class : pandas dataframe
        Dataframe containing information on manually classified points. The
        dataframe is keyed on PointId and contains:
            user: name or identifier of person classifying point
            datetime: datetime when point was classified
            X, Y, Z: position of the point in the scanner's own coordinate 
                system
            Classification: Manual classification (number)
        The expected usage here is that the scan was orginally loaded from a
        LAS file and that the PointId field created on that original loading
        is the same as the PointId's of the points we add to this dataframe.
        Doing otherwise may cause duplicate points which could cause errors
        later on.
    
    Methods
    -------
    load_man_class()
        Load the manual classification table
    apply_transforms(transform_list)
        updates transform to be concatenation of transforms in transform list.
    add_sop()
        load the appropriate SOP matrix into transform_dict
    add_transform(key, matrix)
        add a transform to transform_dict
    create_elevation_pipeline(z_min, z_max, lower_threshold=-1000,
                              upper_threshold=1000)
        create mapper and actor for displaying points with colors by elevation
    apply_elevation_filter(z_max)
        Filter out all points above a certain height. Sets the flag in 
        Classification to 64.
    apply_snowflake_filter_3(z_std_mult, leafsize):
        Filter points as snowflakes based on whether their z value in the
        transformed reference frame exceeds z_std_mult multiples of the mean
        z values for points nearby (within a bucket of size, leafsize)
    apply_snowflake_filter_returnindex(cylinder_rad, radial_precision)
        Filter snowflakes based on their return index and whether they are on
        the border of the visible region.
    apply_early_return_filter():
        Label all early return points as snowflakes (classification 65).
    clear_classification
        Reset all Classification values to 0.
    update_man_class(pdata, classification)
        Update the points in man_class with the points in pdata.
    write_npy_pdal(output_dir, filename, mode)
        Write SingleScan to numpy structured array that can be read by pdal.
    write_scan(write_dir, class_list, suffix)
        Write the SingleScan to numpy files that can be loaded on init.
    """

    def __init__(self, project_path, project_name, scan_name, 
                 import_mode='import_las', create_id=True,
                 las_fieldnames=None, 
                 class_list=[0, 1, 2, 70], read_dir=None, suffix=''):
        """
        Creates SingleScan object and transformation pipeline.
        
        Note, if a polydata folder with the desired suffix does not exist then
        we will produce many vtk warnings (so I don't recommend this)

        Parameters
        ----------
        project_path : str
            Path to folder containing all Riscan projects.
        project_name : str
            Name of Riscan project.
        scan_name : str
            Typically ScanPos0XX where XX is the scan number.
        import_mode : str, optional
            How to create polydata_raw, the base data for this SingleScan. 
            Options are: 'poly' (read from Riscan generated poly), 'read_scan'
            (read saved vtp file), 'import_las' (use pdal to import from las
            file generate by Riscan), 'empty' (create an empty polydata, 
            useful if we just want to work with transformations). The default 
            is 'import_las.
        create_id: bool, optional
            If true and PointId's do not exist create PointIds. The default
            is True.
        las_fieldnames: list, optional
            List of fieldnames to load if we are importing from a las file
            Must include 'Points'. If None, and we are loading scans, read
            all arrays. If None and we are importing las then set to
            ['Points', 'NumberOfReturns', 'ReturnIndex', 'Reflectance',
             'Amplitude']. The default is None.
        class_list : list, optional
            List of categories this filter will return, if special value: 
            'all' Then we do not have a selection filter and we pass through 
            all points. The default is [0, 1, 2, 70].
        read_dir : str, optional
            Directory to read scan from. Defaults to npyfiles if None. The
            default is None.
        suffix : str, optional
            Suffix for npyfiles directory if we are reading scans. The default
            is '' which corresponds to the regular npyfiles directory.

        Returns
        -------
        None.

        """
        # Store instance attributes
        self.project_path = project_path
        self.project_name = project_name
        self.scan_name = scan_name

        # Read scan
        if import_mode=='read_scan':
            # Import directly from numpy files that we've already saved
            if read_dir is None:
                npy_path = os.path.join(self.project_path, self.project_name,
                                        'npyfiles' + suffix, self.scan_name)
            else:
                npy_path = read_dir
            
            if not os.path.isdir(npy_path):
                raise ValueError('npyfiles directory does not exist')
            # If las_fieldnames is None load all numpy files
            if las_fieldnames is None:
                filenames = os.listdir(npy_path)
                las_fieldnames = []
                for filename in filenames:
                    if re.search('.*npy$', filename):
                        las_fieldnames.append(filename)
            else:
                las_fieldnames = copy.deepcopy(las_fieldnames)
                for i in range(len(las_fieldnames)):
                    las_fieldnames[i] = las_fieldnames[i] + '.npy'
            
            pdata = vtk.vtkPolyData()
            self.np_dict = {}
            for k in las_fieldnames:
                try:
                    name = k.split('.')[0]
                    self.np_dict[name] = np.load(os.path.join(npy_path, k))
                    if name=='Points':
                        pts = vtk.vtkPoints()
                        if self.np_dict[name].dtype=='float64':
                            arr_type = vtk.VTK_DOUBLE
                        elif self.np_dict[name].dtype=='float32':
                            arr_type = vtk.VTK_FLOAT
                        else:
                            raise RuntimeError('Unrecognized dtype in ' + k)
                        pts.SetData(numpy_to_vtk(self.np_dict[name], 
                                                 deep=False, 
                                                 array_type=arr_type))
                        pdata.SetPoints(pts)
                    elif name=='Normals':
                        vtk_arr = numpy_to_vtk(self.np_dict[name], 
                                               deep=False, 
                                               array_type=vtk.VTK_FLOAT)
                        vtk_arr.SetName('Normals')
                        pdata.GetPointData().SetNormals(vtk_arr)
                    elif name=='PointId':
                        vtkarr = numpy_to_vtk(self.np_dict[name], deep=False,
                                              array_type=vtk.VTK_UNSIGNED_INT)
                        vtkarr.SetName(name)
                        pdata.GetPointData().SetPedigreeIds(vtkarr)
                        pdata.GetPointData().SetActivePedigreeIds('PointId')
                    else:
                        if self.np_dict[name].dtype=='float64':
                            arr_type = vtk.VTK_DOUBLE
                        elif self.np_dict[name].dtype=='float32':
                            arr_type = vtk.VTK_FLOAT
                        elif self.np_dict[name].dtype=='int8':
                            arr_type = vtk.VTK_SIGNED_CHAR
                        elif self.np_dict[name].dtype=='uint8':
                            arr_type = vtk.VTK_UNSIGNED_CHAR
                        elif self.np_dict[name].dtype=='uint32':
                            arr_type = vtk.VTK_UNSIGNED_INT
                        else:
                            raise RuntimeError('Unrecognized dtype in ' + k)
                        vtkarr = numpy_to_vtk(self.np_dict[name], deep=False,
                                              array_type=arr_type)
                        vtkarr.SetName(name)
                        pdata.GetPointData().AddArray(vtkarr)                
                except IOError:
                    print(k + ' does not exist in ' + npy_path)
                
            # Create VertexGlyphFilter so that we have vertices for
            # displaying
            pdata.Modified()
            self.polydata_raw = pdata

        elif import_mode=='import_las':
            # If las_fieldnames is None set it
            if las_fieldnames is None:
                las_fieldnames = ['Points', 'NumberOfReturns', 'ReturnIndex', 
                                  'Reflectance', 'Amplitude']
            # import las file from lasfiles directory in project_path
            filenames = os.listdir(os.path.join(self.project_path, 
                                                self.project_name, 
                                                "lasfiles"))
            pattern = re.compile(self.scan_name + '.*las')
            matches = [pattern.fullmatch(filename) for filename in filenames]
            if any(matches):
                # Create filename input
                filename = next(f for f, m in zip(filenames, matches) if m)
                json_list = [os.path.join(self.project_path, self.project_name, 
                             "lasfiles", filename)]
                json_data = json.dumps(json_list, indent=4)
                # Load scan into numpy array
                pipeline = pdal.Pipeline(json_data)
                _ = pipeline.execute()
                
                # Create pdata and populate with points from las file
                pdata = vtk.vtkPolyData()
                
                # np_dict stores references to underlying np arrays so that
                # they do not get garbage-collected
                self.np_dict = {}
                
                for k in las_fieldnames:
                    if k=='Points':
                        self.np_dict[k] = np.hstack((
                            np.float32(pipeline.arrays[0]['X'])[:, np.newaxis],
                            np.float32(pipeline.arrays[0]['Y'])[:, np.newaxis],
                            np.float32(pipeline.arrays[0]['Z'])[:, np.newaxis]
                            ))
                        pts = vtk.vtkPoints()
                        pts.SetData(numpy_to_vtk(self.np_dict[k], 
                                                 deep=False, 
                                                 array_type=vtk.VTK_FLOAT))
                        pdata.SetPoints(pts)
                    elif k in ['NumberOfReturns', 'ReturnIndex']:
                        if k=='ReturnIndex':
                            self.np_dict[k] = pipeline.arrays[0][
                                'ReturnNumber']
                            # Fix that return number 7 should be 0
                            self.np_dict[k][self.np_dict[k]==7] = 0
                            # Now convert to return index, so -1 is last return
                            # -2 is second to last return, etc
                            self.np_dict[k] = (self.np_dict[k] - 
                                               pipeline.arrays[0]
                                               ['NumberOfReturns'])
                            self.np_dict[k] = np.int8(self.np_dict[k])
                        else:
                            self.np_dict[k] = pipeline.arrays[0][k]
        
                        vtkarr = numpy_to_vtk(self.np_dict[k],
                                              deep=False,
                                              array_type=vtk.VTK_SIGNED_CHAR)
                        vtkarr.SetName(k)
                        pdata.GetPointData().AddArray(vtkarr)
                    elif k in ['Reflectance', 'Amplitude']:
                        self.np_dict[k] = pipeline.arrays[0][k]
                        vtkarr = numpy_to_vtk(self.np_dict[k],
                                              deep=False,
                                              array_type=vtk.VTK_DOUBLE)
                        vtkarr.SetName(k)
                        pdata.GetPointData().AddArray(vtkarr)
                
                # Create VertexGlyphFilter so that we have vertices for
                # displaying
                pdata.Modified()
                self.polydata_raw = pdata
            else:
                raise RuntimeError('Requested LAS file not found')
        else:
            raise ValueError('Invalid import_mode provided')
        
        # Create dataset adaptor for interacting with polydata_raw
        self.dsa_raw = dsa.WrapDataObject(self.polydata_raw)
        
        # Add Classification array to polydata_raw if it's not present
        if not self.polydata_raw.GetPointData().HasArray('Classification'):
            arr = vtk.vtkUnsignedCharArray()
            arr.SetName('Classification')
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.polydata_raw.GetNumberOfPoints())
            arr.FillComponent(0, 0)
            self.polydata_raw.GetPointData().AddArray(arr)
            self.polydata_raw.GetPointData().SetActiveScalars('Classification')
        # Set Classification array as active scalars
        self.polydata_raw.GetPointData().SetActiveScalars('Classification')
        
        
        # Add PedigreeIds if they are not already present
        if create_id and not ('PointId' in 
                              list(self.dsa_raw.PointData.keys())):
            pedigreeIds = vtk.vtkTypeUInt32Array()
            pedigreeIds.SetName('PointId')
            pedigreeIds.SetNumberOfComponents(1)
            pedigreeIds.SetNumberOfTuples(self.polydata_raw.
                                          GetNumberOfPoints())
            np_pedigreeIds = vtk_to_numpy(pedigreeIds)
            np_pedigreeIds[:] = np.arange(self.polydata_raw.
                                          GetNumberOfPoints(), dtype='uint32')
            self.polydata_raw.GetPointData().SetPedigreeIds(pedigreeIds)
            self.polydata_raw.GetPointData().SetActivePedigreeIds('PointId')

        self.polydata_raw.Modified()
        
        self.transform = vtk.vtkTransform()
        # Set mode to post-multiply, so concatenation is successive transforms
        self.transform.PostMultiply()
        self.transformFilter = vtk.vtkTransformPolyDataFilter()
        self.transformFilter.SetTransform(self.transform)
        self.transformFilter.SetInputData(self.polydata_raw)
        self.transformFilter.Update()
   
        # Create other attributes
        self.transform_dict = {}
        self.trans_history_dict = {}
        self.filterName = 'None'
        self.filterDict = {}
        
        # Create currentFilter
        if class_list=='all':
            self.currentFilter = self.transformFilter
        else:
            selectionList = vtk.vtkUnsignedCharArray()
            for v in class_list:
                selectionList.InsertNextValue(v)
            selectionNode = vtk.vtkSelectionNode()
            selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
            selectionNode.SetContentType(vtk.vtkSelectionNode.VALUES)
            selectionNode.SetSelectionList(selectionList)
            
            selection = vtk.vtkSelection()
            selection.AddNode(selectionNode)
            
            self.extractSelection = vtk.vtkExtractSelection()
            self.extractSelection.SetInputData(1, selection)
            self.extractSelection.SetInputConnection(0, 
                                        self.transformFilter.GetOutputPort())
            self.extractSelection.Update()
            
            # Unfortunately, extractSelection produces a vtkUnstructuredGrid
            # so we need to use vtkGeometryFilter to convert to polydata
            self.currentFilter = vtk.vtkGeometryFilter()
            self.currentFilter.SetInputConnection(self.extractSelection
                                                  .GetOutputPort())
            self.currentFilter.Update()

    def load_man_class(self):
        """
        Load the man_class dataframe. Create if it does not exist.

        Returns
        -------
        None.

        """
        
        # Check if directory for manual classifications exists and create
        # if it doesn't.
        create_df = False
        if os.path.isdir(os.path.join(self.project_path, self.project_name, 
                         'manualclassification')):
            # Check if file exists
            if os.path.isfile(os.path.join(self.project_path, self.project_name, 
                              'manualclassification', self.scan_name +
                              '.parquet')):
                self.man_class = pd.read_parquet(os.path.join(self.project_path, 
                                                 self.project_name,
                                                 'manualclassification', 
                                                 self.scan_name + '.parquet'),
                                                 engine="pyarrow")
            # otherwise create dataframe
            else:
                create_df = True
        else:
            # Create directory and dataframe
            create_df = True
            os.mkdir(os.path.join(self.project_path, self.project_name, 
                     'manualclassification'))
        
        if create_df:
            self.man_class = pd.DataFrame({'user':
                                           pd.Series([], dtype='string'),
                                           'datetime':
                                           pd.Series([], dtype='datetime64[ns]'),
                                           'X':
                                           pd.Series([], dtype=np.float32),
                                           'Y':
                                           pd.Series([], dtype=np.float32),
                                           'Z':
                                           pd.Series([], dtype=np.float32),
                                           'Classification':
                                           pd.Series([], dtype=np.uint8)})
            self.man_class.index.name = 'PointId'

    def add_transform(self, key, matrix):
        """
        Adds a new transform to the transform_dict

        Parameters
        ----------
        key : str
            Name of the tranform (e.g. 'sop')
        matrix : 4x4 array-like
            4x4 matrix of transformation in homologous coordinates.

        Returns
        -------
        None.

        """
        
        # Create vtk transform object
        vtk4x4 = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk4x4.SetElement(i, j, matrix[i, j])
        transform = vtk.vtkTransform()
        transform.SetMatrix(vtk4x4)
        # Add transform to transform_dict
        self.transform_dict.update({key : transform})
    
    def add_sop(self):
        """
        Add the sop matrix to transform_dict. Must have exported from RiSCAN

        Returns
        -------
        None.

        """
        
        trans = np.genfromtxt(os.path.join(self.project_path, self.project_name, 
                              self.scan_name + '.DAT'), delimiter=' ')
        self.add_transform('sop', trans)

    def apply_transforms(self, transform_list):
        """
        Update transform to be a concatenation of transform_list.
        
        Clears existing transform!

        Parameters
        ----------
        transform_list : list
            str in list must be keys in transform_dict. Transformations are 
            applied in the same order as in the list (postmultiply order)

        Returns
        -------
        None.

        """
        # Reset transform to the identity
        self.transform.Identity()
        
        for i, key in enumerate(transform_list):
            try:
                self.transform.Concatenate(self.transform_dict[key])
                
            except Exception as e:
                print("Requested transform " + key + " is not in " +
                      "transform_dict")
                print(e)

        self.transformFilter.Update()
        self.currentFilter.Update()

    def clear_classification(self, ignore_list=[]):
        """
        Reset Classification for all points to 0
        
        Parameters:
        -----------
        ignore_list : list, optional
            List of categories to ignore when clearing classification. The
            default is [].

        Returns
        -------
        None.

        """
        
        uni = np.unique(self.dsa_raw.PointData['Classification'])
        for u in uni:
            if not (u in ignore_list):
                self.dsa_raw.PointData['Classification'][
                    self.dsa_raw.PointData['Classification']==u] = 0
        # Update currentTransform
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
    
    def update_man_class(self, pdata, classification, user=''):
        """
        Update the points in man_class with the points in pdata.
        
        See documentation under SingleScan for description of man_class

        Parameters
        ----------
        pdata : vtkPolyData
            PolyData containing the points to add to man_class.
        classification : uint8
            The classification code of the points. See SingleScan 
            documentation for mapping from code to text
        user : string
            Identifier or name for the person classifying. The default is ''

        Returns
        -------
        None.

        """
        
        # Raise exception if man class table doesn't exist
        if not hasattr(self, 'man_class'):
            raise RuntimeError('man_class table does not exist. '
                               + 'load it first?')
        
        # Inverse Transform to get points in Scanners Own Coordinate System
        invTransform = vtk.vtkTransformFilter()
        invTransform.SetTransform(self.transform.GetInverse())
        invTransform.SetInputData(pdata)
        invTransform.Update()
        pdata_inv = invTransform.GetOutput()
        
        # Create a dataframe from selected points
        dsa_pdata = dsa.WrapDataObject(pdata_inv)
        n_pts = pdata_inv.GetNumberOfPoints()
        df_trans = pd.DataFrame({'X' : dsa_pdata.Points[:,0],
                                 'Y' : dsa_pdata.Points[:,1],
                                 'Z' : dsa_pdata.Points[:,2],
                                 'Classification' : classification * np.ones(
                                     n_pts, dtype=np.uint8),
                                 'user' : pd.Series([user for i in range(n_pts)]
                                                    , dtype='string'),
                                 'datetime' : (np.datetime64(datetime.now(), 
                                                             'ns') +
                                               np.zeros(n_pts, 
                                                       dtype='timedelta64[ns]'))
                                 },
                                index=dsa_pdata.PointData['PointId'], 
                                copy=True)
        df_trans.index.name = 'PointId'
        
        # Join the dataframe with the existing one, overwrite points if we
        # have repicked some points.
        self.man_class = df_trans.combine_first(self.man_class)
        
        # drop columns that we don't have. Because they show up as 
        # vtkNoneArray their datatype is object.
        self.man_class = self.man_class.select_dtypes(exclude=['object'])
        
        # Write to file to save
        self.man_class.to_parquet(os.path.join(self.project_path, 
                                                 self.project_name, 
                                                 'manualclassification', 
                                                 self.scan_name + '.parquet'),
                                                 engine="pyarrow", 
                                                 compression=None)

    def apply_elevation_filter(self, z_max):
        """
        Set Classification for all points above z_max to be 64. 

        Parameters
        ----------
        z_max : float
            Maximum z-value (in reference frame of currentTransform).

        Returns
        -------
        None.

        """
        
        # If the current filter output has no points, return
        if self.currentFilter.GetOutput().GetNumberOfPoints()==0:
            return
        # Get the points of the currentTransform as a numpy array
        Points = vtk_to_numpy(self.currentFilter.GetOutput().GetPoints()
                              .GetData())
        PointIds = vtk_to_numpy(self.currentFilter.GetOutput().GetPointData().
                        GetArray('PointId'))
        # Set the in Classification for points whose z-value is above z_max to 
        # 64
        self.dsa_raw.PointData['Classification'][np.isin(self.dsa_raw.PointData
            ['PointId'], PointIds[Points[:,2]>z_max], assume_unique=True)] =64
        # Update currentTransform
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
    
    def apply_snowflake_filter_3(self, z_std_mult, leafsize):
        """
        Filter points as snowflakes based on whether their z value in the
        transformed reference frame exceeds z_std_mult multiples of the mean
        z values for points nearby (within a bucket of size leafsize).

        We apply this only to the output of currentFilter!

        All points that this filter identifies as snowflakes are set to
        Classification=65

        Parameters
        ----------
        z_std_mult : float
            The number of positive z standard deviations greater than other
            nearby points for us to classify it as a snowflake.
        leafsize : int
            maximum number of points in each bucket (we use scipy's
            KDTree)

        Returns
        -------
        None.

        """
        
        # If the current filter output has no points, return
        if self.currentFilter.GetOutput().GetNumberOfPoints()==0:
            return
        # Step 1, get pointer to points array and create tree
        Points = vtk_to_numpy(self.currentFilter.GetOutput().GetPoints()
                              .GetData())
        PointIds = vtk_to_numpy(self.currentFilter.GetOutput().GetPointData().
                        GetArray('PointId'))
        tree = cKDTree(Points[:,:2], leafsize=leafsize)
        # Get python accessible version
        ptree = tree.tree

        # Step 2, define the recursive function that we'll use
        def z_std_filter(node, z_std_mult, Points, bool_arr):
            # If we are not at a leaf, call this function on each child
            if not node.split_dim==-1:
                # Call this function on the lesser node
                z_std_filter(node.lesser, z_std_mult, Points, bool_arr)
                # Call this function on the greater node
                z_std_filter(node.greater, z_std_mult, Points, bool_arr)
            else:
                # We are at a leaf. Compute distance from mean
                ind = node.indices
                z_mean = Points[ind, 2].mean()
                z_std = Points[ind, 2].std()
                bool_arr[ind] = (Points[ind, 2] - z_mean) > (z_std_mult * z_std)

        # Step 3, Apply function
        bool_arr = np.empty(Points.shape[0], dtype=np.bool_)
        z_std_filter(ptree, z_std_mult, Points, bool_arr)

        # Step 4, modify Classification field in polydata_raw
        # Use bool_arr to index into PointIds, use np.isin to find indices
        # in dsa_raw
        self.dsa_raw.PointData['Classification'][np.isin(self.dsa_raw.PointData
            ['PointId'], PointIds[bool_arr], assume_unique=True)] = 65
        del ptree, tree, PointIds, Points
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()

    def apply_snowflake_filter_returnindex(self, cylinder_rad=0.025*np.sqrt(2)
                                           *np.pi/180, radial_precision=0):
        """
        Filter snowflakes using return index visible space.
        
        Snowflakes are too small to fully occlude the laser pulse. Therefore
        all snowflakes will be one of multiple returns (returnindex<-1).
        However, the edges of shadows will also be one of multiple returns. To
        address this we look at each early return and check if it's on the 
        border of the visible area from the scanner's perspective. We do this
        by finding all points within cylinder_rad of the point in question
        in panorama space. Then, if the radial value of the point in question
        is greater than any of these radial values that means the point
        in question is on the border of the visible region and we should keep
        it.
        
        All points that this filter identifies as snowflakes are set to
        Classification=65

        Parameters
        ----------
        cylinder_rad : float, optional
            The radius of a cylinder, in radians around an early return
            to look for last returns. The default is 0.025*np.sqrt(2)*np.pi/
            180.
        radial_precision : float, optional
            If an early return's radius is within radial_precision of an
            adjacent last return accept it as surface. The default is 0.

        Returns
        -------
        None.

        """
        
        # Convert to polar coordinates
        sphere2cart = vtk.vtkSphericalTransform()
        cart2sphere = sphere2cart.GetInverse()
        transformFilter = vtk.vtkTransformFilter()
        transformFilter.SetTransform(cart2sphere)
        transformFilter.SetInputData(self.polydata_raw)
        transformFilter.Update()
        
        # Get only last returns
        (transformFilter.GetOutput().GetPointData().
         SetActiveScalars('ReturnIndex'))
        thresholdFilter = vtk.vtkThresholdPoints()
        thresholdFilter.ThresholdByUpper(-1.5)
        thresholdFilter.SetInputConnection(transformFilter.GetOutputPort())
        thresholdFilter.Update()
        
        # Transform such that points are  in x and y and radius is in Elevation field
        swap_r_phi = vtk.vtkTransform()
        swap_r_phi.SetMatrix((0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1))
        filter_r_phi = vtk.vtkTransformFilter()
        filter_r_phi.SetTransform(swap_r_phi)
        filter_r_phi.SetInputConnection(thresholdFilter.GetOutputPort())
        filter_r_phi.Update()
        radialElev = vtk.vtkSimpleElevationFilter()
        radialElev.SetVector(0, 0, 1.0)
        radialElev.SetInputConnection(filter_r_phi.GetOutputPort())
        radialElev.Update()
        flattener = vtk.vtkTransformFilter()
        transFlat = vtk.vtkTransform()
        transFlat.Scale(1, 1, 0)
        flattener.SetTransform(transFlat)
        flattener.SetInputConnection(radialElev.GetOutputPort())
        flattener.Update()
        
        # Create locator for last returns
        locator = vtk.vtkStaticPointLocator2D()
        flat_last_returns = flattener.GetOutput()
        flat_last_returns.SetPointLocator(locator)
        locator.SetDataSet(flat_last_returns)
        flat_last_returns.BuildPointLocator()
        
        # Get early returns as possible snowflakes
        thresholdFilterL = vtk.vtkThresholdPoints()
        thresholdFilterL.ThresholdByLower(-1.5)
        thresholdFilterL.SetInputConnection(transformFilter.GetOutputPort())
        thresholdFilterL.Update()
        early_returns = thresholdFilterL.GetOutput()
        
        # Allocate objects needed to find nearby points
        result = vtk.vtkIdList()
        pt = np.zeros(3)
        snowflake = True
        
        for i in np.arange(early_returns.GetNumberOfPoints()):
            # Get the point in question
            early_returns.GetPoint(i, pt)
            # Get the adjacent points from last_returns and place id's in result
            (flat_last_returns.GetPointLocator().FindPointsWithinRadius(
                cylinder_rad, pt[2], pt[1], 0, result))
            # If the radius of the point in question is larger than that of 
            # any of the adjacent point, then that means we are on the edge of
            # the lidar's vision and this point is probably not a snowflake
            snowflake = True
            for j in range(result.GetNumberOfIds()):
                if pt[0] >= (flat_last_returns.GetPointData().
                             GetAbstractArray('Elevation').GetTuple(result.
                                                                    GetId(j)
                                                                    )[0]
                             -radial_precision):
                    snowflake = False
                    break
            if snowflake:
                self.dsa_raw.PointData['Classification'][self.dsa_raw.PointData[
                    'PointId']==early_returns.GetPointData().
                    GetPedigreeIds().GetValue(i)] = 65
        
        # Update currentTransform
        self.polydata_raw.GetPointData().SetActiveScalars('Classification')
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()

    def apply_early_return_filter(self):
        """
        Label any early returns in currentFilter output as snowflakes (65)

        Returns
        -------
        None.


        """

        # Get relevant arrays from currentFilter output
        Points = vtk_to_numpy(self.currentFilter.GetOutput().GetPoints()
                              .GetData())
        PointIds = vtk_to_numpy(self.currentFilter.GetOutput().GetPointData().
                        GetArray('PointId'))
        ReturnIndex = vtk_to_numpy(self.currentFilter.GetOutput().GetPointData().
                        GetArray('ReturnIndex'))

        # Set Classification field in polydata_raw to be 65 where ReturnIndex
        # is less than -1 (the point is an early return)
        self.dsa_raw.PointData['Classification'][np.isin(self.dsa_raw.PointData
            ['PointId'], PointIds[ReturnIndex<-1], assume_unique=True)] = 65
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()

    def apply_radius_outlier_filter(self, nb_points, radius):
        """
        Use Open3D to apply radius outlier filter to currentFilter output.

        Parameters
        ----------
        nb_points : int
            If number of points within sphere is less than nb_points set
            Classification to 65 (snowflake)
        radius : float
            Radius of sphere to find neighbors within

        Returns
        -------
        None.


        """

        # Get relevant arrays from currentFilter output
        Points = vtk_to_numpy(self.currentFilter.GetOutput().GetPoints()
                              .GetData())
        PointIds = vtk_to_numpy(self.currentFilter.GetOutput().GetPointData().
                        GetArray('PointId'))

        # Create Open3d pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(Points)

        # Apply radius outlier removal
        _, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)


        # Set Classification field in polydata_raw to be 65 where radius
        # outlier removal removed points
        self.dsa_raw.PointData['Classification'][np.isin(self.dsa_raw.PointData
            ['PointId'], PointIds[ind], assume_unique=True)] = 65
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()

    def write_npy_pdal(self, output_dir=None, filename=None, 
                       mode='transformed', skip_fields=[]):
        """
        Write scan to structured numpy array that can be read by PDAL.

        Parameters
        ----------
        output_dir : str, optional
            Directory to write to. If none will write to the 'temp' folder
            under the project name.
        filename : str, optional
            Filename to write, if None will write PROJECT_NAME_SCAN_NAME. 
            The default is None.
        mode : str, optional
            Whether to write 'raw' points, 'transformed' points, or 'filtered'
            points. The default is 'transformed'.
        skip_fields : list, optional
            Fields to skip in writing. If this is 'all' then only write x,
            y, z. Otherwise should be a list of field names. The default is []

        Returns
        -------
        None.

        """
        
        if mode=='raw':
            pdata = self.polydata_raw
            dsa_pdata = self.dsa_raw
        elif mode=='transformed':
            pdata = self.transformFilter.GetOutput()
            dsa_pdata = dsa.WrapDataObject(pdata)
        elif mode=='filtered':
            pdata = self.currentFilter.GetOutput()
            dsa_pdata = dsa.WrapDataObject(pdata)
        else:
            raise ValueError('mode must be raw, transformed, or filtered')
        
        n_pts = pdata.GetNumberOfPoints()
        
        # Create numpy output
        names = []
        for name in dsa_pdata.PointData.keys():
            if name=='PointId':
                names.append(name)
            else:
                if skip_fields=='all':
                    continue
                elif name in skip_fields:
                    continue
                else:
                    names.append(name)
        formats = []
        for name in names:
            formats.append(dsa_pdata.PointData[name].dtype)
        names = tuple(names + ['X', 'Y', 'Z'])
        formats.append(np.float32)
        formats.append(np.float32)
        formats.append(np.float32)
        formats = tuple(formats)
        output_npy = np.zeros(n_pts, dtype={'names':names, 'formats':formats})
        for name in names:
            if name=='X':
                output_npy['X'] = dsa_pdata.Points[:,0]
            elif name=='Y':
                output_npy['Y'] = dsa_pdata.Points[:,1]
            elif name=='Z':
                output_npy['Z'] = dsa_pdata.Points[:,2]                
            else:
                output_npy[name] = dsa_pdata.PointData[name]
                
        if output_dir is None:
            output_dir = os.path.join(self.project_path, 'temp')
        if filename is None:
            filename = self.project_name + '_' + self.scan_name + '.npy'
        
        np.save(os.path.join(output_dir, filename), output_npy)

    def write_las_pdal(self, output_dir=None, filename=None, 
                       mode='transformed', skip_fields=[]):
        """
        Write the data in the project to LAS using pdal

        Parameters
        ----------
        output_dir : str, optional
            Directory to write to. If none defaults to project_path +
            project_name + '\\lasfiles\\pdal_output\\'. The default is None
        filename : str, optional
            Filename, if none uses scan name. The default is None.
        mode : str, optional
            Whether to write 'raw' points, 'transformed' points, or 'filtered'
            points. The default is 'transformed'.
        skip_fields : list, optional
            Fields to skip in writing. If this is 'all' then only write x,
            y, z. Otherwise should be a list of field names. The default is []

        Returns
        -------
        None.

        """
        
        # Handle output dir
        if output_dir is None:
            if not os.path.isdir(os.path.join(self.project_path, 
                                              self.project_name, 'lasfiles', 
                                              'pdal_output')):
                os.mkdir(os.path.join(self.project_path, self.project_name, 
                          'lasfiles', 'pdal_output'))
            output_dir = os.path.join(self.project_path, self.project_name, 
                          'lasfiles', 'pdal_output')
        if filename is None:
            filename = self.scan_name
        
        # Write each scan individually to a numpy output
        json_list = []
        self.write_npy_pdal(output_dir, mode=mode,skip_fields=skip_fields)
        json_list.append({"filename": os.path.join(output_dir, 
                                                   self.project_name + '_' + 
                                                   self.scan_name + '.npy'), 
                          "type": "readers.numpy"})
        
        # Create JSON to instruct conversion
        json_list.append({"type": "writers.las",
                          "filename": os.path.join(output_dir, 
                                                   filename + '.las'),
                          "minor_version": 4,
                          "dataformat_id": 0})
        json_data = json.dumps(json_list, indent=4)
        pipeline = pdal.Pipeline(json_data)
        _ = pipeline.execute()
        del _

    def write_scan(self, write_dir=None, class_list=None, suffix=''):
        """
        Write the scan to a collection of numpy files.
        
        This enables us to save the Classification field so we don't need to 
        run all of the filters each time we load data. Additionally, npy files
        are much faster to load than vtk files. Finally, we need to write
        the history_dict to this directory as well.
        
        Parameters
        ----------
        write_dir: str, optional
            Directory to write scan files to. If None write default npyfiles
            location. The default is None.
        class_list: list, optional
            Whether to first filter the data so that we only write points whose
            Classification values are in class_list. If None do not filter.
            The default is None.
        suffix: str, optional
            Suffix for writing to the correct npyfiles directory. The default
            is ''.

        Returns
        -------
        None.

        """
        
        npy_dir = "npyfiles" + suffix
        
        
        if write_dir is None:
            # If the write directory doesn't exist, create it
            if not os.path.isdir(os.path.join(self.project_path, 
                                              self.project_name, npy_dir)):
                os.mkdir(os.path.join(self.project_path, self.project_name, 
                                      npy_dir))
            # Within npyfiles we need a directory for each scan
            if not os.path.isdir(os.path.join(self.project_path, 
                                              self.project_name, npy_dir, 
                                              self.scan_name)):
                os.mkdir(os.path.join(self.project_path, self.project_name, 
                                      npy_dir, self.scan_name))
            write_dir = os.path.join(self.project_path, self.project_name, 
                                     npy_dir, self.scan_name)
        
        # Delete old saved SingleScan files in the directory
        for f in os.listdir(write_dir):
            os.remove(os.path.join(write_dir, f))

        # If class_list is None just write raw data
        if class_list is None:
            # Save Points
            np.save(os.path.join(write_dir, 'Points.npy'), self.dsa_raw.Points)
            # Save Normals if we have them
            if not self.polydata_raw.GetPointData().GetNormals() is None:
                np.save(os.path.join(write_dir, 'Normals.npy'), vtk_to_numpy(
                    self.polydata_raw.GetPointData().GetNormals()))
            # Save arrays
            for name in self.dsa_raw.PointData.keys():
                np.save(os.path.join(write_dir, name), 
                        self.dsa_raw.PointData[name])
        else:
            ind = np.isin(self.dsa_raw.PointData['Classification'], class_list)

            # Save Points
            np.save(os.path.join(write_dir, 'Points.npy'), 
                    self.dsa_raw.Points[ind, :])
            # Save Normals if we have them
            if not self.polydata_raw.GetPointData().GetNormals() is None:
                np.save(os.path.join(write_dir, 'Normals.npy'), vtk_to_numpy(
                    self.polydata_raw.GetPointData().GetNormals())[ind, :])
            # Save arrays
            for name in self.dsa_raw.PointData.keys():
                np.save(os.path.join(write_dir, name), 
                        self.dsa_raw.PointData[name][ind])
    
    def write_classification_suffix(self, class_suffix, pts_suffix=''):
        """
        Write the current classification array to the npyfiles directory with
        a suffix appended to the filename (so it may be loaded in the future)

        Parameters
        ----------
        class_suffix : str
            Suffix to append to filename 
            (will be Classification[class_suffix].npy)
        pts_suffix : str, optional
            Suffix for this set of npyfiles.

        Returns
        -------
        None.

        """

        npy_dir = "npyfiles" + pts_suffix
        write_dir = os.path.join(self.project_path, self.project_name, 
                                     npy_dir, self.scan_name)
        np.save(os.path.join(write_dir, 'Classification' + class_suffix), 
                        self.dsa_raw.PointData['Classification'])
