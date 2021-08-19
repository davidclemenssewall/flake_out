#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manual_snowflake_classifier.py

Application for manually classifying snowflakes

Created on Mon Aug  9 15:51:25 2021

@author: David Clemens-Sewall
"""

import os
import sys
import vtk
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from vtk.util.numpy_support import vtk_to_numpy
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# this code block is just for development, could be omitted
try:
    from flake_out.single_scan import SingleScan
except ModuleNotFoundError:
    print('flake_out not found, importing from local directory')
    sys.path.append('..')
    from flake_out.single_scan import SingleScan

class MyQVTKRenderWindowInteractor(QVTKRenderWindowInteractor):
    keyPressed = QtCore.pyqtSignal(int)
    
    def keyPressEvent(self, event):
        self.keyPressed.emit(event.key())

class MainWindow(Qt.QMainWindow):
    
    def __init__(self, parent=None):
        Qt.QMainWindow.__init__(self, parent)
        
        # Create the main layout
        self.frame = Qt.QFrame()
        main_layout = Qt.QHBoxLayout()
        
        # Create the visualization layout, will contain the renderwindow
        # and a toolbar with options
        vis_layout = Qt.QVBoxLayout()
        
        # Create the vis_tools_layout to sit beneath the renderwindow
        vis_tools_layout = Qt.QHBoxLayout()
        
        # Populate the vis_tools_layout
        self.v_min = Qt.QLineEdit('-5.0')
        self.v_min.setValidator(Qt.QDoubleValidator())
        self.v_min.setEnabled(0)
        self.v_max = Qt.QLineEdit('-1.0')
        self.v_max.setValidator(Qt.QDoubleValidator())
        self.v_max.setEnabled(0)
        self.near_label = Qt.QLineEdit('0.1')
        self.near_label.setValidator(Qt.QDoubleValidator())
        self.far_label = Qt.QLineEdit('1000.0')
        self.far_label.setValidator(Qt.QDoubleValidator())
        reset_focus_button = Qt.QPushButton('Reset Focus')
        self.focal_distance_label = Qt.QLabel('Foc. Dist: ')
        see_class_button = Qt.QPushButton('See Class')
        vis_tools_layout.addWidget(self.v_min)
        vis_tools_layout.addWidget(self.v_max)
        vis_tools_layout.addWidget(self.near_label)
        vis_tools_layout.addWidget(self.far_label)
        vis_tools_layout.addWidget(reset_focus_button)
        vis_tools_layout.addWidget(self.focal_distance_label)
        vis_tools_layout.addWidget(see_class_button)
        
        # Populate the vis_layout
        self.vtkWidget = MyQVTKRenderWindowInteractor(self.frame)
        self.vtkWidget.setSizePolicy(Qt.QSizePolicy.Expanding, 
                                     Qt.QSizePolicy.Expanding)
        vis_layout.addWidget(self.vtkWidget)
        vis_layout.addLayout(vis_tools_layout)
        
        # Create the Options layout, which will contain tools to select files
        # classify points, etc
        # Some additions here to make this scrollable
        opt_scroll = Qt.QScrollArea()
        opt_scroll.setWidgetResizable(True)
        opt_inner = Qt.QFrame(opt_scroll)
        opt_layout = Qt.QVBoxLayout()
        opt_inner.setLayout(opt_layout)
        opt_scroll.setWidget(opt_inner)
        
        # Populate the opt_layout
        # User
        opt_layout.addWidget(Qt.QLabel('User ID:'))
        self.user_lineedit = Qt.QLineEdit('')
        opt_layout.addWidget(self.user_lineedit)
        # Random seed, put this at the top so that users can adjust
        opt_layout.addWidget(Qt.QLabel('Random Seed'))
        self.rseed_lineedit = Qt.QLineEdit('1234')
        self.rseed_lineedit.setValidator(Qt.QIntValidator())
        opt_layout.addWidget(self.rseed_lineedit)
        self.on_rseed_changed('1234')
        # What proportion of snowflakes to show user
        opt_layout.addWidget(Qt.QLabel('Snowflake Prop:'))
        self.prop_lineedit = Qt.QLineEdit('0.5')
        zero_to_one = Qt.QDoubleValidator()
        zero_to_one.setBottom(0.0)
        zero_to_one.setTop(1.0)
        self.prop_lineedit.setValidator(zero_to_one)
        opt_layout.addWidget(self.prop_lineedit)
        
        # Data button
        self.sel_scan_area_button = Qt.QPushButton("Select Project Dir:")
        # Create the file dialog that we'll use
        self.proj_dialog = Qt.QFileDialog(self)
        def_path = os.path.join('..', '..')
        if os.path.isdir(def_path):
            self.proj_dialog.setDirectory(def_path)
        self.proj_dialog.setFileMode(4) # set file mode to pick directories
        opt_layout.addWidget(self.sel_scan_area_button)
        
        # ComboBox containing available projects
        self.proj_combobox = Qt.QComboBox()
        self.proj_combobox.setEnabled(0)
        self.proj_combobox.setSizeAdjustPolicy(0)
        opt_layout.addWidget(self.proj_combobox)
        
        # Combobox containing available scans
        self.scan_combobox = Qt.QComboBox()
        self.scan_combobox.setEnabled(0)
        self.scan_combobox.setSizeAdjustPolicy(0)
        opt_layout.addWidget(self.scan_combobox)

        # Lineedit to enter the Classification suffix
        opt_layout.addWidget(Qt.QLabel('Classification suffix:'))
        self.class_lineedit = Qt.QLineEdit('')
        opt_layout.addWidget(self.class_lineedit)
        
        # Button to prompt us to select a project
        self.sel_proj_button = Qt.QPushButton("Select Scan")
        self.sel_proj_button.setEnabled(0)
        opt_layout.addWidget(self.sel_proj_button)
        
        # PointID entry
        opt_layout.addWidget(Qt.QLabel('PointID:'))
        self.pointid_lineedit = Qt.QLineEdit('1')
        self.pointid_lineedit.setValidator(Qt.QIntValidator())
        opt_layout.addWidget(self.pointid_lineedit)
        goto_point_button = Qt.QPushButton('Go To Point')
        opt_layout.addWidget(goto_point_button)
        self.return_label = Qt.QLabel('')
        opt_layout.addWidget(self.return_label)
        
        # Choices for manual classification
        ground_button = Qt.QPushButton('Ground')
        opt_layout.addWidget(ground_button)
        flake_button = Qt.QPushButton('Snowflake')
        opt_layout.addWidget(flake_button)
        other_button = Qt.QPushButton('Other')
        opt_layout.addWidget(other_button)
        next_button = Qt.QPushButton('Skip Point')
        opt_layout.addWidget(next_button)
        self.undo_button = Qt.QPushButton('Undo Last')
        opt_layout.addWidget(self.undo_button)
        save_button = Qt.QPushButton('Save Classified')
        opt_layout.addWidget(save_button)
        
        # Populate the main layout
        main_layout.addLayout(vis_layout, stretch=5)
        main_layout.addWidget(opt_scroll)
        
        # Set layout for the frame and set central widget
        self.frame.setLayout(main_layout)
        self.setCentralWidget(self.frame)
        
        # Signals and slots
        # vis_tools
        self.vtkWidget.keyPressed.connect(self.on_key)
        self.v_min.editingFinished.connect(self.on_v_edit)
        self.v_max.editingFinished.connect(self.on_v_edit)
        self.near_label.editingFinished.connect(self.on_clip_changed)
        self.far_label.editingFinished.connect(self.on_clip_changed)
        reset_focus_button.clicked.connect(self.reset_focus)
        see_class_button.clicked.connect(self.on_see_class_button_click)
        # options layout
        self.sel_scan_area_button.clicked.connect(
            self.on_sel_scan_area_button_click)
        self.proj_dialog.fileSelected.connect(self.on_scan_area_selected)
        self.proj_combobox.currentTextChanged.connect(self.on_proj_change)
        self.sel_proj_button.clicked.connect(self.on_sel_proj_button_click)
        goto_point_button.clicked.connect(self.goto_point)
        ground_button.clicked.connect(self.on_ground_button_click)
        flake_button.clicked.connect(self.on_flake_button_click)
        other_button.clicked.connect(self.on_other_button_click)
        next_button.clicked.connect(self.next_point)
        self.undo_button.clicked.connect(self.on_undo_button_click)
        save_button.clicked.connect(self.on_save_button_click)
        
        self.show()
        
        # Create polydata for point
        pts0 = vtk.vtkPoints()
        pts0.SetNumberOfPoints(1)
        pts0.SetPoint(0, 0.0, 0.0, 0.0)
        pt_0 = vtk.vtkPolyData()
        pt_0.SetPoints(pts0)
        self.vgf_pt_0 = vtk.vtkVertexGlyphFilter()
        self.vgf_pt_0.SetInputData(pt_0)
        self.vgf_pt_0.Update()
        mapper_pt_0 = vtk.vtkPolyDataMapper()
        mapper_pt_0.SetInputConnection(self.vgf_pt_0.GetOutputPort())
        actor_pt_0 = vtk.vtkActor()
        actor_pt_0.SetMapper(mapper_pt_0)
        actor_pt_0.GetProperty().RenderPointsAsSpheresOn()
        actor_pt_0.GetProperty().SetPointSize(10)
        actor_pt_0.GetProperty().SetColor(cm.turbo(0.5)[:3])
        
        # Renderer and interactor
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(actor_pt_0)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkWidget.GetRenderWindow().AddObserver("ModifiedEvent", 
                                                     self.
                                                     on_modified_renderwindow)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.Initialize()
        self.iren.SetInteractorStyle(style)
        self.iren.Start()
    
    def on_key(self, key):
        """
        Classify the point according to the key pressed

        Parameters
        ----------
        key : int
            int for the key that was pressed.

        Returns
        -------
        None.

        """
        
        if key==71:
            # user pressed 'g', classify as ground
            self.on_ground_button_click(1)
        elif key==70:
            # user pressed 'f', classify as snowflake
            self.on_flake_button_click(1)
        elif key==68:
            # user pressed 'd', classify as other
            self.on_other_button_click(1)
    
    def on_v_edit(self):
        """
        When one of the value boxes is edited update the color limits.

        Returns
        -------
        None.

        """
        if float(self.v_min.text())<float(self.v_max.text()):
            self.mapper.SetLookupTable(
                mplcmap_to_vtkLUT(float(self.v_min.text()),
                                        float(self.v_max.text())))
            self.mapper.SetScalarRange(
                float(self.v_min.text()), float(self.v_max.text()))
            self.vtkWidget.GetRenderWindow().Render()
    
    def on_clip_changed(self):
        """
        Change the clipping planes for the camera

        Returns
        -------
        None.

        """
        
        self.renderer.GetActiveCamera().SetClippingRange(
            float(self.near_label.text()), float(self.far_label.text()))
        self.vtkWidget.GetRenderWindow().Render()
        
    def reset_focus(self, s):
        """
        Reset camera focal point to current point.

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """
        
        self.renderer.GetActiveCamera().SetFocalPoint(self.vgf_pt_0.GetInput()
                                                      .GetPoint(0))
        self.vtkWidget.GetRenderWindow().Render()
    
    def on_see_class_button_click(self, s):
        """
        Show the class of the current point in a pop up window

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """
        
        pdata = self.ss.transformFilter.GetOutput()
        ind = vtk_to_numpy(pdata.GetPointData().GetArray('PointId'))==int(
            self.pointid_lineedit.text())
        category = vtk_to_numpy(
            pdata.GetPointData().GetArray('Classification'))[ind]
        
        msg = Qt.QMessageBox()
        msg.setText(str(category))
        msg.exec_()
        
    
    def on_rseed_changed(self, str_seed):
        """
        Create random number generator

        Parameters
        ----------
        str_seed : str
            Random seed as a string.

        Returns
        -------
        None.

        """
        
        self.rng = np.random.default_rng(int(str_seed))
    
    def on_sel_scan_area_button_click(self, s):
        """
        Open file dialog to select scan area directory

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """
        
        self.proj_dialog.exec_()
        
    def on_scan_area_selected(self, dir_str):
        """
        Load the selected scan area and enable project selection

        Parameters
        ----------
        dir_str : str
            Path that the user selected.

        Returns
        -------
        None.

        """
        
        # Parse project path
        self.project_path = dir_str
        
        project_names = os.listdir(dir_str)
        
        # Update proj_combobox with available scans
        self.proj_combobox.addItems(project_names)
        self.proj_combobox.setEnabled(1)
        
        # Enable sel_proj_button and scan combobox
        self.sel_proj_button.setEnabled(1)
        self.scan_combobox.setEnabled(1)
        
    def on_proj_change(self, project_name):
        """
        Place all available scan positions in the scan_combobox.

        Parameters
        ----------
        project_name : str
            Project name currently in the combobox

        Returns
        -------
        None.

        """
        
        # Clear any existing scan names
        self.scan_combobox.clear()
        
        # Get the available scan positions (ones we've saved data for)
        scan_names = os.listdir(os.path.join(self.project_path, project_name,
                                             'npyfiles'))
        self.scan_combobox.addItems(scan_names)
        
    def on_sel_proj_button_click(self, s):
        """
        Load requested scan

        Parameters
        ----------
        s : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # If we were previously rendering a scan, delete it
        if hasattr(self, 'ss'):
            del self.ss
        
        # Load scan apply sop
        self.ss = SingleScan(self.project_path, 
                             self.proj_combobox.currentText(),
                             self.scan_combobox.currentText(), import_mode=
                             'read_scan', class_list='all', las_fieldnames=
                             ['Points', 'Classification', 'PointId',
                              'ReturnIndex'], 
                              class_suffix=self.class_lineedit.text())
        self.ss.add_sop()
        self.ss.apply_transforms(['sop'])
        
        # Load manual classification
        self.ss.load_man_class()
        
        # Create dictionary to hold classified points 
        self.classified_dict = {np.uint8(2): [],
                                np.uint8(65): [],
                                np.uint8(73): []
                                }
        
        # Enable range boxes
        self.v_min.setEnabled(1)
        self.v_max.setEnabled(1)
        
        # Go to the first point
        self.next_point(1)
    
    def goto_point(self, s):
        """
        Set the point given by the PointId as the focal point and zoom to it.

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """
        
        if hasattr(self, 'actor'):
            self.renderer.RemoveActor(self.actor)
            del self.actor
            del self.mapper
        
        # Get requested point coordinates
        pdata = self.ss.transformFilter.GetOutput()
        ind = vtk_to_numpy(pdata.GetPointData().GetArray('PointId'))==int(
            self.pointid_lineedit.text())
        pt = vtk_to_numpy(pdata.GetPoints().GetData())[ind, :].squeeze()
        cts = 0
        for c in self.classified_dict:
            cts += len(self.classified_dict[c])
        self.return_label.setText(str(cts))
        
        # Limit pointcloud to cylinder 10 m around point
        cyl = vtk.vtkCylinder()
        cyl.SetCenter(pt)
        cyl.SetRadius(10)
        cyl.SetAxis(0, 0, 1)
        extractPoints = vtk.vtkExtractPoints()
        extractPoints.SetImplicitFunction(cyl)
        extractPoints.SetInputConnection(
            self.ss.transformFilter.GetOutputPort())
        extractPoints.Update()
        # Create mapper, actor and render
        vgf_ss = vtk.vtkVertexGlyphFilter()
        vgf_ss.SetInputConnection(extractPoints.GetOutputPort())
        vgf_ss.Update()
        elevFilter = vtk.vtkSimpleElevationFilter()
        elevFilter.SetInputConnection(vgf_ss.GetOutputPort())
        elevFilter.Update()
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(elevFilter.GetOutputPort())
        self.mapper.ScalarVisibilityOn()
        self.mapper.SetLookupTable(mplcmap_to_vtkLUT(float(self.v_min.text()),
                                                     float(self.v_max.text())))
        self.mapper.SetScalarRange(
                        float(self.v_min.text()), float(self.v_max.text()))
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().RenderPointsAsSpheresOn()
        self.actor.GetProperty().SetPointSize(5)
        self.renderer.AddActor(self.actor)
        self.vtkWidget.GetRenderWindow().Render()
        
        # Set highlighted point to be so
        pts0 = vtk.vtkPoints()
        pts0.SetNumberOfPoints(1)
        pts0.SetPoint(0, pt[0], pt[1], pt[2])
        pt_0 = vtk.vtkPolyData()
        pt_0.SetPoints(pts0)
        self.vgf_pt_0.SetInputData(pt_0)
        self.vgf_pt_0.Update()
        
        # Update color window
        self.v_min.setText(str(pt[2]-0.01))
        self.v_max.setText(str(pt[2]+0.01))
        
        # Update focal point
        self.renderer.GetActiveCamera().SetFocalPoint(pt)
        self.renderer.GetActiveCamera().SetPosition(pt[0], pt[1], pt[2]+1)
        
        
        self.on_v_edit()
        
    def next_point(self, s):
        """
        Select the next point randomly based on the current inputs
        
        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """
        
        pdata = self.ss.transformFilter.GetOutput()
        # First determine if next point will be what our filter classified
        # as a snowflake or not
        if self.rng.random()<float(self.prop_lineedit.text()):
            # Pick a point classified as a snowflake
            point_id = self.rng.choice(
                vtk_to_numpy(pdata.GetPointData().GetArray('PointId'))[
                    np.nonzero(vtk_to_numpy(pdata.GetPointData()
                                           .GetArray('Classification'))==65)])
        else:
            # Pick a point classified as ground
            point_id = self.rng.choice(
                vtk_to_numpy(pdata.GetPointData().GetArray('PointId'))[
                    np.nonzero(vtk_to_numpy(pdata.GetPointData()
                                           .GetArray('Classification'))==0)])
        self.pointid_lineedit.setText(str(point_id))
        self.goto_point(1)
    
    def on_ground_button_click(self, s):
        """
        Classify the current point as ground and move to the next random point

        Parameters
        ----------
        s : int
            Button status. Not used

        Returns
        -------
        None.

        """
        
        self.classified_dict[np.uint8(2)].append(
            int(self.pointid_lineedit.text()))
        self.last_class = np.uint8(2)
        self.next_point(1)
        self.sel_proj_button.setEnabled(0)
        self.undo_button.setEnabled(1)
        
    def on_flake_button_click(self, s):
        """
        Classify the current point as flake and move to the next random point

        Parameters
        ----------
        s : int
            Button status. Not used

        Returns
        -------
        None.

        """
        
        self.classified_dict[np.uint8(65)].append(
            int(self.pointid_lineedit.text()))
        self.last_class = np.uint8(65)
        self.next_point(1)
        self.sel_proj_button.setEnabled(0)
        self.undo_button.setEnabled(1)
    
    def on_other_button_click(self, s):
        """
        Classify the current point as other and move to the next random point

        Parameters
        ----------
        s : int
            Button status. Not used

        Returns
        -------
        None.

        """
        
        self.classified_dict[np.uint8(73)].append(
            int(self.pointid_lineedit.text()))
        self.last_class = np.uint8(73)
        self.next_point(1)
        self.sel_proj_button.setEnabled(0)
        self.undo_button.setEnabled(1)
    
    def on_undo_button_click(self, s):
        """
        Undo last manual classification and go back to that point

        Parameters
        ----------
        s : int
            Button status. Not used

        Returns
        -------
        None.

        """
        
        last_id = self.classified_dict[self.last_class].pop()
        self.undo_button.setEnabled(0)
        self.pointid_lineedit.setText(str(last_id))
        self.goto_point(1)
        
    
    def on_save_button_click(self, s):
        """
        Save the classified points into the manual classification table.

        Parameters
        ----------
        s : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        for category in self.classified_dict:
            if len(self.classified_dict[category])==0:
                continue
            pointids = np.array(self.classified_dict[category], 
                                dtype=np.uint32)
            # Get the points corresponding to the saved pedigree ids
            pedigreeIds = vtk.vtkTypeUInt32Array()
            pedigreeIds.SetNumberOfComponents(1)
            pedigreeIds.SetNumberOfTuples(pointids.size)
            np_pedigreeIds = vtk_to_numpy(pedigreeIds)
            if np.max(pointids)>np.iinfo(np.uint32).max:
                raise RuntimeError('PointId exceeds size of uint32')
            np_pedigreeIds[:] = pointids
            pedigreeIds.Modified()
            # Use PedigreeId selection to get points
            selectionNode = vtk.vtkSelectionNode()
            selectionNode.SetFieldType(1) # we want to select points
            selectionNode.SetContentType(2) # 2 corresponds to pedigreeIds
            selectionNode.SetSelectionList(pedigreeIds)
            selection = vtk.vtkSelection()
            selection.AddNode(selectionNode)
            extractSelection = vtk.vtkExtractSelection()
            extractSelection.SetInputData(0, 
                                          self.ss.transformFilter.GetOutput())
            extractSelection.SetInputData(1, selection)
            extractSelection.Update()
            vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
            vertexGlyphFilter.SetInputConnection(
                extractSelection.GetOutputPort())
            vertexGlyphFilter.Update()
            pdata = vertexGlyphFilter.GetOutput()
            # Update man_class table
            self.ss.update_man_class(pdata, category, 
                                     user=str(self.user_lineedit.text()))
        
        self.sel_proj_button.setEnabled(1)
    
    ### VTK Methods
    def on_modified_renderwindow(self, obj, event):
        """
        When the renderwindow is modified, update near and far clipping
        plane labels and distance to focal point

        Parameters
        ----------
        obj : TYPE
            DESCRIPTION.
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Get current clipping distances
        cam = self.renderer.GetActiveCamera()
        clipping_dists = np.array(cam.GetClippingRange())
        foc_dist = cam.GetDistance()
        
        # If the near clipping dist is > XX multiple of focal distance
        # if clipping_dists[0]>(0.25*foc_dist):
        #     clipping_dists[0] = 0.25*foc_dist
        #     clipping_dists[1] = 100*clipping_dists[0]
        #     cam.SetClippingRange(clipping_dists)
        #     self.vtkWidget.GetRenderWindow().Render()
        
        # update labels
        self.near_label.setText(str(clipping_dists[0]))
        self.far_label.setText(str(clipping_dists[1]))
        self.focal_distance_label.setText('Foc. Dist: ' 
                                          + str(round(foc_dist, 2)))
        
# Other useful functions
def mplcmap_to_vtkLUT(vmin, vmax, name='turbo', N=256, 
                      color_under='fuchsia', color_over='white'):
    """
    Create a vtkLookupTable from a matplotlib colormap.

    Parameters
    ----------
    vmin : float
        Minimum value for colormap.
    vmax : float
        Maximum value for colormap.
    name : str, optional
        Matplotlib name of the colormap. The default is 'rainbow'
    N : int, optional
        Number of levels in colormap. The default is 256.
    color_under : str, optional
        Color for values less than vmin, should be in vtkNamedColors. 
        The default is 'fuchsia'.
    color_over : str, optional
        Color for values greater than vmax, should be in vtkNamedColors. 
        The default is 'white'.

    Returns
    -------
    vtkLookupTable

    """

    # Pull the matplotlib colormap
    mpl_cmap = cm.get_cmap(name, N)
    
    # Create Lookup Table
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(vmin, vmax)
    
    # Add Colors from mpl colormap
    lut.SetNumberOfTableValues(N)
    for (i, c) in zip(np.arange(N), mpl_cmap(range(N))):
        lut.SetTableValue(i, c)

    # Add above and below range colors
    nc = vtk.vtkNamedColors()
    lut.SetBelowRangeColor(nc.GetColor4d(color_under))
    lut.SetAboveRangeColor(nc.GetColor4d(color_over))

    lut.Build()
    
    return lut


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())