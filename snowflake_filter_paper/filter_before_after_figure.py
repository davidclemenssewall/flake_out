#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_before_after_figure.py

Created on Tue Aug 24 17:43:08 2021

@author: thayer
"""

# Imports
import os
import sys
import vtk
# this code block is just for development, could be omitted
try:
    from flake_out.single_scan import SingleScan
except ModuleNotFoundError:
    print('flake_out not found, importing from local directory')
    sys.path.append(os.path.join('..', 'src'))
    from flake_out.single_scan import SingleScan
    from flake_out.manual_snowflake_classifier import mplcmap_to_vtkLUT

# Specify single scan to read in
# relative path to folder containing TLS data
project_path = os.path.join('..', 'data')
project_name = 'mosaic_rov_220220.RiSCAN.RiSCAN'
scan_name = 'ScanPos001'

# Load scan
ss = SingleScan(project_path, project_name, scan_name, 
                    import_mode='read_scan', 
                    las_fieldnames=['Points', 'PointId', 'Classification']
                    , class_list=[0])
ss.add_sop()
ss.apply_transforms(['sop'])

# %% Begin by finding a good viewpoint to see snowflakes from
v_min = -3.5
v_max = -2.5

# Define function for writing the camera position and focal point to
# std out when the user presses 'u'
def cameraCallback(obj, event):
    print("Camera Pos: " + str(obj.GetRenderWindow().
                                   GetRenderers().GetFirstRenderer().
                                   GetActiveCamera().GetPosition()))
    print("Focal Point: " + str(obj.GetRenderWindow().
                                    GetRenderers().GetFirstRenderer().
                                    GetActiveCamera().GetFocalPoint()))
    print("Roll: " + str(obj.GetRenderWindow().
                                    GetRenderers().GetFirstRenderer().
                                    GetActiveCamera().GetRoll()))

vgf_ss = vtk.vtkVertexGlyphFilter()
vgf_ss.SetInputConnection(ss.transformFilter.GetOutputPort())
vgf_ss.Update()
elevFilter = vtk.vtkSimpleElevationFilter()
elevFilter.SetInputConnection(vgf_ss.GetOutputPort())
elevFilter.Update()
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(elevFilter.GetOutputPort())
mapper.ScalarVisibilityOn()
mapper.SetLookupTable(mplcmap_to_vtkLUT(v_min,v_max))
mapper.SetScalarRange(v_min, v_max)
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().RenderPointsAsSpheresOn()
actor.GetProperty().SetPointSize(10)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

# Create RenderWindow and interactor, set style to trackball camera
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(500, 500)
renderWindow.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

style = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(style)
    
iren.Initialize()
renderWindow.Render()

iren.AddObserver('UserEvent', cameraCallback)
iren.Start()

# %% create before filter snapshot
v_min = -2.82
v_max = -2.72

camera_pos = (-88.27284833010232, -221.3828089106876, -2.432749478881751)
focal_point = (-86.50635058648565, -220.8674241168494, -2.6169033090310494)
roll = 87.59836036603093

camera_pos = (-100.31426700454924, -225.85534288739854, -2.533032850954018)
focal_point = (-106.89571786604078, -230.06264150638754, -2.737415427529452)
roll = -92.34638327100336

vgf_ss = vtk.vtkVertexGlyphFilter()
vgf_ss.SetInputConnection(ss.transformFilter.GetOutputPort())
vgf_ss.Update()
elevFilter = vtk.vtkSimpleElevationFilter()
elevFilter.SetInputConnection(vgf_ss.GetOutputPort())
elevFilter.Update()
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(elevFilter.GetOutputPort())
mapper.ScalarVisibilityOn()
mapper.SetLookupTable(mplcmap_to_vtkLUT(v_min,v_max))
mapper.SetScalarRange(v_min, v_max)
actor = vtk.vtkActor()
actor.SetMapper(mapper)
#actor.GetProperty().RenderPointsAsSpheresOn()
actor.GetProperty().SetPointSize(5)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

# Create RenderWindow and interactor, set style to trackball camera
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(500, 500)
renderWindow.AddRenderer(renderer)

# Create Camera
camera = vtk.vtkCamera()
camera.SetFocalPoint(focal_point)
camera.SetPosition(camera_pos)
camera.SetRoll(roll)
renderer.SetActiveCamera(camera)

renderWindow.Render()
        
# Screenshot image to save
w2if = vtk.vtkWindowToImageFilter()
w2if.SetInput(renderWindow)
w2if.Update()

writer = vtk.vtkPNGWriter()
writer.SetFileName(os.path.join('figures', 'before_filt.png'))
writer.SetInputData(w2if.GetOutput())
writer.Write()

renderWindow.Finalize()
del renderWindow

# %% After snapshot

vgf_ss = vtk.vtkVertexGlyphFilter()
vgf_ss.SetInputConnection(ss.currentFilter.GetOutputPort())
vgf_ss.Update()
elevFilter = vtk.vtkSimpleElevationFilter()
elevFilter.SetInputConnection(vgf_ss.GetOutputPort())
elevFilter.Update()
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(elevFilter.GetOutputPort())
mapper.ScalarVisibilityOn()
mapper.SetLookupTable(mplcmap_to_vtkLUT(v_min,v_max))
mapper.SetScalarRange(v_min, v_max)
actor = vtk.vtkActor()
actor.SetMapper(mapper)
#actor.GetProperty().RenderPointsAsSpheresOn()
actor.GetProperty().SetPointSize(5)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

# Create RenderWindow and interactor, set style to trackball camera
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(500, 500)
renderWindow.AddRenderer(renderer)

# Create Camera
camera = vtk.vtkCamera()
camera.SetFocalPoint(focal_point)
camera.SetPosition(camera_pos)
camera.SetRoll(roll)
renderer.SetActiveCamera(camera)

renderWindow.Render()
        
# Screenshot image to save
w2if = vtk.vtkWindowToImageFilter()
w2if.SetInput(renderWindow)
w2if.Update()

writer = vtk.vtkPNGWriter()
writer.SetFileName(os.path.join('figures', 'after_filt.png'))
writer.SetInputData(w2if.GetOutput())
writer.Write()

renderWindow.Finalize()
del renderWindow
