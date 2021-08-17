#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manually_remove_manmade.py

Manually remove manmade objects from scan. This is just a convenience because
we care about how well the filter removes snowflakes from a natural scene.

Created on Tue Aug  17 16:50:00 2021

@author: David Clemens-Sewall
"""

import numpy as np
import os
import json
import time
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# Set what category we want to list these points as. 
category = 73

project_path = '../data'

project_name = 'mosaic_rov_220220.RiSCAN.RiSCAN'

# %% Load all areapoints into a dictionary structure.

areapoint_dict = {}

filenames = os.listdir(os.path.join(project_path, project_name, 
                                    'manualclassification'))
for filename in filenames:
    if filename[-4:]=='.txt':
        if not project_name in areapoint_dict:
            areapoint_dict[project_name] = {}
            
        f = open(os.path.join(project_path, project_name, 
                                    'manualclassification', filename), 'r')
        areapoint_dict[project_name][filename] = json.load(f)[project_name]
        f.close()

# %% Get the cornercoords for each set of areapoints

cornercoord_dict = {}

for project_name in areapoint_dict:
    print(project_name)
    cornercoord_dict[project_name] = {}
    # Load scan and transform
    project = pydar.Project(project_path, project_name, 
                            import_mode='read_scan', class_list='all')
    project.apply_transforms(['sop'])
    
    for areapoint_name in areapoint_dict[project_name]:
        print(areapoint_name)
        # apply manual filter
        cornercoords = project.areapoints_to_cornercoords(areapoint_dict
                                                          [project_name]
                                                          [areapoint_name])
        project.apply_manual_filter(cornercoords, category=category)


# Write the scan to a file
project.write_scans()



del project
