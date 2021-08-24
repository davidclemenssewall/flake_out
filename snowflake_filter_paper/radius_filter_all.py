#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radius_filter_all.py

Apply radius outlier removal filter to all scan positions. Save output.

Created on Thu Aug 19 00:47:11 2021

@author: thayer
"""

# Imports
import os
import sys
# this code block is just for development, could be omitted
try:
    from flake_out.single_scan import SingleScan
except ModuleNotFoundError:
    print('flake_out not found, importing from local directory')
    sys.path.append(os.path.join('..', 'src'))
    from flake_out.single_scan import SingleScan

# Specify single scan to read in
# relative path to folder containing TLS data
project_path = os.path.join('..', 'data')
project_name = 'mosaic_rov_220220.RiSCAN.RiSCAN'
scan_names = ['ScanPos001',
              'ScanPos002',
              'ScanPos003',
              'ScanPos004',
              'ScanPos005',
              'ScanPos006',
              'ScanPos007',
              'ScanPos008',
              'ScanPos009',
              ]

# Parameters
nb_points = 4
radius = 0.14

for scan_name in scan_names:
    # Read in scan
    print(scan_name)
    ss = SingleScan(project_path, project_name, scan_name, 
                        import_mode='read_scan', class_list=[0])
    # Wipe all classifications except manual ones
    ss.clear_classification(ignore_list=[73])
    # apply radius filter
    ss.apply_radius_outlier_filter(nb_points, radius)
    ss.write_classification_suffix('_radius')