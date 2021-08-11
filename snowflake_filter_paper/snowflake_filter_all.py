#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snowflake_filter_all.py

Filter snowflakes out of all TLS scan positions using the proposed filter.

Created on Mon Aug  9 15:26:19 2021

@author: David Clemens-Sewall
"""

# Imports
import os
import sys
from math import sqrt, pi
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
z_max = 1.75
cylinder_rad = 0.025*sqrt(2)*pi/180 # TLS point spacing from scanners 
                                    # perspective in radians for this scan
radial_precision = 0.005 # Riegl VZ1000 range precision in m
leafsize = 100
z_std_mult = 3.5

for scan_name in scan_names:
    # Load single scan
    ss = SingleScan(project_path, project_name, scan_name, 
                    import_mode='import_las', create_id=True, class_list=[0])
    
    # Load the Scanner's Own Positions, this provides leveling and puts all
    # scans in a common reference frame (so we can use just a single z_max)
    ss.add_sop()
    ss.apply_transforms(['sop'])
    
    # Apply elevation filtering, here the scanner (z=0) is higher than the 
    # surface everywhere.
    ss.apply_elevation_filter(z_max)
    
    # Apply 'visible region' snowflake filter (named return index)
    ss.apply_snowflake_filter_returnindex(cylinder_rad=cylinder_rad,
                                          radial_precision=radial_precision)
    
    # Apply vertical standard deviation filter 
    ss.apply_snowflake_filter_3(z_std_mult, leafsize)
    
    # Save scans to numpy files (makes for quicker loading)
    ss.write_scan()
    
    del ss