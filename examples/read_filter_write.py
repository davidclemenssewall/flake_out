#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read_filter_write.py

Example script to demonstrate reading a single TLS scan position, 

Created on Mon Aug  9 12:54:23 2021

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
scan_name = 'ScanPos001'

# Load single scan
ss = SingleScan(project_path, project_name, scan_name, 
                import_mode='import_las', create_id=True, class_list=[0])

# Apply elevation filtering, here the scanner (z=0) is higher than the surface
# everywhere.
ss.apply_elevation_filter(0.0)

# Apply 'visible region' snowflake filter (named return index)
cylinder_rad = 0.025*sqrt(2)*pi/180 # TLS point spacing from scanners 
                                    # perspective in radians for this scan
radial_precision = 0.005 # Riegl VZ1000 range precision in m
ss.apply_snowflake_filter_returnindex(cylinder_rad=cylinder_rad,
                                      radial_precision=radial_precision)

# Apply vertical standard deviation filter 
leafsize = 100
z_std_mult = 3.5
ss.apply_snowflake_filter_3(z_std_mult, leafsize)

# Write scan to an las file
ss.write_las_pdal(mode='raw')