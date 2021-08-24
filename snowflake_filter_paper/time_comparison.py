#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:26:19 2021

@author: thayer
"""

# Imports
import time
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

# Parameters
nb_points = 4
radius = 0.14
z_max = 1.75
cylinder_rad = 0.025*sqrt(2)*pi/180 # TLS point spacing from scanners 
                                    # perspective in radians for this scan
radial_precision = 0.005 # Riegl VZ1000 range precision in m
leafsize = 100
z_std_mult = 3.5

# Read in scan
ss = SingleScan(project_path, project_name, scan_name, 
                    import_mode='read_scan', class_list=[0])
ss.add_sop()
ss.apply_transforms(['sop'])

# Wipe all classifications except manual ones
ss.clear_classification(ignore_list=[73])
# apply filter
t0 = time.perf_counter()
ss.apply_elevation_filter(z_max)
ss.apply_snowflake_filter_returnindex(cylinder_rad=cylinder_rad,
                                      radial_precision=radial_precision)
ss.apply_snowflake_filter_3(z_std_mult, leafsize)
t1 = time.perf_counter()
print('flake out time')
print(t1-t0)

# Wipe all classifications except manual ones
ss.clear_classification(ignore_list=[73])
# apply filter
t2 = time.perf_counter()
ss.apply_early_return_filter()
t3 = time.perf_counter()
print('Early time')
print(t3-t2)

# Wipe all classifications except manual ones
ss.clear_classification(ignore_list=[73])
# apply filter
t4 = time.perf_counter()
ss.apply_radius_outlier_filter(nb_points, radius)
t5 = time.perf_counter()
print('Radius time')
print(t5-t4)


