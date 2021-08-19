#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
early_filter_all.py

Apply early return filtering to all scan positions. Save output.

Created on Wed Aug 18 16:19:32 2021

@author: David Clemens-Sewall
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

for scan_name in scan_names:
    # Read in scan
    ss = SingleScan(project_path, project_name, scan_name, 
                        import_mode='read_scan', class_list=[0])
    # Wipe all classifications except manual ones
    ss.clear_classification(ignore_list=[73])
    # apply early return filter
    ss.apply_early_return_filter()
    ss.write_classification_suffix('_early')