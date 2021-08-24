#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:46:14 2021

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

for scan_name in scan_names:
    # Read in scan
    print(scan_name)
    ss = SingleScan(project_path, project_name, scan_name, 
                        import_mode='read_scan', class_list='all')
    ss.load_man_class()
    ss.man_class.query('class_suffix != "_radius"', inplace=True)
    # Write to file to save
    ss.man_class.to_parquet(os.path.join(ss.project_path, 
                                              ss.project_name, 
                                              'manualclassification', 
                                              ss.scan_name + '.parquet'),
                                              engine="pyarrow", 
                                              compression=None)