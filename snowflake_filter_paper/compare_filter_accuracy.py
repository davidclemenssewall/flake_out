#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_filter_accuracy.py

Compare the results of the filter with manual classifications.

Created on Tue Aug 10 14:02:48 2021

@author: David Clemens-Sewall
"""

# Imports
import os
import sys
import pandas as pd
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
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

# Read in each scan and create dataframe
df_list = []
counts_dict = {0: [],
               64: [],
               65: [],
               73: []}
for scan_name in scan_names:
    # Load scan
    ss = SingleScan(project_path, project_name, scan_name, 
                    import_mode='read_scan', 
                    las_fieldnames=['Points', 'PointId', 'Classification',
                                    'Classification_early'],
                    class_list='all')
    ss.load_man_class()
    df = ss.man_class.copy(deep=True)
    df['scan_name'] = scan_name
    # Create dataframe from classification field in ss to join to
    df_ss = pd.DataFrame(data={'vis_region_class':
                               vtk_to_numpy(ss.polydata_raw.GetPointData().
                                           GetArray('Classification')),
                               'early_class':
                               vtk_to_numpy(ss.polydata_raw.GetPointData().
                                           GetArray('Classification_early'))
                               },
                         index=vtk_to_numpy(ss.polydata_raw.GetPointData().
                                            GetArray('PointId')))
    df_list.append(pd.concat([df, df_ss], axis=1, join='inner'))
    df_list[-1].set_index('scan_name', append=True, inplace=True)
    
    # Get counts of each class type
    uni, cts = np.unique(vtk_to_numpy(ss.polydata_raw.GetPointData().
                                           GetArray('Classification')),
                         return_counts=True)
    for u in [0, 64, 65, 73]:
        if not u in uni:
            uni = np.append(uni, u)
            cts = np.append(cts, 0)
    for u, c in zip(uni, cts):
        counts_dict[u].append(c)
    
    del ss

df = pd.concat(df_list)
df_cts = pd.DataFrame(data=counts_dict, index=scan_names)

# Get the numbers of true positives, false positives, true negatives and
# false negatives
tp = np.logical_and(df['Classification'].values==65, 
                    np.isin(df['vis_region_class'].values, [64, 65])).sum()
fp = np.logical_and(df['Classification'].values==2, 
                    np.isin(df['vis_region_class'].values, [64, 65])).sum()
tn = np.logical_and(df['Classification'].values==2, 
                    df['vis_region_class'].values==0).sum()
fn = np.logical_and(df['Classification'].values==65, 
                    df['vis_region_class'].values==0).sum()

# tp = np.logical_and(df['Classification'].values==65, 
#                     np.isin(df['early_class'].values, [64, 65])).sum()
# fp = np.logical_and(df['Classification'].values==2, 
#                     np.isin(df['early_class'].values, [64, 65])).sum()
# tn = np.logical_and(df['Classification'].values==2, 
#                     df['early_class'].values==0).sum()
# fn = np.logical_and(df['Classification'].values==65, 
#                     df['early_class'].values==0).sum()

print('Manually Classified Points: ')
print('True Positives: %.0f' % tp)
print('False Positives: %.0f' % fp)
print('True Negatives: %.0f' % tn)
print('False Negatives: %.0f' % fn)
print('sensitivity: %.3f' % (tp/(tp+fn)))
print('specificity: %.3f' % (tn/(tn+fp)))
print('precision: %.3f' % (tp/(tp+fp)))
print('recall: %.3f' % (tp/(tp+fn)))
print('false positive rate: %.3f' % (fp/(fp+tn)))
print('false negative rate: %.3f' % (fn/(fn+tp)))

print('\nAll Points:')
print('Classified As Snowflakes: %.0f' % df_cts[65].sum())
print('Not Classified As Snowflakes: %.0f' % df_cts[0].sum())
print('Max. false positive rate: %.5f' % (df_cts[65].sum()/(df_cts[65].sum() 
                                                + df_cts[0].sum())))

# Need to use the importance sampling distribution for each filter!
# otherwise get inaccurate false positive rate results!!!

# %% Incorporating Matt's work
q_s = 0.5

df_cts['Pk_over_Qk_65'] = 1/(q_s*(df_cts[0]+df_cts[65])/df_cts[65])
df_cts['Pk_over_Qk_0'] = 1/((1-q_s)*(df_cts[0]+df_cts[65])/df_cts[0])


