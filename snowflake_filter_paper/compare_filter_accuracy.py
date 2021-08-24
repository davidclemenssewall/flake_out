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

# Specify single scans to read in
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

# Specify filter names to use
filter_names = ['', # proposed snowflake filter
                '_early',
                '_radius',
                ]

# currently dcs is the only user, in the future could have multiple
user = 'dcs'

# ratio that was used for classification
q_s = 0.5

list_df = []
for scan_name in scan_names:
    # Load scan
    ss = SingleScan(project_path, project_name, scan_name, 
                    import_mode='read_scan', 
                    las_fieldnames=['Points', 'PointId', 'Classification',
                                    'Classification_early', 
                                    'Classification_radius']
                    , class_list='all')
    
    # Get requested filter and user from man_class table
    ss.load_man_class()
    df = ss.man_class.copy(deep=True)
    df = df.query('user == @user')# and class_suffix == @filter_name')
    df.index = df.index.droplevel('user')#['user', 'class_suffix'])
    df = df[df.Classification.isin([2.0, 65.0])]
    man_class = df['Classification']
    man_class.name = 'man_class'
    
    # Get classification field in ss to join to
    filt_class = pd.DataFrame(data={'': 
                                    vtk_to_numpy(ss.polydata_raw
                                             .GetPointData()
                                             .GetArray('Classification')),
                                    '_early': 
                                    vtk_to_numpy(ss.polydata_raw
                                             .GetPointData()
                                             .GetArray(
                                                 'Classification_early')),
                                    '_radius': 
                                    vtk_to_numpy(ss.polydata_raw
                                             .GetPointData()
                                             .GetArray(
                                                 'Classification_radius'))
                                    }
                           , index=vtk_to_numpy(ss.polydata_raw
                                              .GetPointData()
                                              .GetArray('PointId'))
                           ) #,name='filt_class')
    filt_class.index.rename('PointId', inplace=True)
    filt_class = filt_class.melt(
                         value_vars=['', '_early', '_radius'], 
                         ignore_index=False, var_name='class_approx', 
                         value_name='filt_class')
    #filt_class.set_index('class_approx', append=True, inplace=True)
    
    # Merge to get dataframe from which we'll compute tp, fp, tn, fn
    df_class = pd.merge(man_class.reset_index(), filt_class, how='inner', 
                        on='PointId')
    df_class.set_index(['PointId', 'class_suffix', 'class_approx']
                       , inplace=True)
    
    # Get number of true positives, etc...
    df_class['tp'] = ((df_class.man_class==65.0) 
              & (df_class.filt_class.isin([64, 65])))
    df_class['fp'] = ((df_class.man_class==2.0) 
              & (df_class.filt_class.isin([64, 65])))
    df_class['tn'] = (df_class.man_class==2.0) & (df_class.filt_class==0)
    df_class['fn'] = (df_class.man_class==65.0) & (df_class.filt_class==0)
    df_cts = df_class.groupby(by=['class_suffix', 'class_approx']
                              )[['tp', 'fp', 'tn', 'fn']].sum()
    
    # Get N and N_s for each approximate classifier
    filt_class['N'] = filt_class.filt_class.isin([0, 64, 65])
    filt_class['N_s'] = filt_class.filt_class.isin([64, 65])
    df_N = filt_class.groupby(by='class_approx')[['N', 'N_s']].sum()
    df_N['wt_approx_c_S'] = 1/(q_s * df_N.N / df_N.N_s)
    df_N['wt_approx_c_G'] = 1/(q_s * df_N.N / (df_N.N - df_N.N_s))
    
    # Merge
    df_res = pd.merge(df_cts.reset_index(), df_N, how='inner', 
                      on='class_approx')
    df_res['scan_name'] = scan_name
    list_df.append(df_res)

df = pd.concat(list_df, ignore_index=True)
df.set_index(['scan_name', 'class_suffix', 'class_approx'], inplace=True)

# Compute false postive and false negative rates
df['fpr'] = df.fp * df.wt_approx_c_S / (
    df.tn * df.wt_approx_c_G + df.fp * df.wt_approx_c_S)
df['fnr'] = df.fn * df.wt_approx_c_G / (
    df.tp * df.wt_approx_c_S + df.fn * df.wt_approx_c_G)

df['frac_filt'] = df.N_s/df.N

# Get values for paper
print(df.groupby(['class_suffix', 'class_approx'])[['N', 'N_s', 'frac_filt', 
                                                    'fpr', 'fnr']].mean()
      .query('class_suffix==class_approx'))

print(df.groupby(['class_suffix', 'class_approx'])[['N', 'N_s', 'frac_filt', 
                                                    'fpr', 'fnr']].mean()
      .query('class_suffix==class_approx'))
