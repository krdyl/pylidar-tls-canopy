import os
import numpy as np
import glob

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator

from pylidar_tls_canopy import riegl_io, plant_profile, grid

import pandas as pd
import openpyxl
from pathlib import Path
import shutil
import timeit
start = timeit.timeit()

os.chdir(f'/mnt/c/users/kdayal/Documents/2_data/1_Bosland/4_TLS/2_benchmark/VZ400i/2024-04-08-BOSLAND-1i/2024-04-08-BOSLAND-1i.RiSCAN/')
cwd = os.getcwd()
scans = os.listdir('/mnt/c/users/kdayal/Documents/2_data/1_Bosland/4_TLS/2_benchmark/VZ400i/2024-04-08-BOSLAND-1i/2024-04-08-BOSLAND-1i.RiSCAN/SCANS')
scans_df = pd.DataFrame(scans)


def get_filepaths(data):
    pth = cwd+"/project.rdb/SCANS/"+data+"/SINGLESCANS/"
    dirs = os.listdir(pth)
    files_rdb = (glob.glob(os.path.join(pth, dirs[0],"*.rdbx")))
    
    pth = cwd+"/SCANS/"+data+"/SINGLESCANS/"
    files_rxp = glob.glob(os.path.join(pth,"*.rxp"))
    files_rxp = [i for i in files_rxp if "residual" not in i]
    
    pth = os.path.join(cwd,"Matrices","V2")
    fname = data+".DAT"
    files_sop = glob.glob(os.path.join(pth, fname))
    
    return([files_rdb[0], files_rxp[0], files_sop[0]])
    #return(pd.DataFrame[files_rxp, files_rdb, files_sop])




paths_v = scans_df[0:9].map(get_filepaths)
paths_v = pd.DataFrame(paths_v.iloc[:,0].tolist(), columns = ["rdb_v", "rxp_v", "dat_v"])
paths_v = pd.concat([scans_df[0:9], paths_v], axis=1)

    
paths_h = scans_df[9:18].map(get_filepaths).reset_index(drop=True)
paths_h = pd.DataFrame(paths_h.iloc[:,0].tolist(), columns = ["rdb_h", "rxp_h", "dat_h"])
paths_h = pd.concat([scans_df[9:18].reset_index(drop=True), paths_h], axis=1)

paths = pd.concat([paths_v, paths_h], axis = 1)



scan_meta = pd.DataFrame(data = {'freq': [300, 300, 300, 600, 600, 600, 1200, 1200, 1200],
                                 'res' : [0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.06, 0.06, 0.06]})

scan_info = pd.concat([paths, scan_meta], axis =1)

scan_info['rdb_v'][0]
scan_info['rxp_v'][0]
scan_info['dat_v'][0]

scan_info['rdb_h'][0]
scan_info['rxp_h'][0]
scan_info['dat_h'][0]


test_df = scan_info.iloc[8:9,:]

def get_plantprofiles(scans):
    
    upright_rdbx_fn = scans['rdb_v']
    upright_rxp_fn = scans['rxp_v']
    upright_transform_fn = scans['dat_v']
    

    tilt_rdbx_fn = scans['rdb_h']
    tilt_rxp_fn = scans['rxp_h']
    tilt_transform_fn = scans['dat_h']
    print("done1")
    # Determine the origin coordinates to use
    transform_matrix = riegl_io.read_transform_file(upright_transform_fn)
    x0,y0,z0,_ = transform_matrix[3,:]

    grid_extent = 60
    grid_resolution = 1
    grid_origin = [x0,y0]
    

    # If using RXP files only as input, set rxp to True:
    x,y,z,r = plant_profile.get_min_z_grid([upright_rdbx_fn], 
                                           [upright_transform_fn], 
                                        grid_extent, grid_resolution, grid_origin=grid_origin,
                                        rxp=False)
    
    # Optional weighting of points by 1 / range
    planefit = plant_profile.plane_fit_hubers(x, y, z, w=1/r)
    #planefit['Summary']
    
    print("done2")
    # If the ground plane is not defined then set ground_plane to None
    # and use the sensor_height argument whe adding scan positions
    vpp = plant_profile.Jupp2009(hres=0.5, zres=5, ares=90, 
                                min_z=35, max_z=70, min_h=0, max_h=50,
                                ground_plane=planefit['Parameters'])

    # If using RXP files only as input, set rdbx_file to None (the default)
    query_str = ['reflectance > -20']
    vpp.add_riegl_scan_position(upright_rxp_fn, upright_transform_fn, sensor_height=None,
        rdbx_file=upright_rdbx_fn, method='WEIGHTED', min_zenith=35, max_zenith=70,
        query_str=query_str)
    
    print("done3")

    # If using RXP files only as input, set rdbx_file to None (the default)
    query_str = ['reflectance > -20']
    vpp.add_riegl_scan_position(tilt_rxp_fn, tilt_transform_fn, sensor_height=None,
        rdbx_file=tilt_rdbx_fn, method='WEIGHTED', min_zenith=5, max_zenith=35,
        query_str=query_str)
    
    vpp.get_pgap_theta_z(min_azimuth=0, max_azimuth=360)
    weighted_pai = vpp.calcSolidAnglePlantProfiles()
    weighted_pavd = vpp.get_pavd(weighted_pai)

    #weighted_pai= pd.DataFrame(weighted_pai)
    #weighted_pai.to_csv(str(up)+'pai'+'.csv', sep=',', index=False, encoding='utf-8')

    #weighted_pavd= pd.DataFrame(weighted_pavd)
    #weighted_pavd.to_csv(str(up)+'pavd'+'.csv', sep=',', index=False, encoding='utf-8')
    
    return(weighted_pai)

pai = test_df.apply(get_plantprofiles, axis=1)

    



def get_test(scans):

    upright_rdbx_fn = scans['rdb_v']
    upright_rxp_fn = scans['rxp_v']
    upright_transform_fn = scans['dat_v']
    

    tilt_rdbx_fn = scans['rdb_h']
    tilt_rxp_fn = scans['rxp_h']
    tilt_transform_fn = scans['dat_h']
    
    print(upright_transform_fn)

pai = test_df.apply(get_test, axis=1)


test_df.iloc[0]['rdb_v']  