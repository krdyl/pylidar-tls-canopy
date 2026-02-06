import numpy as np
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
from pylidar_tls_canopy import riegl_io, plant_profile, plant_profile_2, grid
from os import walk
import pandas as pd
#import openpyxl
from pathlib import Path
import shutil
import math
import riegl_rdb
import ast
import re
from datetime import datetime
import json 

all_paths_df = pd.read_csv("/home/kdayal/projects/pylidar-tls-canopy/results/all_paths_df_20260206.csv")

def get_plantprofiles(scans):

    project = scans['project_path']
    
    upright_rdbx_fn = scans['rdb_v']
    upright_rxp_fn = scans['rxp_v']
    upright_transform_fn = scans['dat_v']
    

    tilt_rdbx_fn = scans['rdb_h']
    tilt_rxp_fn = scans['rxp_h']
    tilt_transform_fn = scans['dat_h']
    
    
    # Determine the origin coordinates to use
    transform_matrix = riegl_io.read_transform_file(upright_transform_fn)
    x0,y0,z0,_ = transform_matrix[3,:]

    grid_extent = 60
    grid_resolution = 1
    grid_origin = [x0,y0]
    
    # If the ground plane is not defined then set ground_plane to None
    # and use the sensor_height argument when adding scan positions
    terrain_params = np.array([scans['parameter_c'], scans['parameter_a'], scans['parameter_b']])

    vpp = plant_profile_2.Jupp2009(hres=0.5, 
                                    zres=5, 
                                    ares=360,
                                    min_z=5, 
                                    max_z=70, 
                                    min_h=0, 
                                    max_h=50,
                                    ground_plane=terrain_params)

    # If using RXP files only as input, set rdbx_file to None (the default)
    query_str = ['reflectance > -20', 'range > 1.5']
    vpp.add_riegl_scan_position(upright_rxp_fn, 
                                upright_transform_fn, 
                                sensor_height=None,
                                rdbx_file=upright_rdbx_fn, 
                                method='WEIGHTED', 
                                min_zenith=35, 
                                max_zenith=70,
                                query_str=query_str)
    

    # If using RXP files only as input, set rdbx_file to None (the default)
    query_str = ['reflectance > -20', 'range > 1.5']
    vpp.add_riegl_scan_position(tilt_rxp_fn, 
                                tilt_transform_fn, 
                                sensor_height=None,
                                rdbx_file=tilt_rdbx_fn, 
                                method='WEIGHTED', 
                                min_zenith=5, 
                                max_zenith=35,
                                query_str=query_str)
    
    vpp.get_pgap_theta_z(min_azimuth=0, max_azimuth=360)
    
    hinge_idx = np.argmin(abs(vpp.zenith_bin - 57.5))

    pgap_phi_z = []
    for az in range(0, 360, vpp.ares):
        # Set invert to True if min_azimuth and max_azimuth specify the range to exclude
        vpp.get_pgap_theta_z(min_azimuth=az, max_azimuth=az+vpp.ares, invert=False)
        pgap_phi_z.append(vpp.pgap_theta_z[hinge_idx])
    
    
    hinge_pai = vpp.calcHingePlantProfiles()
    weighted_pai = vpp.calcSolidAnglePlantProfiles()
    linear_pai = vpp.calcLinearPlantProfiles()

    hinge_pavd = vpp.get_pavd(hinge_pai)
    linear_pavd = vpp.get_pavd(linear_pai)
    weighted_pavd = vpp.get_pavd(weighted_pai)
    
    
    pattern  = json.loads(riegl_rdb.readHeader(upright_rdbx_fn)['riegl.scan_pattern'])
    freq = pattern['rectangular']['program']['name']
    res = round(pattern['rectangular']['phi_increment'], 2)
    
    vrdbf = os.path.split(upright_rdbx_fn)[1]
    vrxpf = os.path.split(upright_rxp_fn)[1]
    hrdbf = os.path.split(tilt_rdbx_fn)[1]
    hrxpf = os.path.split(tilt_rxp_fn)[1]

    
    prof_dict = {"projpath":project,
                "scanvrdb": vrdbf,
                "scanvrxp": vrxpf,
                "scanhrdb": hrdbf,
                "scanhrxp": hrxpf,
                "frequency":freq,
                "resolution":res,
                "hinge_pai":hinge_pai, 
                "weighted_pai":weighted_pai, 
                "linear_pai":linear_pai, 
                "hinge_pavd":hinge_pavd, 
                "linear_pavd":linear_pavd, 
                "weighted_pavd": weighted_pavd}
    
    # Add one key per zenith angle, each containing a full height profile
    for z, pgap_vec in zip(vpp.zenith_bin, vpp.pgap_theta_z):
        key = f"pgap{z:05.1f}".replace('.', '')   # 7.5 → 'pgap0075', 12.5 → 'pgap0125'
        prof_dict[key] = pgap_vec.tolist()
    
    df = pd.DataFrame(prof_dict)
    df.to_csv(f"/home/kdayal/projects/pylidar-tls-canopy/results/benchmarking/20260205/reprocessed_new_{vrdbf}.csv", index=False)
    
    print("done")
    print("***************************")
    
    

    return prof_dict


        
test1 = all_paths_df.apply(get_plantprofiles, axis=1)   