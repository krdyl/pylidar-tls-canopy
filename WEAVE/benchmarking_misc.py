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
from pylidar_tls_canopy import riegl_io

all_paths_df = pd.read_csv("/home/kdayal/projects/pylidar-tls-canopy/results/all_paths_df_20260206.csv")

##snippet to test for one row of the dataframe

scans = all_paths_df[0:1]
project = scans['project_path'].values[0]    
    
upright_rdbx_fn = scans['rdb_v'].values[0] 
upright_rxp_fn = scans['rxp_v'].values[0] 
upright_transform_fn = scans['dat_v'].values[0] 


tilt_rdbx_fn = scans['rdb_h'].values[0] 
tilt_rxp_fn = scans['rxp_h'].values[0] 
tilt_transform_fn = scans['dat_h'].values[0] 

# Ground plane
transform_matrix = riegl_io.read_transform_file(upright_transform_fn)
x0, y0, z0, _ = transform_matrix[3, :]
grid_extent = 60
grid_resolution = 10
grid_origin = [x0, y0]

x, y, z, r = plant_profile.get_min_z_grid([upright_rdbx_fn, tilt_rdbx_fn], 
                                          [upright_transform_fn, tilt_transform_fn],
                                          grid_extent, 
                                          grid_resolution,
                                          grid_origin=grid_origin,
                                          rxp=False)


# Determine the origin coordinates to use
transform_matrix = riegl_io.read_transform_file(upright_transform_fn)
x0,y0,z0,_ = transform_matrix[3,:]

grid_extent = 60
grid_resolution = 1
grid_origin = [x0,y0]

# Optional weighting of points by 1 / range
planefit = plant_profile.plane_fit_hubers(x, y, z, w=1/r)
planefit['Summary']

planefit['Parameters']
scans[["parameter_c", "parameter_a", "parameter_b"]]

# If the ground plane is not defined then set ground_plane to None
# and use the sensor_height argument when adding scan positions
terrain_params = np.array([scans['parameter_c'], scans['parameter_a'], scans['parameter_b']])
print(terrain_params)
vpp = plant_profile_2.Jupp2009(hres=0.5, 
                                zres=5, 
                                ares=360,
                                min_z=5, 
                                max_z=70, 
                                min_h=0, 
                                max_h=50,
                                ground_plane=terrain_params)

# If using RXP files only as input, set rdbx_file to None (the default)
query_str = ['reflectance > -20']
vpp.add_riegl_scan_position(upright_rxp_fn, 
                            upright_transform_fn, 
                            sensor_height=None,
                            rdbx_file=upright_rdbx_fn, 
                            method='WEIGHTED', 
                            min_zenith=35, 
                            max_zenith=70,
                            query_str=query_str)


# If using RXP files only as input, set rdbx_file to None (the default)
query_str = ['reflectance > -20']
vpp.add_riegl_scan_position(tilt_rxp_fn, 
                            tilt_transform_fn, 
                            sensor_height=None,
                            rdbx_file=tilt_rdbx_fn, 
                            method='WEIGHTED', 
                            min_zenith=5, 
                            max_zenith=35,
                            query_str=query_str)

vpp.get_pgap_theta_z(min_azimuth=0, max_azimuth=360)


prof_dict = {"projpath":project}






all_paths_df['project_path']



all_paths_df = pd.read_csv("/home/kdayal/projects/pylidar-tls-canopy/all_paths_df.csv")


## examining the data for differences in vz400 and vz400i


vz = all_paths_df.loc[(all_paths_df['project_path'] == '2024-04-08.001.RiSCAN') & 
                 (all_paths_df['resolution'] == 0.03)]

vzi = all_paths_df.loc[(all_paths_df['project_path'] == '2024-04-08-BOSLAND-1i.RiSCAN') & 
                 (all_paths_df['resolution'] == 0.03) &
                 (all_paths_df['frequency'] == 300)]


scans = vz


upright_rdbx_fn = scans['rdb_v'].iloc[0]
upright_rxp_fn = scans['rxp_v'].iloc[0]
upright_transform_fn = scans['dat_v'].iloc[0]


tilt_rdbx_fn = scans['rdb_h'].iloc[0]
tilt_rxp_fn = scans['rxp_h'].iloc[0]
tilt_transform_fn = scans['dat_h'].iloc[0]



# Determine the origin coordinates to use
transform_matrix = riegl_io.read_transform_file(upright_transform_fn)
x0,y0,z0,_ = transform_matrix[3,:]

grid_extent = 60
grid_resolution = 1
grid_origin = [x0,y0]

# If the ground plane is not defined then set ground_plane to None
# and use the sensor_height argument when adding scan positions
terrain_params = np.array([scans['parameter_c'].iloc[0], scans['parameter_a'].iloc[0], scans['parameter_b'].iloc[0]])
print(terrain_params)
vpp = plant_profile.Jupp2009(hres=0.5, 
                            zres=5, 
                            ares=90,
                            min_z=35, 
                            max_z=70, 
                            min_h=0, 
                            max_h=50,
                            ground_plane=terrain_params)

min_zenith_r = np.radians(35)
max_zenith_r = np.radians(70)
pulse_cols = ['zenith','azimuth','target_count']
point_cols = ['x','y','z','range','target_index',
                'zenith','azimuth','target_count']


pulses = {}
with riegl_io.RXPFile(rxp_file, transform_file=transform_file, query_str=query_str) as rxp:
    for col in pulse_cols:
        pulses[col] = rxp.get_data(col, return_as_point_attribute=False)
    idx = (pulses['zenith'] >= min_zenith_r) & (pulses['zenith'] < max_zenith_r)
    if np.any(idx):
        self.add_shots(pulses['target_count'][idx], pulses['zenith'][idx],
            pulses['azimuth'][idx], method=method)

    points = {}
    if rdbx_file:
        with riegl_io.RDBFile(rdbx_file, transform_file=transform_file, query_str=query_str) as f:
            for col in point_cols:
                points[col] = f.get_data(col)
    else:
        for col in point_cols:
            points[col] = rxp.get_data(col, return_as_point_attribute=True)

    if self.ground_plane is None:
        if sensor_height is not None:
            zoffset = rxp.transform[3,2] - sensor_height
        else:
            zoffset = rxp.transform[3,2]
    else:
        zoffset = self.ground_plane[0]

if self.ground_plane is None:
    height = points['z'] + zoffset
else: 
    height = points['z'] - (self.ground_plane[1] * points['x'] +
        self.ground_plane[2] * points['y'] + zoffset)

idx = (points['zenith'] >= min_zenith_r) & (points['zenith'] < max_zenith_r)
if max_hr is not None:
    hr = points['range'] * np.sin(points['zenith'])
    idx &= hr < max_hr
if np.any(idx):
    self.add_targets(height[idx], points['target_index'][idx], 
        points['target_count'][idx], points['zenith'][idx],
        points['azimuth'][idx], method=method)







def add_riegl_scan_position(self, rxp_file, transform_file, rdbx_file=None, sensor_height=None,
        method='WEIGHTED', min_zenith=5, max_zenith=70, max_hr=None, query_str=None):
        """
        Add a RIEGL scan position to the profile
        """
        min_zenith_r = np.radians(min_zenith)
        max_zenith_r = np.radians(max_zenith)
        pulse_cols = ['zenith','azimuth','target_count']
        point_cols = ['x','y','z','range','target_index',
                      'zenith','azimuth','target_count']

        pulses = {}
        with riegl_io.RXPFile(rxp_file, transform_file=transform_file, query_str=query_str) as rxp:
            for col in pulse_cols:
                pulses[col] = rxp.get_data(col, return_as_point_attribute=False)
            idx = (pulses['zenith'] >= min_zenith_r) & (pulses['zenith'] < max_zenith_r)
            if np.any(idx):
                self.add_shots(pulses['target_count'][idx], pulses['zenith'][idx],
                    pulses['azimuth'][idx], method=method)

            points = {}
            if rdbx_file:
                with riegl_io.RDBFile(rdbx_file, transform_file=transform_file, query_str=query_str) as f:
                    for col in point_cols:
                        points[col] = f.get_data(col)
            else:
                for col in point_cols:
                    points[col] = rxp.get_data(col, return_as_point_attribute=True)

            if self.ground_plane is None:
                if sensor_height is not None:
                    zoffset = rxp.transform[3,2] - sensor_height
                else:
                    zoffset = rxp.transform[3,2]
            else:
                zoffset = self.ground_plane[0]
       
        if self.ground_plane is None:
            height = points['z'] + zoffset
        else: 
            height = points['z'] - (self.ground_plane[1] * points['x'] +
                self.ground_plane[2] * points['y'] + zoffset)
        
        idx = (points['zenith'] >= min_zenith_r) & (points['zenith'] < max_zenith_r)
        if max_hr is not None:
            hr = points['range'] * np.sin(points['zenith'])
            idx &= hr < max_hr
        if np.any(idx):
            self.add_targets(height[idx], points['target_index'][idx], 
                points['target_count'][idx], points['zenith'][idx],
                points['azimuth'][idx], method=method)




vpp_vz = plant_profile.Jupp2009(hres=0.5, 
                                zres=5, 
                                ares=90,
                                min_z=35, 
                                max_z=70, 
                                min_h=0, 
                                max_h=50,
                                ground_plane=terrain_params)


# If using RXP files only as input, set rdbx_file to None (the default)
query_str = ['reflectance > -20']
vpp_vz.add_riegl_scan_position(upright_rxp_fn, 
                            upright_transform_fn, 
                            sensor_height=None,
                            rdbx_file=upright_rdbx_fn, 
                            method='WEIGHTED', 
                            min_zenith=35, 
                            max_zenith=70,
                            query_str=query_str)

vpp_vz.add_riegl_scan_position(tilt_rxp_fn, 
                            tilt_transform_fn, 
                            sensor_height=None,
                            rdbx_file=tilt_rdbx_fn, 
                            method='WEIGHTED', 
                            min_zenith=35, 
                            max_zenith=70,
                            query_str=query_str)

vpp_vz.get_pgap_theta_z(min_azimuth=0, max_azimuth=360)

hinge_pai_vz = vpp_vz.calcHingePlantProfiles()
linear_pai_vz = vpp_vz.calcLinearPlantProfiles()
weighted_pai_vz = vpp_vz.calcSolidAnglePlantProfiles()

hinge_pavd_vz = vpp_vz.get_pavd(hinge_pai_vz)
linear_pavd_vz = vpp_vz.get_pavd(linear_pai_vz)
weighted_pavd_vz = vpp_vz.get_pavd(weighted_pai_vz)

scans = vzi


upright_rdbx_fn = scans['rdb_v'].iloc[0]
uptight_rxp_fn = scans['rxp_v'].iloc[0]
upright_transform_fn = scans['dat_v'].iloc[0]


tilt_rdbx_fn = scans['rdb_h'].iloc[0]
tilt_rxp_fn = scans['rxp_h'].iloc[0]
tilt_transform_fn = scans['dat_h'].iloc[0]


vpp_vzi = plant_profile.Jupp2009(hres=0.5, 
                                zres=5, 
                                ares=90,
                                min_z=35, 
                                max_z=70, 
                                min_h=0, 
                                max_h=50,
                                ground_plane=terrain_params)


# If using RXP files only as input, set rdbx_file to None (the default)
query_str = ['reflectance > -20']
vpp_vzi.add_riegl_scan_position(upright_rxp_fn, 
                            upright_transform_fn, 
                            sensor_height=None,
                            rdbx_file=upright_rdbx_fn, 
                            method='WEIGHTED', 
                            min_zenith=35, 
                            max_zenith=70,
                            query_str=query_str)

vpp_vzi.add_riegl_scan_position(tilt_rxp_fn, 
                            tilt_transform_fn, 
                            sensor_height=None,
                            rdbx_file=tilt_rdbx_fn, 
                            method='WEIGHTED', 
                            min_zenith=5, 
                            max_zenith=35,
                            query_str=query_str)

vpp_vzi.get_pgap_theta_z(min_azimuth=0, max_azimuth=360)


hinge_pai_vzi = vpp_vzi.calcHingePlantProfiles()
linear_pai_vzi = vpp_vzi.calcLinearPlantProfiles()
weighted_pai_vzi = vpp_vzi.calcSolidAnglePlantProfiles()

hinge_pavd_vzi = vpp_vzi.get_pavd(hinge_pai_vzi)
linear_pavd_vzi = vpp_vzi.get_pavd(linear_pai_vzi)
weighted_pavd_vzi = vpp_vzi.get_pavd(weighted_pai_vzi)

import numpy as np
import matplotlib.pyplot as plt

z = np.arange(0, 101)  # height in meters

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Panel 1 – Hinge
axes[0].plot(hinge_pai_vz, z, label='vz', linestyle='-')
axes[0].plot(hinge_pai_vzi, z, label='vzi', linestyle='--')
axes[0].set_title('Hinge PAI')
axes[0].invert_yaxis()
axes[0].set_ylabel('Height (m)')
axes[0].set_xlabel('PAI')
axes[0].legend()

# Panel 2 – Linear
axes[1].plot(linear_pai_vz, z, label='vz', linestyle='-')
axes[1].plot(linear_pai_vzi, z, label='vzi', linestyle='--')
axes[1].set_title('Linear PAI')
axes[1].set_xlabel('PAI')

# Panel 3 – Weighted
axes[2].plot(weighted_pai_vz, z, label='vz', linestyle='-')
axes[2].plot(weighted_pai_vzi, z, label='vzi', linestyle='--')
axes[2].set_title('Weighted PAI')
axes[2].set_xlabel('PAI')

plt.tight_layout()
plt.savefig("pai_comparison.png", dpi=300, bbox_inches='tight')



import numpy as np
import matplotlib.pyplot as plt

z = np.arange(0, 100)  # Height in meters

plt.figure(figsize=(5, 6))
plt.plot(weighted_pavd_vz, z, label='vz', linestyle='-')
plt.plot(weighted_pavd_vzi, z, label='vzi', linestyle='--')

plt.xlabel('PAVD')
plt.ylabel('Height (m)')
plt.title('Weighted PAVD Profile')
plt.legend()
plt.grid(True)


plt.savefig("weighted_pavd_comparison.png", dpi=300)



plt.figure(figsize=(5, 5))
plt.scatter(weighted_pavd_vz, weighted_pavd_vzi, s=10, alpha=0.7)

plt.xlabel('VZ')
plt.ylabel('VZI')
plt.title('Scatter Plot: VZ vs VZI (Weighted PAVD)')
plt.grid(True)
plt.axis('equal')  # Equal scaling for both axes
plt.tight_layout()
plt.savefig("scatter_vz_vzi.png", dpi=300)
plt.show()



import riegl_rdb
riegl_rdb.