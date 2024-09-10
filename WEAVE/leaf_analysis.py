import os
import glob
import time
import datetime
import numpy as np
from scipy.optimize import curve_fit,minimize
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import matplotlib

from pylidar_tls_canopy import leaf_io, plant_profile, grid
from pylidar_tls_canopy.rsmooth import rsmooth

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from tkinter import filedialog

#directory = filedialog.askdirectory()



os.chdir(f'/mnt/c/users/kdayal/Documents/2_data/1_Bosland/003/LF/002')

hinge_csv_list = glob.glob('data/ESS?????_*_hinge_*.csv')

hres = 0.5
max_h = 40
nbins = int(max_h / hres)

leaf_hinge_files = sorted(hinge_csv_list)

#create an empty list to collect the vertical plant profiles (VPPs) 
leaf_hinge_vpp = []

#loop through the list of LEAF files to generate and store the corresponding VPPs
for leaf_file in leaf_hinge_files:
    vpp = plant_profile.Jupp2009(hres=hres, zres=5, ares=90, min_h=0, max_h=max_h)
    try:
        valid = vpp.add_leaf_scan_position(leaf_file, method='FIRSTLAST', 
                                           min_zenith=55, max_zenith=60, sensor_height=1.8)

        if valid:
            vpp.get_pgap_theta_z(min_azimuth=0, max_azimuth=360)
        else:
            msg = f'Empty scan: {leaf_file}'
            print(msg)

        leaf_hinge_vpp.append(vpp)
    except Exception:
        msg = f'Problem scan: {leaf_file}'
        print(msg)


# count the number of days of data present in the folder and add one day
ndays = (leaf_hinge_vpp[-1].datetime - leaf_hinge_vpp[0].datetime).days + 1

#get the dates for the labels
leaf_dates = np.array([leaf_hinge_vpp[0].datetime + datetime.timedelta(days=d) for d in range(ndays)])


nobs = len(leaf_hinge_vpp)

#check which hinge scan was done and for how many hours
#here scan_hours refers to 'hours' as a name of the scan and not the duration
#for example, 12 (midday) scan, 21 (9 pm) scan etc.
#the code below results in, for example, the 9 pm scan (scan_hours) was done x (scan_count) times
#this is useful to retain only those hinge scans that were done periodically as a part of the time series i.e scan_count>1
scan_hours, scan_count = np.unique([p.datetime.hour for p in leaf_hinge_vpp], return_counts=True)
scan_hours = [h for h,c in zip(scan_hours,scan_count) if c > 1]

pai_z_grids = []
quality_flag = []
for scan_hour in scan_hours:
    
    with grid.LidarGrid(ndays, nbins, 0, nbins, resolution=1, init_cntgrid=True) as grd:
        for i in range(nobs):
            d = leaf_hinge_vpp[i].datetime
            day = (d - leaf_hinge_vpp[0].datetime).days
            if d.hour == scan_hour:
                if hasattr(leaf_hinge_vpp[i], 'pgap_theta_z'):
                    pai = leaf_hinge_vpp[i].calcHingePlantProfiles()
                    grd.add_column(pai, day, method='MEAN')
        
        grd.finalize_grid(method='MEAN')
        pai_z_grid = grd.get_grid()
    
    pai_z_grids.append(pai_z_grid)
    
    

bp = [0,50,pai_z_grids[0].shape[2]] # breakpoints

pai = []
pai_smooth = []
quality_flags = []
for pai_z_grid in pai_z_grids:
    
    pai_z0 = np.zeros(pai_z_grid.shape[2])
    pai_smooth_tmp = np.zeros(pai_z_grid.shape[2])
    bad_data = np.zeros(pai_z_grid.shape[2], dtype=bool)
    for i in range(len(bp)-1):
        
        #negative values are replaced by nan
        pai_z0_tmp = np.where(pai_z_grid[0,-1,bp[i]:bp[i+1]] > 0, pai_z_grid[0,-1,bp[i]:bp[i+1]], np.nan)

        smooth_tmp,weights = rsmooth(pai_z0_tmp, p=1)
        resid = pai_z0_tmp - smooth_tmp
        bad_data_tmp = (weights <= 0) | np.isnan(pai_z0_tmp) #| (resid < -1)
        
        pai_z0[bp[i]:bp[i+1]] = pai_z0_tmp
        pai_smooth_tmp[bp[i]:bp[i+1]] = smooth_tmp
        bad_data[bp[i]:bp[i+1]] = bad_data_tmp
        
    quality_flags.append(~bad_data)
    pai_smooth.append(pai_smooth_tmp)
    pai.append(pai_z0)
    

# Define a sigmoid function
def sigmoid(x, L , x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

# Define a double sigmoid function
def double_sigmoid(p, x):
    sigma1 = 1.0 / (1 + np.exp(p[2] * (x - p[3])))
    sigma2 = 1.0 / (1 + np.exp(-p[4] * (x - p[5])))
    return p[0] - p[1] * (sigma1 + sigma2 - 1)

# Define the cost function for the double sigmoid
def cost_function(p, t, y_obs, passer, sigma_obs, func=double_sigmoid):
    y_pred = func(p, t)
    cost = -0.5 * (y_pred[passer] - y_obs)**2 / sigma_obs**2
    return -cost.sum()


# Fit the sigmoid model
pai_sigmoid = []
for pai_z0,quality_flag in zip(pai,quality_flags):
    real_dates = np.array([(d - leaf_dates[0]).days for d in leaf_dates], dtype=float)
    try:      
        p0 = np.array([2, 5, 0.1, 20, 0.1, 200])
        n = np.count_nonzero(quality_flag)
        result = minimize(cost_function, p0, args=(real_dates, pai_z0[quality_flag], quality_flag, np.ones(n)))
        yfit = double_sigmoid(result.x, real_dates)
        
        pai_sigmoid.append(yfit)
    except (ValueError,RuntimeError) as e:
        pai_sigmoid.append(None)
        print(e)
    
 
def plot_timseries_1d(dates, values, quality, fitted=None, xlim=[None,None], ylim=[0,None], 
    title=None, ylabel=r'PAI ($m^{2} m^{-2}$)', linestyle=None, figsize=(20,10)):
    """Example function to plot a 2D histogram of the LEAF PAVD time-series""" 
    fig, ax = plt.subplots(ncols=1, nrows=1, squeeze=True, figsize=figsize)
    with plt.style.context('ggplot'):
        ax.scatter(dates[quality], values[quality], linestyle=linestyle, color='DarkGreen', label='True')
        ax.scatter(dates[~quality], values[~quality], linestyle=linestyle, color='Brown', label='False')
        if fitted is not None:
            ax.plot(dates, fitted, linestyle=linestyle, color='Black')
        date_format = mdates.DateFormatter('%d-%b-%y')
        ax.xaxis.set_major_formatter(date_format)
        ax.set(xlabel='Date', ylabel=ylabel, xlim=xlim, ylim=ylim, title=title)
    fig.autofmt_xdate()
    plt.legend(title='Quality')
    plt.tight_layout()
    plt.show() 

   
for i,scan_hour in enumerate(scan_hours):
    title = f'{scan_hour:d}AM Scans'
    plot_timseries_1d(leaf_dates, pai[i], quality_flags[i],
                      xlim=[None,None], ylim=[0,7], title=title, linestyle='dashed',
                      ylabel=r'PAI ($m^{2} m^{-2}$)', figsize=(12,5))
    
    
    
    
    
        
#HOBO weather data
weath_data=pd.read_csv(f"/mnt/c/users/kdayal/Documents/2_data/1_Bosland/5_LEAF/2_data/export.csv")

weath_data['date']=weath_data['date'].astype('datetime64[ns]')



fig, ax1 = plt.subplots(figsize=(12,5))
ax1.scatter(leaf_dates, pai[0], color='Green')
ax2 = ax1.twinx()
ax2.bar(weath_data['date'], weath_data['prcp'], alpha=0.2)
ax1.set(ylabel="PAI m²/m²", ylim=[0,5])
ax2.set(ylabel="Precipitation (mm)")
plt.grid(axis = 'x')
plt.show()



fig, ax1 = plt.subplots(figsize=(12,5))
ax1.plot(weath_data['date'], weath_data['prcp'], alpha=0.5)
ax1.set(ylabel="Average temperature (°C)")
plt.show()

weath_data



