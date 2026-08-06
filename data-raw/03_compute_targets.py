# compute targets (PCWD and SCE)

# PCWD done already: see geco-bern/cwd_global at https://github.com/geco-bern/cwd_global/tree/main/src/ERA5Land-fullResNoNA

# SCE: done hereafter


# aggregate targets over week and aggregate over region

### Step 1: Import Libraries 
# Import the required libraries:
import os, fnmatch
import pprint
import time
import xarray as xr
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
import dask
import geopandas as gpd
import regionmask
import re

import datetime as dt

import cartopy.crs as ccrs
#import library.utils as utils
#import library.io as io
#import skimage.morphology as skimo
import skimage.transform as skit
import skimage
# import metpy

from scipy.ndimage import convolve
import timeit
from glob import glob
from datetime import datetime
import dask
import argparse as ap

# # for plotting
import matplotlib.pyplot as plt

input_path    = '/storage/scratch/giub_geco/fbernhard/ERA5_aggregated_newCodeNewDownload/'

out_directory = '/storage/scratch/giub_geco/fbernhard/FORHYX/'
# os.makedirs(out_directory, exist_ok=True)
# ll -h /storage/scratch/giub_geco/fbernhard/FORHYX/03*
# rm -r /storage/scratch/giub_geco/fbernhard/FORHYX/03*



# python3 ~/GitHub/fabern/download-ECMWF-data/inventory_ECMWF.py --download-path  /storage/capacity/occr_geco/data_2/scratch/fbernhard/ERA5_1996-2025_run2026-07-14 # TODO: 
# ### NOTE: for predictors: For 2008-2025 with z.500, sst, siconc: nothing missing
# ### NOTE: for targets:    For 2008-2026 with cape, cp, 500.u,500.v,925.u,925.v: cape.2026.01-07, cp.2026.06-07
# python3 ~/GitHub/fabern/download-ECMWF-data/inventory_ECMWF.py --aggregate-path /storage/scratch/giub_geco/fbernhard/ERA5_aggregated_newCodeNewDownload/
# ### NOTE: for predictors: For 2010-2025 with mean_z.500, mean_sst, mean_siconc: nothing missing
# ### NOTE: for targets:    For 2010-2025 with max_cape, tot_cp, mean_u500,mean_v500,mean_u925,mean_v925: missing tot_cp, max_cape




dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "120s",
})

def list_netcdf_files(root_dir, pattern):
    netcdf_files = []
    for root, dirs, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, pattern):
            if filename.endswith('.nc'):
                netcdf_files.append(os.path.join(root, filename))
    return netcdf_files

if __name__ == "__main__":

    cluster = LocalCluster(n_workers=min(64, int(os.environ['SLURM_CPUS_PER_TASK'])), threads_per_worker=1)
    # os.environ['SLURM_NTASKS'] # TODO: determine what to use in the next line
    # os.environ['SLURM_NPROCS'] # TODO: determine what to use in the next line
    # cluster = LocalCluster(n_workers=min(64, int(os.environ['SLURM_NPROCS'])), threads_per_worker=1)
    # client = Client(address=cluster)
    with Client(address=cluster) as client:
        print("Cluster setup: ", flush=True)
        print(client, flush=True)
        print(client.dashboard_link, flush=True)

        #all_needed_vars_pressure_level = ['500.mean_u','500.mean_v','925.mean_u','925.mean_v'] Not named like this
        all_needed_vars_pressure_level = ['500_900_925.mean_u','500_900_925.mean_v']
        all_needed_vars_single_level   = ['max_cape', 'tot_cp']

        ### Step 3: Check presence/absence and validity of NetCDF Files
        # List all NetCDF files in path
        print(input_path, flush=True)
        file_pattern = f'*_UTCDaily.*.nc'
        netcdf_files = list_netcdf_files(input_path, file_pattern)
        netcdf_files.sort()
        # pprint.pp(netcdf_files)

        all_vars = all_needed_vars_pressure_level + all_needed_vars_single_level
        vars_regex = re.compile(r'.*\.(?:' + '|'.join(map(re.escape, all_vars)) + r')\..*\.nc$')
        netcdf_files2 = [ s for s in netcdf_files if vars_regex.match(s) ]

        print("All files to treat:", flush=True)
        pprint.pp(netcdf_files2)

        ### Step 4: Compute severe convective environments (SCE) (code adapted from https://github.com/feldmann-m/EU_conv)
        #### Step 4a: compute shear
        vars_pl_regex = re.compile(r'.*\.(?:' + '|'.join(map(re.escape, all_needed_vars_pressure_level)) + r')\..*\.nc$')
        netcdf_files_on_pressure_level = [ s for s in netcdf_files2 if vars_pl_regex.match(s) ]
        # pprint.pp(netcdf_files_on_pressure_level)
        
        # open pressure level data
        pl_data = xr.open_mfdataset(
            netcdf_files_on_pressure_level,
            combine='by_coords', 
            engine="h5netcdf", 
            parallel =True, 
            compat='no_conflicts',
            join='inner', #join='exact', # TODO: go back to exact
            chunks={"valid_time": "auto", "latitude": "auto", "longitude": "auto"})
        
        pl_data = pl_data.sel(valid_time = slice("2020-01-01", None)) # TODO: remove this subset (only used for development)
        # subset the European regions extent: [-7.625 32.625 30.125 54.625]
        pl_data = pl_data.sel(latitude  = slice(60,25))
        # pl_data = xr.concat([
        #     pl_data.sel(longitude=slice(350, 360)),
        #     pl_data.sel(longitude=slice(0, 35))
        #     ], dim='longitude')
        pl_data = pl_data.assign_coords(longitude=(((pl_data.longitude + 180) % 360) - 180))
        pl_data = pl_data.sortby('longitude')
        pl_data = pl_data.sel(longitude=slice(-10, 35))  # 10°W..35°E

        verbose = True
        if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': computing shear', flush=True)
        # sfc_data_2['shear'] = ((pl_data.u.sel(level=500)**2 + pl_data.v.sel(level=500)**2)**0.5 - (pl_data.u.sel(level=900)**2 + pl_data.v.sel(level=900)**2)**0.5).squeeze()
        du = pl_data.mean_u.sel(pressure_level=500) - pl_data.mean_u.sel(pressure_level=925) # TODO change to 925
        dv = pl_data.mean_v.sel(pressure_level=500) - pl_data.mean_v.sel(pressure_level=925) # TODO change to 925

        sfc_data_2={}
        sfc_data_2['shear'] = (du**2 + dv**2)**0.5                # TODO(fabian): is this the same as 3 lines prior? shear = (u5^2 + v5^2)^0.5 - (u9^2 + v9^2)^0.5 = A - B
                                                                #                                                  shear^2 = A^2 - 2AB + B^2 
                                                                #                                                          = (u5^2 + v5^2) - 2*(A*B) + (u9^2 + v9^2)
                                                                #                                                          = (u5^2 + v5^2) - 2*((u5^2 + v5^2)*(u9^2 + v9^2))^0.5 + (u9^2 + v9^2)
                                                                #                                                  shear   = sqrt(u5^2 + v5^2 + u9^2 + v9^2 - 2*()^0.5) # ????
        for key, da in sfc_data_2.items():
            da.name = key
        if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': converting array to xarray dataset', flush=True)
        #sfc_data_2 = xr.Dataset(sfc_data_2,compat='override')
        sfc_data_2_dataset = xr.merge(list(sfc_data_2.values()), compat='override')
        if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': writing shear dataset', flush=True)
        shear_fname = os.path.join(out_directory, f'03_era5-pressure-levels_UTCDaily.500_925_shear.nc')
        print(sfc_data_2_dataset, flush = True)
        sfc_data_2_dataset.to_netcdf(shear_fname)

        #### Step 4b: compute gridpoint-wise boolean if convective
        vars_sl_regex = re.compile(r'.*\.(?:' + '|'.join(map(re.escape, all_needed_vars_single_level)) + r')\..*\.nc$')
        netcdf_files_on_single_level = [ s for s in netcdf_files2 if vars_sl_regex.match(s) ]
        # pprint.pp(netcdf_files_on_single_level)
        
        # open single level data
        sfc_data_1 = xr.open_mfdataset(
            netcdf_files_on_single_level,
            combine='by_coords', 
            engine="h5netcdf", 
            parallel =True, 
            compat='no_conflicts',
            join='inner', #join='exact', # TODO: go back to exact
            chunks={"valid_time": "auto", "latitude": "auto", "longitude": "auto"})
        sfc_data_1 = sfc_data_1.sel(valid_time = slice("2020-01-01", None)) # TODO: remove this subset (only used for development)
        # subset the European regions extent: [-7.625 32.625 30.125 54.625]
        sfc_data_1 = sfc_data_1.sel(latitude  = slice(60,25))
        # sfc_data_1 = xr.concat([
        #     sfc_data_1.sel(longitude=slice(350, 360)),
        #     sfc_data_1.sel(longitude=slice(0, 35))
        #     ], dim='longitude')
        sfc_data_1 = sfc_data_1.assign_coords(longitude=(((sfc_data_1.longitude + 180) % 360) - 180))
        sfc_data_1 = sfc_data_1.sortby('longitude')
        sfc_data_1 = sfc_data_1.sel(longitude=slice(-10, 35))  # 10°W..35°E

        cape = sfc_data_1.max_cape
        shear = xr.open_mfdataset(shear_fname, combine="by_coords").shear
                        # if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': regridding CAPE')
                        # cape_r = skit.resize(
                        #         cape.values,
                        #         shear.values.shape,
                        #         mode="reflect",  # Handles boundaries gracefully
                        #         anti_aliasing=True,  # Smooth resizing
                        #     )
                        # regridder = xe.Regridder(cape, shear, method="bilinear")
                        # cape_r = regridder(cape)

        conv={}
        conv['conv_EU'] = (cape > 500) & (shear > 10) # US was: (cape > 1000) & (shear > 20)
        for key, da in conv.items():
            da.name = key
        if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': converting array to xarray dataset', flush=True)
        #sfc_data_2 = xr.Dataset(sfc_data_2,compat='override')
        conv_dataset = xr.merge(list(conv.values()), compat='override')
        if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': writing conv dataset', flush=True)
        conv_fname = os.path.join(out_directory, f'03_era5-pressure-levels_UTCDaily.conv.nc')
        print(conv_dataset, flush = True)
        conv_dataset.to_netcdf(conv_fname)


        #### Steb 5c: analyse spatial coherence (blops) to define SCE
        if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading data', flush=True)
        ticc = timeit.default_timer()
        conv = xr.open_dataset(conv_fname).conv_EU

        cp = sfc_data_1.tot_cp
        # subset convective precipitation
        #TODO: check if needed: cp_r = cp.sel(valid_time=slice(conv.valid_time[0],conv.valid_time[-1]+10**15))
        cp_r = cp
        cp_bin = cp_r > 0

        if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': retrieving object properties', flush=True)
        maxtime=len(conv.valid_time)

        SCE_OBJ_fname_csv = os.path.join(out_directory, f'03_era5-pressure-levels_UTCDaily.SCE.tttt.csv')
        SCE_OBJ_fname_nc  = os.path.join(out_directory, f'03_era5-pressure-levels_UTCDaily.SCE.tttt.nc')
        tracked_regions = np.zeros(conv.shape)
        tracked_properties = pd.DataFrame(columns=['time','itime','label','size','precip','ilat','ilon','lat','lon'])
        a1=0
        for t in range(maxtime):
            tic = timeit.default_timer()
            # if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': identifying conv objects')
            conv_obj = skimage.measure.label(conv.isel(valid_time=t).values) # uses skimage.measure.label()
            maxlabel = np.nanmax(conv_obj)
            
            for lab in range(maxlabel):
                t_p = pd.DataFrame(columns=['time','itime','label','size','precip','ilat','ilon','lat','lon'],data=np.zeros([1,9]))
                binary = (conv_obj==lab)
                size = np.nansum(binary)
                if size < 50: continue

                precip = np.nansum(cp_r.isel(valid_time=t) * (conv_obj==lab))
                if precip == 0: continue

                tracked_regions[t,:,:]+= binary*lab

                yy,xx = np.where(binary)
                ilon = np.nanmean(xx); ilat = np.nanmean(yy)
                ilon2 = np.round(ilon).astype(int); ilat2 = np.round(ilat).astype(int)

                if ilon2 >= len(conv.longitude): ilon2=len(conv.longitude)-1
                if ilon2 < 0 : ilon2=0
                if ilat2 >= len(conv.latitude): ilat2=len(conv.latitude)-1
                if ilat2 < 0 : ilat2=0

                lon = conv.longitude.isel(longitude=ilon2); lat = conv.latitude.isel(latitude=ilat2)
                t_p['itime']=t
                t_p['time']=conv.valid_time.isel(valid_time=t).values
                t_p['label']=lab
                t_p['size']=size
                t_p['precip']=precip
                t_p['a_precip']=np.nansum(cp_bin.isel(valid_time=t) * (conv_obj==lab))
                t_p['ilat']=ilat
                t_p['ilon']=ilon
                t_p['lat']=lat.values
                t_p['lon']=lon.values
                a1+=1

                tracked_properties = pd.concat([tracked_properties,t_p])
            if (t%100==0 and t>0) or range(maxtime): # intermediate save i.e save every 100 time steps of the growing DataFrame into (*.csv and *.nc)
                toc = timeit.default_timer()
                print('Timestep '+str(t)+' out of '+str(maxtime)+', duration '+str((toc-tic)/60)+' min', flush=True)
                tracked_properties.to_csv(SCE_OBJ_fname_csv.replace("tttt",str(t))) # time step specific fname
                tracked_properties = pd.DataFrame(columns=['time','itime','label','size','precip','ilat','ilon','lat','lon'])
                tracked_regions_xr = xr.DataArray(
                    tracked_regions[t-100:t, :, :],  # New data
                    coords=conv[t-100:t,:,:].coords,  # Retain the original coordinates
                    dims=conv[t-100:t,:,:].dims  # Retain the original dimensions
                )
                # tracked_regions_xr = copy.deepcopy(conv[t-100:t,:,:])
                # tracked_regions_xr.data = conv_obj[t-100:t,:,:]
                print(tracked_regions_xr, flush = True)
                tracked_regions_xr.to_netcdf(SCE_OBJ_fname_nc.replace("tttt",str(t))) # time step specific fname
        if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': saving final data', flush=True)
        # tracked_properties.to_csv(scr_data+str(t)+'conv_obj.csv')


        tocc = timeit.default_timer()
        print('total script time '+str((tocc-ticc)/60)+' min', flush=True)



        # MERGE DATA
        print('merging data', flush=True)
        
        files = sorted(glob(SCE_OBJ_fname_csv.replace("tttt",'*')))
        #pprint.pp(files)
        a1=0
        for file in files:
            if a1==0:
                tracked_properties = pd.read_csv(file)
                print(tracked_properties.shape, flush=True)
            else:
                tracked_properties_2 =  pd.read_csv(file)
                print(tracked_properties_2.shape, flush=True)
                tracked_properties = pd.concat([tracked_properties,tracked_properties_2])
                print(tracked_properties.shape, flush=True)
            a1+=1
        tracked_properties = tracked_properties.drop(columns='Unnamed: 0')

        print(tracked_properties.shape, flush=True)
        tracked_properties.to_csv(SCE_OBJ_fname_csv.replace("tttt","all-steps")) # sotre final

        files = sorted(glob(SCE_OBJ_fname_nc.replace("tttt",'[0-9]*')))
        #pprint.pp(files)
        data = xr.open_mfdataset(files).rename_vars({'__xarray_dataarray_variable__':'SCE_ID'})
        print(data, flush = True)
        data.to_netcdf(SCE_OBJ_fname_nc.replace("tttt","all-steps"))

        # plt.figure()
        # data.sel(longitude = slice(0,30), latitude = slice(65,25)).isel(valid_time=181)
        # data.sel(longitude = slice(0,30), latitude = slice(65,25)).isel(valid_time=181)['SCE'].plot();
        # data.sel(longitude = slice(0,30), latitude = slice(65,25)
        #                 ).sel(valid_time="2022-06-27")['SCE'].plot();
        # plt.savefig('figure_SCE.png')

        # xr.open_dataset(SCE_OBJ_fname_nc.replace("tttt","all-steps"))

                        # # For output..... TODO
                        # compr_level = 5
                        # chunk_size_definition = {
                        # 'valid_time': 4,   # in weekly resolution = 4 weeks
                        # 'latitude':   18,  # if 1-deg resolution = 18 deg
                        # 'longitude':  36,
                        # 'pressure_level': 3, # NOTE: only used for pressure level ERA5 (UNUSED FOR PREDICTORS)
                        # }