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

# # for plotting
import matplotlib.pyplot as plt

out_directory = '/storage/scratch/giub_geco/fbernhard/FORHYX/'
# os.makedirs(out_directory, exist_ok=True)


# A) PCWD:

# B) PCWD:
pcwd_input_path = '/storage/capacity/occr_geco/data_2/archive/era5land_munoz-sabater_2021/data_derived_03_daily_pcwd.narm_v2-doy-reset_netcdf'

## some configurations
dask.config.set({
    'distributed.comm.timeouts.connect': '60s',
    'distributed.comm.timeouts.tcp': '120s',
})

def list_netcdf_files(root_dir, pattern):
    netcdf_files = []
    for root, dirs, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, pattern):
            if filename.endswith('.nc'):
                netcdf_files.append(os.path.join(root, filename))
    return netcdf_files

if __name__ == '__main__':

    # cluster = LocalCluster(n_workers=min(64, int(os.environ['SLURM_NPROCS'])), threads_per_worker=1)
    # cluster = LocalCluster(n_workers=4, threads_per_worker=1)

    cluster = LocalCluster(n_workers=min(64, int(os.environ['SLURM_CPUS_PER_TASK'])), threads_per_worker=1)
    # client = Client(address=cluster)
    with Client(address=cluster) as client:
        print('Cluster setup: ', flush=True)
        print(client, flush=True)
        print(client.dashboard_link, flush=True)

        ### Step 2: Loop over different target variables
        vars = ['pcwd', 'sce']
        #curr_var = vars[0]
        for curr_var in vars:

            ### Step 3: Check presence/absence and validity of NetCDF Files
            if curr_var == 'pcwd':
                # List all NetCDF files in path
                print(pcwd_input_path, flush=True)
                file_pattern = f'*data_derived_03_daily_pcwd_v2-doy_*_r-generated.nc'
                netcdf_files = list_netcdf_files(pcwd_input_path, file_pattern)
                netcdf_files.sort()

                print('All files to treat:')
                pprint.pp(netcdf_files)

            if curr_var == 'sce':
                # List all NetCDF files in path
                raise ValueError(f'Target aggregation not yet implemented for {curr_var}')

            ### Step 4: Open Multiple NetCDF Files
            # Use `xarray.open_mfdataset` to open multiple NetCDF files as a single dataset:
            if not netcdf_files:
                raise FileNotFoundError(f'No files found for {curr_var}')
            
            ds_daily = xr.open_mfdataset(
                netcdf_files,
                combine='by_coords',
                #engine='h5netcdf', 
                parallel =True, 
                compat='no_conflicts',
                join='exact',
                chunks="auto" # {'time': 28, 'lat': 256, 'lon': 256}
            )

            print('\n\nFULL DAILY INPUT DATA SET TO AGGREGATE: ########################', flush=True)
            print(ds_daily, flush=True)

            # Remove the duplicate 360° endpoint and wrap to [-180, 180).
            if np.isclose(ds_daily.lon[-1], 360.0):
                ds_daily = ds_daily.isel(lon=slice(None, -1))
            ds_daily = (
                ds_daily
                .assign_coords(lon=((ds_daily.lon + 180) % 360) - 180)
                .sortby('lon')
            )

            ### Step 5: Aggregate in time (weekly) and in space (to regions)
            regions = gpd.read_file('/storage/homefs/fb24k097/GitHub/fabern/forhyx/data/regions/shapefile/regions.shp')
            
            # ensure there is a human-readable name column if you want one
            if 'region_name' not in regions.columns:
                regions['region_name'] = regions['region_id'].astype(str)

            min_lon, min_lat, max_lon, max_lat = regions.total_bounds
            
            ## Prepare ERA5Land data coordinates:
            # Crop before constructing a region × latitude × longitude array.
            ds_daily2 = ds_daily.sel(
                lon=slice(min_lon, max_lon),
                lat=slice(min_lat, max_lat),
            )
            
            # Create mask with dimensions: region, lat, lon
            mask = regionmask.mask_3D_geopandas(
                regions, 
                ds_daily2, 
                numbers='region_id', 
                drop=True)

            # Append regional mask to xarray
            ds_daily3 = ds_daily2.where(mask)

            # # Check output visually:
            # plt.figure()
            # ds_daily.sel(lon = slice(0,30), lat = slice(35,65)).isel(time=181)['pcwd_mm'].plot();
            # plt.savefig('figure.png')
            # plt.figure()
            # # mask.sel(lon = slice(0,30), lat = slice(35,65), region = 2).plot();
            # # mask.sel(lon = slice(0,30), lat = slice(35,65), region = 3).plot();
            # mask.sel(lon = slice(0,30), lat = slice(35,65), region = 10).plot();
            # plt.savefig('figure_regions.png')
            # plt.figure()
            # ds_daily3.sel(region = 10, lon = slice(0,30), lat = slice(35,65)).isel(time=181)['pcwd_mm'].plot();
            # plt.savefig('figure_pcwd_region10.png')
            # plt.figure()
            # ds_daily3.sel(region = 9, lon = slice(0,30), lat = slice(35,65)).isel(time=181)['pcwd_mm'].plot();
            # plt.savefig('figure_pcwd_region9.png')

            # 1. Resample spatially
            # Compute mean per region using area-based weights
            
            latitude_weights = np.cos(np.deg2rad(ds_daily3.lat))

            regional_daily = (
                ds_daily3
                .weighted(latitude_weights)
                .mean(dim=('lat', 'lon'), skipna=True)
            )

            # # Check output visually:
            # fig, ax = plt.subplots(figsize=(12, 6))
            # regional_daily['pcwd_mm'].plot.line(
            #     x='time', hue='region', ax=ax, add_legend=True,
            # )
            # ax.set(
            #     title='Daily regional mean potential cumulative water deficit',
            #     xlabel='Date', ylabel='PCWD (mm)',
            # )
            # ax.grid(visible=True, alpha=0.3)
            # fig.tight_layout()
            # fig.savefig('figure_pcwd_regional_daily_timeseries.png', dpi=150)
            # plt.close(fig)


            # 2. Resample temporally using built-in Dask-optimized resample
            # Compute mean per week
            pcwd_regional_weekly = regional_daily.resample(time='7D').mean() # TODO: do we want the regional mean?

            # 3. Output as CSV:
            pcwd_weekly_df = pcwd_regional_weekly['pcwd_mm'].to_pandas()
            pcwd_weekly_df.columns.name = 'region'
            pcwd_weekly_df = pcwd_weekly_df.rename(
                columns=lambda region_id: f"region_{region_id}"
            )
            
            csv_path = os.path.join(out_directory, f'{curr_var}_regional_weekly.csv')
            pcwd_weekly_df.to_csv(csv_path, index_label='time')
            print(f'\n\nWrote weekly regional PCWD to {csv_path}', flush=True)
            print('########################\n\n', flush=True)


