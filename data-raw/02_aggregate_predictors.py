# aggregate predictors over week and coarsen grid to reduce complexity

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

# # for plotting
# # import matplotlib.pyplot as plt
# import pathlib
# import re

# FOR SDSC project we need for predictors
# for ERA5:
#   ds_weekly[['z.500']]
#   ds_weekly[['mean_sst', 'mean_siconc']]
# for ERA5-Land:
#   ds_weekly[['mean_t2m', 'mean_d2m', 'mean_swvl1', 'mean_snowc']]

# input_path = '/storage/capacity/occr_geco/data_2/scratch/fbernhard/ERA5Land-T_2000-2026_run2026-07-16/aggregated_global'
# input_path = '/storage/capacity/occr_geco/data_2/scratch/fbernhard/ERA5_1996-2025_run2026-07-14/aggregated_global'
input_path = '/storage/scratch/giub_geco/fbernhard/aggregated_newCodeNewDownload_2/' # era5land
input_path = '/storage/scratch/giub_geco/fbernhard/aggregated_newCodeNewDownload_2/' # era5land
input_path = '/storage/scratch/giub_geco/fbernhard/ERA5_aggregated_newCodeNewDownload/' # era5-single and era5-pressure

out_directory = '/storage/scratch/giub_geco/fbernhard/FORHYX/'
# os.makedirs(out_directory, exist_ok=True)

# Check current download and aggregation inventory (if files missing, run the corrsponding download slurm scripts)
# # ERA5-Land:
# python3 ~/GitHub/fabern/download-ECMWF-data/inventory_ECMWF.py --download-path /storage/capacity/occr_geco/data_2/scratch/fbernhard/ERA5Land-T_2000-2026_run2026-07-16/
# ### NOTE: for predictors: For 2020-2026 with 't2m', 'd2m', 'swvl1', 'snowc':  nothing missing
# ### NOTE: for targets:    For 2020-2026 with 'ssr','str','tp','sp','t2m'   :  nothing missing (have all of this from earlier download)
# # python3 ~/GitHub/fabern/download-ECMWF-data/inventory_ECMWF.py --aggregate-path /storage/capacity/occr_geco/data_2/scratch/fbernhard/ERA5Land-T_2000-2026_run2026-07-16/aggregated_global
# ### NOTE: for predictors: For 2022-2025 with 'mean_t2m', 'mean_d2m', 'mean_swvl1', 'mean_snowc':  nothing missing
# ### NOTE: for targets:    For 2024-2026 with 'tot_ssr','tot_str','tot_tp','mean_sp','mean_t2m' :  all of 'tot_ssr','tot_str','tot_tp','mean_sp' missing

# # ERA5:
# python3 ~/GitHub/fabern/download-ECMWF-data/inventory_ECMWF.py --download-path  /storage/capacity/occr_geco/data_2/scratch/fbernhard/ERA5_1996-2025_run2026-07-14 # TODO: 
# ### NOTE: for predictors: For 2008-2025 with z.500, sst, siconc: nothing missing
# ### NOTE: for targets:    For 2008-2026 with cape, cp, 500.u,500.v,925.u,925.v: cape.2026.01-07, cp.2026.06-07
# python3 ~/GitHub/fabern/download-ECMWF-data/inventory_ECMWF.py --aggregate-path /storage/scratch/giub_geco/fbernhard/ERA5_aggregated_newCodeNewDownload/
# ### NOTE: for predictors: For 2010-2026 with mean_z.500, mean_sst, mean_siconc: nothing missing
# ### NOTE: for targets:    For 2010-2026 with max_cape, mean_cp, mean_u500,mean_v500,mean_u925,mean_v925: missing mean_cp, max_cape

compr_level = 5
chunk_size_definition = {
  'valid_time': 4,   # in weekly resolution = 4 weeks
  'latitude':   18,  # if 1-deg resolution = 18 deg
  'longitude':  36,
  'pressure_level': 3, # NOTE: only used for pressure level ERA5 (UNUSED FOR PREDICTORS)
}

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

    # os.environ['SLURM_NTASKS'] # TODO: determine what to use in the next line
    # os.environ['SLURM_NPROCS'] # TODO: determine what to use in the next line
    # cluster = LocalCluster(n_workers=min(64, int(os.environ['SLURM_NPROCS'])), threads_per_worker=1)
    # client = Client(address=cluster)
    cluster = LocalCluster(n_workers=min(64, int(os.environ['SLURM_CPUS_PER_TASK'])), threads_per_worker=1)
    with Client(address=cluster) as client:
        print("Cluster setup: ", flush=True)
        print(client, flush=True)
        print(client.dashboard_link, flush=True)

        ### Step 2: Loop over different predictor variables
        # 't2m', 'd2m', 'swvl1', 'snowc' # era5-land # TODO: modify code for this
        vars = ['500.mean_z', 'mean_sst', 'mean_siconc'] # era5
        for curr_var in vars:

            ### Step 3: Check presence/absence and validity of NetCDF Files
            # List all NetCDF files in path
            print(input_path, flush=True)
            file_pattern = f'*_UTCDaily.{curr_var}*.nc'
            netcdf_files = list_netcdf_files(input_path, file_pattern)
            netcdf_files.sort()

            print("All files to treat:")
            pprint.pp(netcdf_files)

            ### Step 4: Open Multiple NetCDF Files
            # Use `xarray.open_mfdataset` to open multiple NetCDF files as a single dataset:
            ds_daily = xr.open_mfdataset(
                netcdf_files,
                combine='by_coords', 
                engine="h5netcdf", 
                parallel =True, 
                compat='no_conflicts',
                join='exact',
                chunks={"valid_time": "auto", "latitude": "auto", "longitude": "auto"})

            print("\n\nFULL DAILY INPUT DATA SET TO AGGREGATE: ########################", flush=True)
            print(ds_daily, flush=True)

            ### Step 5: Aggregate in time (weekly) and in space (1 deg resolution)

            # 1. Coarsen spatially FIRST (100x (or 16x) data reduction before temporal ops)
            target_resolution = 1.0 # degrees

            # get current resolution (either 0.25 (ERA5) or 0.1 (ERA5-Land)) 
            curr_grid_resolution = np.diff(ds_daily.longitude[0:2].values)[0] # NOTE: assumes lon-lat grid is homogenous
            # TODO: check this with ERA5-Land
            
            factor = int(target_resolution/curr_grid_resolution)
            ds_coarse = ds_daily.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()

            # 2. Resample temporally LATER using built-in Dask-optimized resample
            ds_weekly = ds_coarse.resample(valid_time="7D").mean()

            print("Aggregated weekly:", flush=True)
            print(ds_weekly, flush=True)

            # NOTE: For rigorous scientific spatial aggregation, use cosine weighting to account for grid-area on sphere:
            # UNUSED: # Cosine-latitude weighting prior to coarsening or regridding
            # UNUSED: weights = np.cos(np.deg2rad(ds_daily.latitude))
            # UNUSED: weights.name = "weights"
            # UNUSED: ds_weighted = ds_daily.weighted(weights)
            # UNUSED: ds_weekly2 = (
            # UNUSED:     ds_weighted.coarsen(latitude=factor, longitude=factor, boundary="trim")
            # UNUSED:     .mean()
            # UNUSED:     .resample(valid_time="7D")
            # UNUSED:     .mean()
            # UNUSED: )

            ### Step 6: Output as single netCDF file:
            print("\n\nSTORE AS SINGLE netCDF FILE: ########################", flush=True)
            fname_prefix = {
                # era5-pressure:
                "500.mean_z":  "FORHYX.predictors.era5pressure",
                # era5-single:
                "mean_siconc": "FORHYX.predictors.era5single",
                "mean_sst":    "FORHYX.predictors.era5single",
                # era5-land:
                "mean_t2m":    "FORHYX.predictors.era5land",
                "mean_d2m":    "FORHYX.predictors.era5land",
                "mean_swvl1":  "FORHYX.predictors.era5land",
                "mean_snowc":  "FORHYX.predictors.era5land",
                }[curr_var]

            start_year = pd.to_datetime(min(ds_weekly['valid_time'].values)).year
            end_year = pd.to_datetime(max(ds_weekly['valid_time'].values)).year
            year_range = f"{start_year}-{end_year}"

            var_name = curr_var.replace(".", "_")
            new_fpath = f"{out_directory}/{fname_prefix}.{var_name}.{year_range}.nc"
            print(new_fpath, flush=True)

            if curr_var == "500.mean_z":
                ds_weekly = (
                    ds_weekly
                    .sel(pressure_level=500)
                    .reset_coords("pressure_level", drop=True)
                    .rename({"mean_z": "500_mean_z"}) # NOTE: get rid of "."
                )

            curr_chunk_size = tuple(            # this ensures correct order of coords and limit maximum chunk size
                min(
                    chunk_size_definition[k],
                    ds_weekly.sizes[k]
                    )
                for k in ds_weekly.dims)
            print("Target chunk size: ", curr_chunk_size, flush=True)
            
            # Define the new NetCDF encoding
            coord_encoding = {coord: ds_weekly[coord].encoding for coord in ds_weekly.coords}
            var_encoding   = {var:   ds_weekly[var].encoding   for var   in ds_weekly.data_vars}
            # Remove incompatible encoding parameters:
            for coord in coord_encoding:
                coord_encoding[coord].pop('szip', None) # None results in removal if it exists
                coord_encoding[coord].pop('zstd', None) # None results in removal if it exists
                coord_encoding[coord].pop('bzip2', None) # None results in removal if it exists
                coord_encoding[coord].pop('blosc', None) # None results in removal if it exists
            for var in var_encoding:
                var_encoding[var].pop('coordinates', None) # remove error in field 'coordinates'
                var_encoding[var].pop('szip', None)
                var_encoding[var].pop('zstd', None)
                var_encoding[var].pop('bzip2', None)
                var_encoding[var].pop('blosc', None)
                var_encoding[var].pop('preferred_chunks', None)
            # Set 'complevel', and 'chunksizes' of all variables # https://stackoverflow.com/a/66333685
            for var in var_encoding:
                var_encoding[var]['shuffle']    = True                               # update compression level
                var_encoding[var]['complevel']  = compr_level #9                     # update compression level
                var_encoding[var]['chunksizes'] = curr_chunk_size
                var_encoding[var]['zlib'] = True
            
            # Output
            encoding = {}
            encoding.update(coord_encoding)
            encoding.update(var_encoding)

            print("To save:", flush=True)
            print(ds_weekly, flush=True)

            # Save the dataset to a new NetCDF file
            start_time = time.time()
                        
            ds_weekly.to_netcdf(new_fpath, encoding=encoding)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f" Compression level: {compr_level}: Elapsed time: {elapsed_time} seconds", flush=True)
