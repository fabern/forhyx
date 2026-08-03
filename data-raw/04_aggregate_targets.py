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
# TODO
pcwd_input_path = '/storage/capacity/occr_geco/data_2/archive/era5land_munoz-sabater_2021/data_derived_03_daily_pcwd.narm_v2-doy-reset_netcdf'

## some configureations
compr_level = 5
chunk_size_definition = {
  'valid_time': 4,   # in weekly resolution = 4 weeks
  'latitude':   18,  # if 1-deg resolution = 18 deg
  'longitude':  36,
  'pressure_level': 3, # NOTE: only used for pressure level ERA5 (UNUSED FOR TARGETS)
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

    # cluster = LocalCluster(n_workers=min(64, int(os.environ['SLURM_NPROCS'])), threads_per_worker=1)
    # client = Client(address=cluster)
    # TODO: reactivate this: cluster = LocalCluster(n_workers=min(64, int(os.environ['SLURM_CPUS_PER_TASK'])), threads_per_worker=1)
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    with Client(address=cluster) as client:
        print("Cluster setup: ", flush=True)
        print(client, flush=True)
        print(client.dashboard_link, flush=True)

        ### Step 2: Loop over different target variables
        vars = ['pcwd', 'sce']
        # curr_var = vars[0]
        for curr_var in vars:

            ### Step 3: Check presence/absence and validity of NetCDF Files
            if curr_var == 'pcwd':
                # List all NetCDF files in path
                print(pcwd_input_path, flush=True)
                file_pattern = f'*data_derived_03_daily_pcwd_v2-doy_*_r-generated.nc'
                netcdf_files = list_netcdf_files(pcwd_input_path, file_pattern)
                netcdf_files.sort()

                print("All files to treat:")
                pprint.pp(netcdf_files)

            ### Step 4: Open Multiple NetCDF Files
            # Use `xarray.open_mfdataset` to open multiple NetCDF files as a single dataset:
            ds_daily = xr.open_mfdataset(
                netcdf_files[0], # TODO
                combine='by_coords', 
                #engine="h5netcdf", 
                parallel =True, 
                compat='no_conflicts',
                join='exact',
                )

            print("\n\nFULL DAILY INPUT DATA SET TO AGGREGATE: ########################", flush=True)
            print(ds_daily, flush=True)

            ### Step 5: Aggregate in time (weekly) and in space (to regions)
            regions = gpd.read_file("data/regions/shapefile/regions.shp")
                # ERA5 data go from 0 to 360 degrees, adapt regions to that format
            
            # ensure there is a human-readable name column if you want one
            if 'region_name' not in regions.columns:
                regions['region_name'] = regions['region_id'].astype(str)

            
            ## Prepare ERA5Land data coordinates:
            # if coordinate is named `lon`; change to `longitude` if needed
            lons = ds_daily.lon.values
            # drop seam at 360° if present
            if np.isclose(lons.max(), 360.0):
                ds_daily = ds_daily.isel(lon=slice(0, -1))

            # wrap to -180..180 and sort
            ds_daily = ds_daily.assign_coords(lon=((ds_daily.lon + 180) % 360) - 180)
            ds_daily = ds_daily.sortby('lon')

            # guard: remove any remaining duplicates (keep first occurrence)
            wrapped = ds_daily.lon.values
            _, idx = np.unique(wrapped, return_index=True)
            if idx.size != wrapped.size:
                ds_daily = ds_daily.isel(lon=np.sort(idx))

            # Create mask with dimensions: region, lat, lon
            mask = regionmask.mask_3D_geopandas(
                regions, 
                ds_daily, 
                numbers='region_id', 
                drop=True)

            # Append regional mask to xarray
            ds_daily2 = ds_daily.where(mask)

                            # # Apply the mask to every spatial variable, then average over space
                            # regional_mean = ds_daily.where(mask).mean(
                            #     dim=("lat", "lon"),
                            #     skipna=True,
                            # )
                            # # Attach region names
                            # regional_mean = regional_mean.assign_coords(
                            #     region_name=("region", regions.loc[mask.region.values, "region_name"].values)
                            # )
                            # print(regional_mean)

            # Check output visually:
            plt.figure()
            ds_daily.sel(lon = slice(0,30), lat = slice(35,65)).isel(time=181)['pcwd_mm'].plot();
            plt.savefig('figure.png')
            plt.figure()
            # mask.sel(lon = slice(0,30), lat = slice(35,65), region = 2).plot();
            # mask.sel(lon = slice(0,30), lat = slice(35,65), region = 3).plot();
            mask.sel(lon = slice(0,30), lat = slice(35,65), region = 10).plot();
            plt.savefig('figure_regions.png')

            plt.figure()
            ds_daily2.sel(region = 10, lon = slice(0,30), lat = slice(35,65)).isel(time=181)['pcwd_mm'].plot();
            plt.savefig('figure_pcwd_region10.png')
            plt.figure()
            ds_daily2.sel(region = 9, lon = slice(0,30), lat = slice(35,65)).isel(time=181)['pcwd_mm'].plot();
            plt.savefig('figure_pcwd_region9.png')
            
            # # 1. Coarsen spatially FIRST (100x (or 16x) data reduction before temporal ops)
            # target_resolution = 1.0 # degrees

            # # get current resolution (either 0.25 (ERA5) or 0.1 (ERA5-Land)) 
            # curr_grid_resolution = np.diff(ds_daily.longitude[0:2].values)[0] # NOTE: assumes lon-lat grid is homogenous
            # # TODO: check this with ERA5-Land
            
            # factor = int(target_resolution/curr_grid_resolution)
            # ds_coarse = ds_daily.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()

            # # 2. Resample temporally LATER using built-in Dask-optimized resample
            # ds_weekly = ds_coarse.resample(valid_time="7D").mean()

            # print("Aggregated weekly:", flush=True)
            # print(ds_weekly, flush=True)

            # # NOTE: For rigorous scientific spatial aggregation, use cosine weighting to account for grid-area on sphere:
            # # UNUSED: # Cosine-latitude weighting prior to coarsening or regridding
            # # UNUSED: weights = np.cos(np.deg2rad(ds_daily.latitude))
            # # UNUSED: weights.name = "weights"
            # # UNUSED: ds_weighted = ds_daily.weighted(weights)
            # # UNUSED: ds_weekly2 = (
            # # UNUSED:     ds_weighted.coarsen(latitude=factor, longitude=factor, boundary="trim")
            # # UNUSED:     .mean()
            # # UNUSED:     .resample(valid_time="7D")
            # # UNUSED:     .mean()
            # # UNUSED: )

            # ### Step 6: Output as single netCDF file:
            # print("\n\nSTORE AS SINGLE netCDF FILE: ########################", flush=True)
            # fname_prefix = {
            #     # era5-pressure:
            #     "500.mean_z":  "FORHYX.targets.era5pressure",
            #     # era5-single:
            #     "mean_siconc": "FORHYX.targets.era5single",
            #     "mean_sst":    "FORHYX.targets.era5single",
            #     # era5-land:
            #     "mean_t2m":    "FORHYX.targets.era5land",
            #     "mean_d2m":    "FORHYX.targets.era5land",
            #     "mean_swvl1":  "FORHYX.targets.era5land",
            #     "mean_snowc":  "FORHYX.targets.era5land",
            #     }[curr_var]

            # start_year = pd.to_datetime(min(ds_weekly['valid_time'].values)).year
            # end_year = pd.to_datetime(max(ds_weekly['valid_time'].values)).year
            # year_range = f"{start_year}-{end_year}"

            # var_name = curr_var.replace(".", "_")
            # new_fpath = f"{out_directory}/{fname_prefix}.{var_name}.{year_range}.nc"
            # print(new_fpath, flush=True)

            # if curr_var == "500.mean_z":
            #     ds_weekly = (
            #         ds_weekly
            #         .sel(pressure_level=500)
            #         .reset_coords("pressure_level", drop=True)
            #         .rename({"mean_z": "500_mean_z"}) # NOTE: get rid of "."
            #     )

            # curr_chunk_size = tuple(            # this ensures correct order of coords and limit maximum chunk size
            #     min(
            #         chunk_size_definition[k],
            #         ds_weekly.sizes[k]
            #         )
            #     for k in ds_weekly.dims)
            # print("Target chunk size: ", curr_chunk_size, flush=True)
            
            # # Define the new NetCDF encoding
            # coord_encoding = {coord: ds_weekly[coord].encoding for coord in ds_weekly.coords}
            # var_encoding   = {var:   ds_weekly[var].encoding   for var   in ds_weekly.data_vars}
            # # Remove incompatible encoding parameters:
            # for coord in coord_encoding:
            #     coord_encoding[coord].pop('szip', None) # None results in removal if it exists
            #     coord_encoding[coord].pop('zstd', None) # None results in removal if it exists
            #     coord_encoding[coord].pop('bzip2', None) # None results in removal if it exists
            #     coord_encoding[coord].pop('blosc', None) # None results in removal if it exists
            # for var in var_encoding:
            #     var_encoding[var].pop('coordinates', None) # remove error in field 'coordinates'
            #     var_encoding[var].pop('szip', None)
            #     var_encoding[var].pop('zstd', None)
            #     var_encoding[var].pop('bzip2', None)
            #     var_encoding[var].pop('blosc', None)
            #     var_encoding[var].pop('preferred_chunks', None)
            # # Set 'complevel', and 'chunksizes' of all variables # https://stackoverflow.com/a/66333685
            # for var in var_encoding:
            #     var_encoding[var]['shuffle']    = True                               # update compression level
            #     var_encoding[var]['complevel']  = compr_level #9                     # update compression level
            #     var_encoding[var]['chunksizes'] = curr_chunk_size
            #     var_encoding[var]['zlib'] = True
            
            # # Output
            # encoding = {}
            # encoding.update(coord_encoding)
            # encoding.update(var_encoding)

            # print("To save:", flush=True)
            # print(ds_weekly, flush=True)

            # # Save the dataset to a new NetCDF file
            # start_time = time.time()
                        
            # ds_weekly.to_netcdf(new_fpath, encoding=encoding)
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(f" Compression level: {compr_level}: Elapsed time: {elapsed_time} seconds", flush=True)
