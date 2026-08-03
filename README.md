# FORHYX

This repository contains the code for the FORHYX project

## Use

Data preparation was done by running:
    - `data-raw/01_download_scalar_indices.py`.
    - `data-raw/02_aggregate_predictors.py`.
    - `data-raw/03_compute_targets.py`.
    - `data-raw/04_aggregate_targets.py`.

Required input:
    - TODO: document
Result:
    - `data/predictors.zarr`
    - TODO: `data/targets.zarr`

## Structure

### The data folder

The `data` folder contains analysis ready data. This is data which you can use,
as is. It contains the output of a `data-raw` pre-processing workflow.

```
data/
├─ targets/
├─ predictors/
```

## Computing environment

Use a conda environment specified by `environment_forhyx.yml`.

### Install Miniconda

On UBELIX this is not needed. Simply do: `module load Anaconda3`
On a local machine, do:
Feel free to change `~/miniconda3` to a different one you prefer.
```
mkdir ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

### Install dependencies

It should be enough on a new machine to do:
```
conda env create --file environment_forhyx.yml
```
If that doesn't work, try installing the packages without enforcing specific versions:
```
conda env create --file environment_forhyx_from_history.yml
```

If that doesn't work either create your own environment manually following the steps below, which were used to setup the initial environment.
The `environment_forhyx.yml` was created with:
```
# on UBELIX: srun --account=invest --qos=job_icpu-stocker --ntasks=1 --cpus-per-task=2 --mem-per-cpu=6G --job-name="setup_conda" --time=1:00:00 --pty bash
# on UBELIX: module load cURL/8.11.1-GCCcore-14.2.0 OpenSSL/3  # on UBELIX HPC
# on UBELIX: module load Anaconda3                             # on UBELIX HPC
# on UBELIX: eval "$(conda shell.bash hook)"                   # on UBELIX HPC

cd ~/GitHub/fabern/forhyx

conda create -n forhyx python=3.14 -c conda-forge
conda activate forhyx
conda install xarray dask distributed dask-core netcdf4 h5netcdf h5py cftime numpy pandas
conda install matplotlib geopandas regionmask
conda install cartopy scikit-image metpy

conda env export > environment_forhyx.yml
conda export --from-history > environment_forhyx_from_history.yml
```

### Run code

After setup of the environment (see above) python code can be run by 
activating the conda environment.

To activate conda in the current shell use the following command:
```
conda activate forhyx
# run python
conda deactivate
```

For UBELIX use the following snippet for interactive python execution:
```
srun --account=invest --qos=job_icpu-stocker --ntasks=1 --cpus-per-task=34 --mem-per-cpu=3G --job-name="dev_aggregation --time=8:00:00 --pty bash
module load cURL/8.11.1-GCCcore-14.2.0 OpenSSL/3
module load Anaconda3
eval "$(conda shell.bash hook)"

conda activate forhyx
# # run python
# python3
conda deactivate
```
or then use bash scripts based on the above.