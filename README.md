# FORHYX

This repository contains the code for the FORHYX project

## Use

Run `data-raw/pre-process.py`.

## Structure

### The data folder

The `data` folder contains analysis ready data. This is data which you can use,
as is. It contains the output of a `data-raw` pre-processing workflow.

```
data/
├─ targets/
├─ predictors/
```

### Capturing your environment

Use a conda environment specified by 

#### Install Miniconda

On UBELIX this is not needed. Simply do: `module load Anaconda3`
On a local machine, do:
Feel free to change `~/miniconda3` to a different one you prefer.
```
mkdir ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

#### Install dependencies
It should be enough on a new machine to do:
```
conda create --file environment_forhyx.yml
```

If that doesn't work create your own environment.
The `enviornment_forhyx.yml` was created with:
```
# on UBELIX: srun --account=invest --qos=job_icpu-stocker --ntasks=1 --cpus-per-task=2 --mem-per-cpu=6G --job-name="setup_conda" --time=1:00:00 --pty bash
# on UBELIX: module load cURL/8.11.1-GCCcore-14.2.0 OpenSSL/3  # on UBELIX HPC
# on UBELIX: module load Anaconda3                             # on UBELIX HPC
# on UBELIX: eval "$(conda shell.bash hook)"                   # on UBELIX HPC

cd ~/GitHub/fabern/forhyx

conda create -n forhyx python=3.14 -c conda-forge
conda activate forhyx
conda install xarray dask distributed dask-core netcdf4 h5netcdf h5py cftime numpy pandas

conda env export > enviornment_forhyx.yml
conda export --from-history > enviornment_forhyx_from_history.yml
```

#### Use conda
To activate Miniconda in the current shell use the following command:

`source ~/miniconda3/bin/activate`

To deactivate it run

`conda deactivate`

### Installing Dependencies

Install gdal dependency separately.

`conda install -y -c conda-forge gdal zarr`

Go to the `python` directory of the project and install the needed requirements using

`pip install -r requirements.txt`