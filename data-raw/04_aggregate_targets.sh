#! /usr/bin/bash -l
#SBATCH --job-name="04_FORHYX_aggregate_targets"
#SBATCH --time=01:00:00

#SBATCH --account=invest          ### invest           ### FOR DEVELOPMENT: gratis
#SBATCH --qos=job_icpu-stocker    ### job_icpu-stocker ### FOR DEVELOPMENT: job_debug (AND ensure not to use capacity storage!)
#SBATCH --ntasks=1             # nr of tasks (processes), used for MPI jobs that may run distributed on multiple compute nodes
#SBATCH --cpus-per-task=16      # nr of threads, used for shared memory jobs that run locally on a single compute node (default: 1)
#SBATCH --mem-per-cpu=6G

#SBATCH --mail-user=fabian.bernhard@unibe.ch
#SBATCH --mail-type=NONE      ### BEGIN,END,FAIL
#SBATCH --chdir=GitHub/fabern/forhyx/data-raw  # ensure to omit tilde ~/. Defines the working directory (relative to working directory where you submitted job via `sbatch`). This directory contains your script, and where the --output will be written
#SBATCH --output=slurm-%x_%j.out        # --output relative to working directory (takes into account --chdir)

# Taken from Github/fabern/aggregate-era5land-daily on 2026-07-19: proc.sh
echo "Started on: $(date --rfc-3339=seconds)"
echo "Hostname: $(hostname)"

echo "SLURM Job ID: ${SLURM_JOBID}"

module load cURL/8.11.1-GCCcore-14.2.0 OpenSSL/3  # on UBELIX HPC
module load Anaconda3                             # on UBELIX HPC
eval "$(conda shell.bash hook)"                   # on UBELIX HPC

echo "Current directory:"
pwd

cd ~/GitHub/fabern/forhyx/data-raw

echo "Start python script: $(date --rfc-3339=seconds)"
conda activate forhyx
python3 04_aggregate_targets.py
conda deactivate

echo "Finished on: $(date --rfc-3339=seconds)"

# NOTE: if we want to follow the progress, we could do so with the dask scheduler.
#  - run another command on ubelix to forward the corresponding port from worker to submit node: fb24k097@submit04:~$ ssh -L 8787:localhost:8787 cnode16
#  - and also from the submit node forward that port to your local machine (e.g. using the Ports tab in VSCode)