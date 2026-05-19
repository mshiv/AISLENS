#!/bin/bash
#SBATCH --job-name=ts-conv
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=1
#SBATCH --output=out.prep-mpaso.%j
#SBATCH --error=err.prep-mpaso.%j
#SBATCH --exclusive
#SBATCH --time=30:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis

cd ${HOME}/scratch

python convert-to-ts.py
