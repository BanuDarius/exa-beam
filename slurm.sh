#!/bin/bash
#SBATCH --job-name=exa-beam
#SBATCH --output=log-%j.txt
#SBATCH --error=errors-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=256GB
#SBATCH --time=2:00:00

export OMP_PLACES=cores
export OMP_PROC_BIND=true

source $HOME/python-env/bin/activate

srun python3 auto_compute.py

deactivate