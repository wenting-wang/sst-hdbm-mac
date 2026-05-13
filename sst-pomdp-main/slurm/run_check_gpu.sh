#!/bin/bash
#SBATCH --job-name=gpu_fix
#SBATCH --output=debug_fix.out
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00

module purge
# module load anaconda3 (or whatever you use)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sst-pomdp-env

echo "Attempting to force GPU visibility..."
unset CUDA_VISIBLE_DEVICES

echo "Running with Python..."
python check_gpu.py