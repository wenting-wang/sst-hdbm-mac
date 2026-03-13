#!/bin/bash
#SBATCH --job-name=SURR
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm_surrogate-%j.out

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Scratch Directory Setup for fast I/O
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p "${SCRATCH_DIRECTORY}"
echo "Running in scratch directory: ${SCRATCH_DIRECTORY}"

# Copy files to scratch
cd "${SCRATCH_DIRECTORY}"

# A) Copy scripts and metadata CSV from the submission directory
cp "${SLURM_SUBMIT_DIR}/train_surrogate.py" .
cp "${SLURM_SUBMIT_DIR}/pomdp_dataset_100k.csv" .
cp -r "${SLURM_SUBMIT_DIR}/core/"    .

source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate pymc_env

srun python -u train_surrogate.py