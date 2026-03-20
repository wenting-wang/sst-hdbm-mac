#!/bin/bash
#SBATCH --job-name=SURR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24           # 32 CPU cores are sufficient for single-GPU training
#SBATCH --mem=110000                 # Request 128GB RAM to prevent OOM when Pandas loads the dataset
#SBATCH --gres=gpu:1                 # Request 1 MI300A APU/GPU
#SBATCH --time=24:00:00
#SBATCH --output=slurm_surrogate-%j.out
#SBATCH --mail-type=END,FAIL         
#SBATCH --mail-user=wenting.wang.cd@outlook.com

# Recommended AI environment variables by MPCDF (prevents node crashes from excessive threading)
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Scratch Directory Setup for fast I/O
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p "${SCRATCH_DIRECTORY}"
echo "Running in scratch directory: ${SCRATCH_DIRECTORY}"

# Copy files to the scratch directory
# Note: Ensure the CSV, Python script, and 'core' folder are in the directory where you run sbatch
cp "${SLURM_SUBMIT_DIR}/train_e2e.py" .
cp "${SLURM_SUBMIT_DIR}/train_surrogate_v2.py" .
cp "${SLURM_SUBMIT_DIR}/pomdp_surrogate.pth" .
cp "${SLURM_SUBMIT_DIR}/orders.csv" .
cp -r "${SLURM_SUBMIT_DIR}/core/"    .

# Copy files to scratch
cd "${SCRATCH_DIRECTORY}"

# ==========================================
# Load the official ROCm Python environment
# ==========================================
module purge
module load python-waterboa/2025.06

# Activate your custom virtual environment
source ~/rocm_env/bin/activate  

# Run the training script
echo ">>> Starting model training..."
python -u train_e2e.py

# ==========================================
# Copy results back to the original directory
# ==========================================
# echo ">>> Training finished. Copying results back to the submit directory..."

# # Copy model weights, logs, and tensorboard runs
# cp *.pt *.pth *.csv *.log "${SLURM_SUBMIT_DIR}/" 2>/dev/null || true
# cp -r runs/ "${SLURM_SUBMIT_DIR}/" 2>/dev/null || true

# echo ">>> Copy complete."

# ==========================================
# Clean up (Commented out for debugging)
# ==========================================
# echo ">>> Cleaning up scratch directory..."
# rm -rf "${SCRATCH_DIRECTORY}"
echo ">>> Scratch cleanup bypassed. You can inspect the files at: ${SCRATCH_DIRECTORY}"
