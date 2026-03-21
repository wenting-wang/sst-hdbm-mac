#!/bin/bash
#SBATCH --job-name=GEN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128          
#SBATCH --mem=128000                 
#SBATCH --time=12:00:00
#SBATCH --output=slurm_datagen-%j.out
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

cd "${SCRATCH_DIRECTORY}"

cp "${SLURM_SUBMIT_DIR}/train_e2e_v4_cpu.py" .
cp "${SLURM_SUBMIT_DIR}/orders.csv" .
cp -r "${SLURM_SUBMIT_DIR}/core/"    .

# ==========================================
# Load the official Python environment
# ==========================================
module purge
module load python-waterboa/2025.06

# Activate your custom virtual environment
source ~/rocm_env/bin/activate  

# Run the data generation script
echo ">>> Starting REAL POMDP Data Generation on CPU cores..."
srun python3 -u train_e2e_v4_cpu.py

# ==========================================
# Copy results back to the original directory
# ==========================================
echo ">>> Generation finished. Copying dataset back to the submit directory..."

# Copy generated dataset files (e.g., .pt files) back to the original directory
cp *.pt "${SLURM_SUBMIT_DIR}/" 

echo ">>> Copy complete."
echo ">>> Scratch cleanup bypassed. You can inspect the files at: ${SCRATCH_DIRECTORY}"