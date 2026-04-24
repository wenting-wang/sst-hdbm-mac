#!/bin/bash -l
#SBATCH --job-name=TeSBI_E2E
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive                  
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=slurm_tesbi_e2e-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenting.wang.cd@outlook.com

# ==============================================================================
# run_tesbi_e2e.sh
#
# Unified SLURM submission script for the End-to-End TeSBI pipeline on Viper-GPU.
# 
# WORKFLOW:
# 1. Allocates 1 MI300A GPU Node + 64 CPU cores.
# 2. Stages code to high-speed scratch space (/ptmp).
# 3. Executes `tesbi_e2e.py --stage all`:
#    -> CPU Cores generate 30k datasets simultaneously.
#    -> PyTorch dynamically loads ROCm/GPU context.
#    -> GPU trains the Transformer + Zuko flow model.
#    -> GPU performs recovery and real data inference.
# 4. Safely copies all results back to your permanent directory.
# ==============================================================================

set -eo pipefail

# ==============================================================================
# 1. PIPELINE CONFIGURATION
# Modify these hyperparameters directly here to avoid touching the python code.
# ==============================================================================
# ##### TEST
# N_SIMS=30              # Total simulated datasets to generate using CPUs
# EPOCHS=5                # Total training epochs for the End-to-End model
# BATCH_SIZE=128            # Batch size for GPU training
# REC_K=5                  # Number of parameter recovery test cases
# REC_NUM_POST=10         # Number of posterior samples per recovery/inference
# SST_FOLDER="/u/wenwang/data/sst_valid_base" # Path to your real dataset directory

###### REAL RUN
N_SIMS=30000              # Total simulated datasets to generate using CPUs
EPOCHS=1000                # Total training epochs for the End-to-End model
BATCH_SIZE=128            # Batch size for GPU training
REC_K=200                 # Number of parameter recovery test cases
REC_NUM_POST=1000         # Number of posterior samples per recovery/inference
SST_FOLDER="/u/wenwang/data/sst_valid_base" # Path to your real dataset directory

# ==============================================================================
# 2. ENVIRONMENT SETUP
# ==============================================================================
# Recommended MPCDF thread hygiene (Prevents AMD node crashes from over-threading)
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Pass the allocated CPU cores to Joblib for parallel simulation
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "================================================================"

# Load Python & Custom ROCm (AMD GPU) Environment
module purge
module load rocm/7.0
module load python-waterboa/2025.06
source ~/rocm_env/bin/activate  

echo "Python Environment Activated. Checking AMD ROCm GPU Status:"
rocm-smi || echo "rocm-smi not available, moving on."

# ==============================================================================
# 3. SCRATCH SPACE I/O HANDLING
# ==============================================================================
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOB_ID}
mkdir -p "${SCRATCH_DIRECTORY}"
echo "Staging files to scratch directory: ${SCRATCH_DIRECTORY}"

# Copy all necessary source files to scratch
cp "${SLURM_SUBMIT_DIR}/tesbi_e2e.py" "${SCRATCH_DIRECTORY}/"
cp -r "${SLURM_SUBMIT_DIR}/core/"     "${SCRATCH_DIRECTORY}/"
cp -r "${SLURM_SUBMIT_DIR}/utils/"    "${SCRATCH_DIRECTORY}/"

# Move execution into scratch space
cd "${SCRATCH_DIRECTORY}"

# ==============================================================================
# 4. EXECUTE THE END-TO-END PIPELINE
# ==============================================================================
JOB_START_TIME=$(date +%s)
echo -e "\n>>> Starting End-to-End Pipeline execution at $(date)"

srun python3 -u tesbi_e2e.py \
    --stage all \
    --n_sims ${N_SIMS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --K ${REC_K} \
    --num_post ${REC_NUM_POST} \
    --sst_folder "${SST_FOLDER}"

echo -e "\n>>> Pipeline execution finished at $(date)"
JOB_END_TIME=$(date +%s)

# ==============================================================================
# 5. RETRIEVE RESULTS
# ==============================================================================
echo ">>> Copying results back to permanent storage..."

# Define your final permanent output directory
FINAL_OUT_DIR="${SLURM_SUBMIT_DIR}/outputs_run_${SLURM_JOB_ID}"
mkdir -p "${FINAL_OUT_DIR}"

# The python script dumps artifacts in 'outputs/' inside scratch
if [ -d "outputs/" ]; then
    cp -r outputs/* "${FINAL_OUT_DIR}/"
    echo "Results successfully copied to: ${FINAL_OUT_DIR}"
else
    echo "WARNING: Expected output directory 'outputs/' not found!"
    echo "Please inspect scratch directory manually: ${SCRATCH_DIRECTORY}"
fi

total_duration=$((JOB_END_TIME - JOB_START_TIME))
echo "================================================================"
echo "Job Completed Successfully."
echo "Total Runtime: $(($total_duration / 3600)) hrs $((($total_duration % 3600) / 60)) min $(($total_duration % 60)) sec."
echo "================================================================"
exit 0