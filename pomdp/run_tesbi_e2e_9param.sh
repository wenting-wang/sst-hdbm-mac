#!/bin/bash -l
#SBATCH --job-name=TeSBI_E2E_9Param
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive                  
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=slurm_tesbi_e2e_9p-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenting.wang.cd@outlook.com

# ==============================================================================
# run_tesbi_e2e.sh
#
# Unified SLURM submission script for the 9-Parameter TeSBI pipeline.
# 
# WORKFLOW:
# 1. Allocates 1 MI300A GPU Node + 48 CPU cores.
# 2. Stages code to high-speed scratch space (/ptmp).
# 3. Executes Step 1: `tesbi_e2e_9param.py` (Simulate -> Train -> Recover -> Infer)
# 4. Executes Step 2: `ppc_select_subject.py` (Calculate distance & select top 5%)
# 5. Executes Step 3: `plot_posterior_corner.py` (Generate 10k samples & plot corner)
# 6. Safely copies all results back to your permanent directory.
# ==============================================================================

set -eo pipefail

# ==============================================================================
# 1. PIPELINE CONFIGURATION
# ==============================================================================
##### TEST (Uncomment to test pipeline quickly)
# N_SIMS=30              
# EPOCHS=5               
# BATCH_SIZE=128         
# REC_K=5                
# REC_NUM_POST=10        
# SST_FOLDER="/u/wenwang/data/test"

# ###### REAL RUN
N_SIMS=100000             # Updated to 100,000 for the 9-parameter model
EPOCHS=1000               # Total training epochs
BATCH_SIZE=128            # Batch size for GPU training
REC_K=200                 # Number of parameter recovery test cases
REC_NUM_POST=1000         # Number of posterior samples per recovery/inference
SST_FOLDER="/u/wenwang/data/sst_valid_base" # Path to your real dataset directory

# ==============================================================================
# 2. ENVIRONMENT SETUP
# ==============================================================================
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "================================================================"

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
cp "${SLURM_SUBMIT_DIR}/tesbi_e2e_9param.py"    "${SCRATCH_DIRECTORY}/"
cp "${SLURM_SUBMIT_DIR}/ppc.py"  "${SCRATCH_DIRECTORY}/"
cp "${SLURM_SUBMIT_DIR}/plot_corner.py" "${SCRATCH_DIRECTORY}/"
cp -r "${SLURM_SUBMIT_DIR}/core/"               "${SCRATCH_DIRECTORY}/"
cp -r "${SLURM_SUBMIT_DIR}/utils/"              "${SCRATCH_DIRECTORY}/"

cd "${SCRATCH_DIRECTORY}"

# Dynamically patch the Python scripts to point to the actual cluster data folder
# (Replaces the local testing path with the real SST_FOLDER)
sed -i "s|./data/example_processed_data|${SST_FOLDER}|g" ppc.py
sed -i "s|./data/example_processed_data|${SST_FOLDER}|g" plot_corner.py

# ==============================================================================
# 4. EXECUTE THE END-TO-END PIPELINE (3 STEPS)
# ==============================================================================
JOB_START_TIME=$(date +%s)
echo -e "\n================================================================"
echo ">>> STEP 1: Running 9-Parameter E2E Pipeline at $(date)"
echo "================================================================"

srun python3 -u tesbi_e2e_9param.py \
    --stage all \
    --n_sims ${N_SIMS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --K ${REC_K} \
    --num_post ${REC_NUM_POST} \
    --sst_folder "${SST_FOLDER}"

echo -e "\n================================================================"
echo ">>> STEP 2: Running Posterior Predictive Checks & Selection at $(date)"
echo "================================================================"

srun python3 -u ppc.py --sst_folder "${SST_FOLDER}"

echo -e "\n================================================================"
echo ">>> STEP 3: Plotting Corner Matrix for Top Subject(s) at $(date)"
echo "================================================================"

srun python3 -u plot_corner.py --sst_folder "${SST_FOLDER}"

# ==============================================================================
# 5. RETRIEVE RESULTS
# ==============================================================================
echo ">>> Copying results back to permanent storage..."

FINAL_OUT_DIR="${SLURM_SUBMIT_DIR}/outputs_run_${SLURM_JOB_ID}"
mkdir -p "${FINAL_OUT_DIR}"

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