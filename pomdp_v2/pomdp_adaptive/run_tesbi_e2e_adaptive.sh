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
# run_tesbi_e2e_10param.sh
#
# Unified SLURM submission script for the 10-Parameter TeSBI pipeline (incl. tau)
# ==============================================================================

set -eo pipefail

# # ==============================================================================
# # 0. TEST PIPELINE CONFIGURATION
# # ==============================================================================
# N_SIMS=10             
# EPOCHS=2               
# BATCH_SIZE=128            
# REC_K=5                 
# REC_NUM_POST=10         

# # Update these two paths to match your cluster directories
# SST_FOLDER="/u/wenwang/data/test" 
# FILTER_CSV="/u/wenwang/data/clinical_behavior.csv"

# ==============================================================================
# 1. PIPELINE CONFIGURATION
# ==============================================================================
N_SIMS=100000             
EPOCHS=1000               
BATCH_SIZE=128            
REC_K=200                 
REC_NUM_POST=1000         

# Update these two paths to match your cluster directories
MODEL_FILE="tesbi_e2e_7p_v1"
SST_FOLDER="/u/wenwang/data/sst_valid_base" 
FILTER_CSV="/u/wenwang/data/clinical_behavior.csv"

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

cp "${SLURM_SUBMIT_DIR}/${MODEL_FILE}.py" "${SCRATCH_DIRECTORY}/"
cp "${SLURM_SUBMIT_DIR}/ppc_adaptive.py"              "${SCRATCH_DIRECTORY}/"
cp "${SLURM_SUBMIT_DIR}/plot_corner_adaptive.py"          "${SCRATCH_DIRECTORY}/"
cp "${SLURM_SUBMIT_DIR}/fig_ppc_reps_adaptive.py"     "${SCRATCH_DIRECTORY}/"
cp "${SLURM_SUBMIT_DIR}/fig_params_recovery_adaptive.py" "${SCRATCH_DIRECTORY}/"
cp -r "${SLURM_SUBMIT_DIR}/core/"                "${SCRATCH_DIRECTORY}/"
cp -r "${SLURM_SUBMIT_DIR}/utils/"               "${SCRATCH_DIRECTORY}/"

cd "${SCRATCH_DIRECTORY}"

# ==============================================================================
# 4. EXECUTE THE END-TO-END PIPELINE (4 STEPS)
# ==============================================================================
JOB_START_TIME=$(date +%s)
echo -e "\n================================================================"
echo ">>> STEP 1: Running E2E Pipeline at $(date)"
echo "================================================================"

srun python3 -u ${MODEL_FILE}.py \
    --stage all \
    --n_sims ${N_SIMS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --K ${REC_K} \
    --num_post ${REC_NUM_POST} \
    --sst_folder "${SST_FOLDER}"

echo -e "\n================================================================"
echo ">>> STEP 1.5: Plotting Parameter Recovery Grid at $(date)"
echo "================================================================"

srun python3 -u fig_params_recovery_adaptive.py --out_dir "outputs" --model_config ${MODEL_FILE}

echo -e "\n================================================================"
echo ">>> STEP 2: Running Posterior Predictive Checks at $(date)"
echo "================================================================"

srun python3 -u ppc_adaptive.py \
    --sst_folder "${SST_FOLDER}" \
    --model_config ${MODEL_FILE} \
    --filter_csv "${FILTER_CSV}"

echo -e "\n================================================================"
echo ">>> STEP 3: Plotting Corner Matrix at $(date)"
echo "================================================================"
# Assuming plot_corner_adaptive.py accepts --sst_folder. If it requires --csv, adjust as needed.
srun python3 -u plot_corner_adaptive.py --sst_folder "${SST_FOLDER}" --model_config ${MODEL_FILE} || echo "Corner plot skipped or failed."

echo -e "\n================================================================"
echo ">>> STEP 4: Plotting PPC Representations (Good/Mod/Poor) at $(date)"
echo "================================================================"

srun python3 -u fig_ppc_reps_adaptive.py \
    --sst_folder "${SST_FOLDER}" \
    --filter_csv "${FILTER_CSV}" \
    --model_config ${MODEL_FILE}

# ==============================================================================
# 5. RETRIEVE RESULTS
# ==============================================================================
echo ">>> Copying results back to permanent storage..."

FINAL_OUT_DIR="${SLURM_SUBMIT_DIR}/outputs_run_${SLURM_JOB_ID}_${MODEL_FILE}"
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