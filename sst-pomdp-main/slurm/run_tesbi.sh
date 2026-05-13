#!/bin/bash -l
#SBATCH --job-name=TeSBI
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=20-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com   # <-- UPDATE THIS
#SBATCH --output=slurm-%j.out

# ==============================================================================
# run_tesbi.sh
#
# HPC SLURM submission script for the TeSBI pipeline.
#
# USAGE:
#   This is the recommended way to run the full, computationally heavy TeSBI 
#   pipeline. Submit this script to your SLURM cluster using:
#       sbatch run_tesbi.sh
#
#   Note: If you only want to test the pipeline locally, use the Python 
#   commands outlined in `tesbi.py`. However, local runs are only recommended 
#   for quick smoke tests, not full production analyses.
# ==============================================================================

set -eo pipefail

# ==============================================================================
# 1. CONFIGURATION
# Uncomment one block below depending on your run type (Smoke Test OR Real Run).
# ==============================================================================

# --- OPTION A: SMOKE TEST (Fast, for debugging configuration) ---
# N1_PRE=100
# EPOCHS=5
# PATIENCE=1           
# N2=100
# DENSITY=maf            
# REC_K=10
# REC_NUM_POST=100
# NUM_SAMPLES_INF=100   

# --- OPTION B: REAL RUN (Production, computationally heavy) ---
N1_PRE=30000
EPOCHS=100
PATIENCE=15           
N2=20000
DENSITY=nsf              
REC_K=500
REC_NUM_POST=2000
NUM_SAMPLES_INF=2000    

# --- DATA PATHS ---
# Update this to point to the example data directory or your own dataset.
# The example dataset is located in `data/example_processed_data/`
SST_FOLDER="/path/to/sst-pomdp-main/data/example_processed_data"  # <-- UPDATE THIS PATH
GLOB_PAT="EXAMPLE_SUB_*.csv"


# ==============================================================================
# 2. SETUP & ENVIRONMENT
# ==============================================================================

# Threading Hygiene to prevent numpy/scipy from over-subscribing cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Scratch Directory Setup for fast I/O
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p "${SCRATCH_DIRECTORY}"
echo "Running in scratch directory: ${SCRATCH_DIRECTORY}"

# Copy source files to the scratch directory
cd "${SCRATCH_DIRECTORY}"
cp -r "${SLURM_SUBMIT_DIR}/core/"    .
cp -r "${SLURM_SUBMIT_DIR}/utils/"   .
cp "${SLURM_SUBMIT_DIR}/tesbi.py"    .
# cp "${SLURM_SUBMIT_DIR}/check_gpu.py" . # Optional utility

mkdir -p outputs

# Activate Environment
# Update the path to your Conda installation if necessary
source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate sst-pomdp-env

unset CUDA_VISIBLE_DEVICES
echo "Running nvidia-smi to confirm hardware visibility:"
nvidia-smi


# ==============================================================================
# 3. EXECUTION STAGES
# ==============================================================================

JOB_START_TIME=$(date +%s)

# --- Stage 1: Pretrain Encoder ---
echo "----------------------------------------------------------------"
echo "[Stage 1] Pretraining Encoder... Started at $(date)"
SECONDS=0

srun python -u tesbi.py \
  --stage pretrain \
  --n1_pre ${N1_PRE} \
  --epochs ${EPOCHS} \
  --patience ${PATIENCE}

duration=$SECONDS
echo "[Stage 1] Finished. Duration: $(($duration / 60)) min $(($duration % 60)) sec."

# --- Stage 2: Train SNPE ---
echo "----------------------------------------------------------------"
echo "[Stage 2] Training SNPE... Started at $(date)"
SECONDS=0

srun python -u tesbi.py \
  --stage snpe \
  --n1_pre ${N1_PRE} \
  --n2 ${N2} \
  --density ${DENSITY}

duration=$SECONDS
echo "[Stage 2] Finished. Duration: $(($duration / 60)) min $(($duration % 60)) sec."

# --- Stage 3: Recovery Check ---
echo "----------------------------------------------------------------"
echo "[Stage 3] Recovery Sweep... Started at $(date)"
SECONDS=0

srun python -u tesbi.py \
  --stage recover \
  --K ${REC_K} \
  --num_post ${REC_NUM_POST}

duration=$SECONDS
echo "[Stage 3] Finished. Duration: $(($duration / 60)) min $(($duration % 60)) sec."

# --- Stage 4: Inference on Real/Example Data ---
echo "----------------------------------------------------------------"
echo "[Stage 4] Data Inference... Started at $(date)"
SECONDS=0

if [[ -n "${SST_FOLDER}" && -d "${SST_FOLDER}" ]]; then
  srun python -u tesbi.py \
    --stage posterior \
    --sst_folder "${SST_FOLDER}" \
    --glob_pat "${GLOB_PAT}" \
    --num_samples ${NUM_SAMPLES_INF}
    
  duration=$SECONDS
  echo "[Stage 4] Finished. Duration: $(($duration / 60)) min $(($duration % 60)) sec."
else
  echo "[Stage 4] Skipped. Data folder not found or not specified."
fi

# ==============================================================================
# 4. FINAL SUMMARY
# ==============================================================================
JOB_END_TIME=$(date +%s)
total_duration=$((JOB_END_TIME - JOB_START_TIME))
echo "================================================================"

# (Optional) Copy results back to submit directory
# # Define your final permanent output directory
# FINAL_OUT_DIR="/kyb/agpd/wwang/outputs/"
# mkdir -p "${FINAL_OUT_DIR}"
# echo "Saving specific results to ${FINAL_OUT_DIR}..."
# # Define the exact files you want to extract from the scratch directory
# FILES_TO_COPY=(
#     "outputs/encoder.pt"
#     "outputs/params_recovery.csv"
#     "outputs/params_posteriors.csv"
# )
# # Loop through and copy each file, checking if it exists first
# for file in "${FILES_TO_COPY[@]}"; do
#     if [ -f "$file" ]; then
#         cp "$file" "${FINAL_OUT_DIR}/"
#         echo "  -> Successfully copied: $(basename "$file")"
#     else
#         echo "  -> Warning: $file not found in scratch directory!"
#     fi
# done

echo "Job Finished. Results are NOT automatically copied back in this configuration."
echo "You can find your outputs here: ${SCRATCH_DIRECTORY}"

echo "Job Completed Successfully."
echo "Total Runtime: $(($total_duration / 3600)) hrs $((($total_duration % 3600) / 60)) min $(($total_duration % 60)) sec."
exit 0