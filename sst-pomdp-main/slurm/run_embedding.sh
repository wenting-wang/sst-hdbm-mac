#!/bin/bash -l
#SBATCH --job-name=GEN_EMBED
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm_embed-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com   # <-- UPDATE THIS

# ==============================================================================
# run_embedding.sh
#
# HPC SLURM submission script for generating subject embeddings.
#
# USAGE:
#   Submit this script to your SLURM cluster using:
#       sbatch run_embedding.sh
#
#   Note: If you only want to test the embedding generation locally on the 
#   public example dataset, you can run `python embedding.py` directly.
# ==============================================================================

set -eo pipefail

# ==============================================================================
# 1. CONFIGURATION
# Update these paths to point to your cluster directories or example data.
# ==============================================================================

# --- Data & Metadata Paths ---
# Data source folder (Ensure this matches your preprocessing/filter setup)
SST_FOLDER="/path/to/sst-pomdp-main/data/example_processed_data"  # <-- UPDATE THIS
CSV_FILE="example_clinical_behavior.csv"                          # Subject list (expected in submit dir)

# --- Output Paths ---
OUTPUT_FILE="embeddings.csv"                     # Output filename
OUTPUT_DIR="${SLURM_SUBMIT_DIR}"                 # Final destination for results

# --- Model Path ---
# Absolute path to your existing trained model (encoder.pt)
# NOTE: This path must exist before the job starts!
SOURCE_MODEL_PATH="/path/to/sst-pomdp-main/outputs/encoder.pt"    # <-- UPDATE THIS

# --- Environment ---
CONDA_ENV="sst-pomdp-env"


# ==============================================================================
# 2. SETUP & DEPENDENCIES
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

# Copy files to scratch
cd "${SCRATCH_DIRECTORY}"

# A) Copy scripts and metadata CSV from the submission directory
cp "${SLURM_SUBMIT_DIR}/embedding.py" .
cp -r "${SLURM_SUBMIT_DIR}/utils/"    .
cp "${SLURM_SUBMIT_DIR}/${CSV_FILE}"  .

# B) Copy the trained model artifact and rename it locally to 'encoder.pt'
echo "Copying model from ${SOURCE_MODEL_PATH}..."
if [ -f "${SOURCE_MODEL_PATH}" ]; then
    cp "${SOURCE_MODEL_PATH}" ./encoder.pt
else
    echo "ERROR: Model not found at ${SOURCE_MODEL_PATH}"
    exit 1
fi

# Activate Environment (Update path if necessary for your cluster)
source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"


# ==============================================================================
# 3. EXECUTION
# ==============================================================================

echo "----------------------------------------------------------------"
echo "[Embedding Generation] Started at $(date)"

# Run the python script with unbuffered output (-u)
# We pass the metadata CSV, raw data folder, local encoder path, and output filename.
srun python -u embedding.py \
    --csv "${CSV_FILE}" \
    --folder "${SST_FOLDER}" \
    --model_path "./encoder.pt" \
    --out "${OUTPUT_FILE}"

echo "[Embedding Generation] Finished at $(date)"
echo "----------------------------------------------------------------"


# ==============================================================================
# 4. RESULTS EXPORT & CLEANUP
# ==============================================================================

echo "Saving results to ${OUTPUT_DIR}..."

# Copy the output CSV back to the submit directory
if [ -f "${OUTPUT_FILE}" ]; then
    cp "${OUTPUT_FILE}" "${OUTPUT_DIR}/"
    echo "Results copied successfully to ${OUTPUT_DIR}/${OUTPUT_FILE}"
else
    echo "ERROR: Output file ${OUTPUT_FILE} not found in scratch."
fi

# Cleanup is intentionally skipped for this job, but we print the location for reference
echo "Scratch directory preserved at: ${SCRATCH_DIRECTORY}"

exit 0