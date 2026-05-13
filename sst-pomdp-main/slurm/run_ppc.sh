#!/bin/bash -l

#SBATCH --job-name=PPC_SIM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=20-00:00:00
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com   # <-- UPDATE THIS
#SBATCH --output=slurm-ppc-%j.out

# ==============================================================================
# run_ppc.sh
#
# HPC SLURM submission script for the Posterior Predictive Checks (PPC).
#
# USAGE:
#   Submit this script to your SLURM cluster using:
#       sbatch run_ppc.sh
#
#   Note: Ensure that you have updated `ppc.py` (OPTION A configuration) 
#   with your absolute HPC paths before submitting this job.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -eo pipefail

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
ENV_NAME="sst-pomdp-env"
SCRIPT_NAME="ppc.py"
OUTPUT_FILE="ppc_metrics.csv" 
# Note: Ensure this OUTPUT_FILE name matches the 'OUTPUT_PPC_CSV' filename defined in ppc.py

# ==============================================================================
# 2. SETUP & DEPENDENCIES
# ==============================================================================
# Define and create a unique scratch directory for fast I/O
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p "${SCRATCH_DIRECTORY}"
echo "Running in scratch directory: ${SCRATCH_DIRECTORY}"

cd "${SCRATCH_DIRECTORY}"

# Copy all necessary python scripts and modules to the scratch location
cp -r "${SLURM_SUBMIT_DIR}/core/"    .
cp -r "${SLURM_SUBMIT_DIR}/utils/"   .
cp "${SLURM_SUBMIT_DIR}/${SCRIPT_NAME}" .

# ==============================================================================
# 3. ENVIRONMENT & EXECUTION
# ==============================================================================
# Activate Anaconda environment (Update path if necessary for your cluster)
source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "----------------------------------------------------------------"
echo "[PPC Simulation] Started at $(date)"

# Execute the script
python -u ${SCRIPT_NAME}

echo "[PPC Simulation] Finished at $(date)"
echo "----------------------------------------------------------------"

# ==============================================================================
# 4. RESULTS & CLEANUP
# ==============================================================================
# Note: If ppc.py is configured to write to an absolute path (e.g., /home/user/outputs/), 
# it bypasses the scratch directory, making this copy step redundant but harmless.
# If it writes locally to scratch, this ensures the results are safely brought back.

if [ -f "${OUTPUT_FILE}" ]; then
    cp "${OUTPUT_FILE}" "${SLURM_SUBMIT_DIR}/"
    echo "Results copied successfully to ${SLURM_SUBMIT_DIR}"
else
    echo "Notice: Output file ${OUTPUT_FILE} not found in scratch."
    echo "If you used absolute paths in ppc.py, your file is already saved at that location."
fi

# Clean up scratch directory to free up cluster node space
cd "${SLURM_SUBMIT_DIR}"
rm -rf "${SCRATCH_DIRECTORY}"
echo "Scratch directory cleaned up."

exit 0