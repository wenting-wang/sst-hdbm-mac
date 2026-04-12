#!/bin/bash
#SBATCH --job-name=SIM_COMP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128          
#SBATCH --mem=128000     
#SBATCH --time=24:00:00              # 24 hours allocated
#SBATCH --output=slurm_sim_comp-%j.out
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

# ==========================================
# 2. Copy files to Scratch
# ==========================================
cp -r "${SLURM_SUBMIT_DIR}/core/"    .
cp "${SLURM_SUBMIT_DIR}/subject_local_trends.csv" .
cp "${SLURM_SUBMIT_DIR}/pomdp_posterior.csv" .
cp "${SLURM_SUBMIT_DIR}/est_param_additive_2.csv" .
cp "${SLURM_SUBMIT_DIR}/orders.csv" .
cp "${SLURM_SUBMIT_DIR}/simulate_and_compare_v2.py" .

# ==========================================
# 3. Load Environment
# ==========================================
module purge
module load python-waterboa/2025.06
source ~/rocm_env/bin/activate  

# ==========================================
# 4. Run the simulation
# ==========================================
echo ">>> Starting Model Simulation and Comparison..."
srun python3 -u simulate_and_compare_v2.py

# ==========================================
# 5. Copy results back
# ==========================================
echo ">>> Simulation finished. Copying plots back to submit directory..."

# Copy the resulting archetypes plot to the original outputs directory
mkdir -p "${SLURM_SUBMIT_DIR}/outputs"
cp outputs/*.png "${SLURM_SUBMIT_DIR}/outputs/" 2>/dev/null

# Fallback: copy any png files generated in the root directory just in case
cp *.png "${SLURM_SUBMIT_DIR}/" 2>/dev/null

echo ">>> Copy complete. Check your outputs folder!"