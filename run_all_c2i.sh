#!/bin/bash
# ==============================================
# run_all_c2i.sh
# Master launcher for all Class-to-Image experiments
# Submits sbatch jobs with the correct algorithm/delta combinations.
# ==============================================

# --- Experiment parameters ---
# ALGORITHMS=("random" "pairwise" "clustering" "spectral" "spectral-clustering")
ALGORITHMS=("clustering" "spectral" "spectral-clustering")
RESOLUTIONS=(256 384)

# --- Log setup ---
mkdir -p slurm_logs/c2i

echo "=============================================="
echo "Submitting Class-to-Image experiments"
echo "=============================================="

# --- Iterate over all algorithms and resolutions ---
for algo in "${ALGORITHMS[@]}"; do
  for res in "${RESOLUTIONS[@]}"; do

    # Determine valid deltas for this algorithm
    if [[ $algo == "random" ]]; then
      deltas=(0.0 2.0)
    elif [[ $algo == "pairwise" ]]; then
      deltas=(0.0)
    else
      deltas=(2.0)
    fi

    # Submit one job per valid delta
    for delta in "${deltas[@]}"; do
      JOB_NAME="c2i_${algo}_d${delta}_${res}"
      LOG_FILE="logs/c2i/${JOB_NAME}.log"
      echo "Submitting: ${JOB_NAME}"
      sbatch --output=${LOG_FILE} --export=ALGORITHM=${algo},DELTA=${delta},RESOLUTION=${res} c2i_job.sh
    done

  done
done

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo "Monitor with: squeue -u \$USER"
echo "Logs: slurm_logs/c2i/"
