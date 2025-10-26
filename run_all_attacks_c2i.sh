#!/bin/bash
# ==============================================
# run_all_attacks_c2i.sh
# Master launcher for all C2I attack validation experiments
# Submits sbatch jobs with the correct algorithm/delta/resolution/attack combinations.
# ==============================================

# --- Experiment parameters ---
ALGORITHMS=("random" "pairwise" "clustering" "spectral" "spectral-clustering")
RESOLUTIONS=(256 384)  # Only 256 and 384 have index encoders for C2I

# --- Log setup ---
mkdir -p slurm_logs/attacks_c2i
mkdir -p results/attacks_c2i

echo "=============================================="
echo "Submitting C2I Attack Validation experiments"
echo "=============================================="

# --- Iterate over all combinations ---
for algo in "${ALGORITHMS[@]}"; do

  # Determine valid deltas for this algorithm
  if [[ $algo == "random" ]]; then
    deltas=(2.0)  # Skip baseline (0.0) - we only need it as the comparison reference
  elif [[ $algo == "pairwise" ]]; then
    deltas=(0.0)
  else
    deltas=(2.0)
  fi

  for delta in "${deltas[@]}"; do
    for res in "${RESOLUTIONS[@]}"; do

      JOB_NAME="attack_c2i_${algo}_d${delta}_${res}"
      LOG_FILE="slurm_logs/attacks_c2i/${JOB_NAME}.log"

      echo "Submitting: ${JOB_NAME}"
      sbatch --job-name=${JOB_NAME} \
             --output=${LOG_FILE} \
             --export=ALGORITHM=${algo},DELTA=${delta},RESOLUTION=${res} \
             attack_val_c2i_job.sh
    done
  done

done

echo ""
echo "=============================================="
echo "All C2I attack validation jobs submitted!"
echo "=============================================="
echo "Total algorithms: ${#ALGORITHMS[@]}"
echo "Total resolutions: ${#RESOLUTIONS[@]}"
echo "Each job will run all attack types internally"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs: slurm_logs/attacks_c2i/"
echo "Results: results/attacks_c2i/"
