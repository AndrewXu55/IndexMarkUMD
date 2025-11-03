#!/bin/bash
# ==============================================
# run_all_attacks_c2i.sh
# Master launcher for all C2I attack validation experiments
# Submits sbatch jobs with the correct algorithm/delta/resolution/attack combinations.
# NOTE: All algorithms compare watermarked vs baseline versions
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
echo "Directory structure:"
echo "  - Non-pairwise:"
echo "      Watermarked: {algorithm}-delta2.0-{resolution}"
echo "      Baseline:    {algorithm}-baseline-{resolution}"
echo "  - Pairwise (uses delta=0.0 for both):"
echo "      Watermarked: pairwise-{resolution}"
echo "      Baseline:    pairwise-baseline-{resolution}"
echo ""

# --- Iterate over all combinations ---
for algo in "${ALGORITHMS[@]}"; do
  for res in "${RESOLUTIONS[@]}"; do

    # Pairwise uses delta=0.0, others use delta=2.0
    if [[ $algo == "pairwise" ]]; then
      DELTA=0.0
    else
      DELTA=2.0
    fi

    JOB_NAME="attack_c2i_${algo}_d${DELTA}_${res}"
    LOG_FILE="slurm_logs/attacks_c2i/${JOB_NAME}.log"

    echo "Submitting: ${JOB_NAME}"
    sbatch --job-name=${JOB_NAME} \
           --output=${LOG_FILE} \
           --export=ALGORITHM=${algo},DELTA=${DELTA},RESOLUTION=${res} \
           attack_val_c2i_job.sh
  done

done

echo ""
echo "=============================================="
echo "All C2I attack validation jobs submitted!"
echo "=============================================="
echo "Total jobs: $(( ${#ALGORITHMS[@]} * ${#RESOLUTIONS[@]} ))"
echo "Each job runs all attack types internally"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs: slurm_logs/attacks_c2i/"
echo "Results: results/attacks_c2i/"
