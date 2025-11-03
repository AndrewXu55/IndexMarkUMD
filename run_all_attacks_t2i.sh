#!/bin/bash
#SBATCH --time=3-00:00:00
# ==============================================
# run_all_attacks_t2i.sh
# Master launcher for all T2I attack validation experiments
# Submits sbatch jobs with the correct algorithm/delta/resolution/attack combinations.
# NOTE: All algorithms compare watermarked vs baseline versions
# ==============================================

# --- Experiment parameters ---
ALGORITHMS=("random" "pairwise" "clustering" "spectral" "spectral-clustering")
RESOLUTIONS=(256 512)  # Only 256 and 512 have index encoders for T2I

# --- Log setup ---
mkdir -p slurm_logs/attacks_t2i
mkdir -p results/attacks_t2i

echo "=============================================="
echo "Submitting T2I Attack Validation experiments"
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

    JOB_NAME="attack_${algo}_d${DELTA}_${res}"
    LOG_FILE="slurm_logs/attacks_t2i/${JOB_NAME}.log"

    echo "Submitting: ${JOB_NAME}"
    sbatch --job-name=${JOB_NAME} \
           --output=${LOG_FILE} \
           --export=ALGORITHM=${algo},DELTA=${DELTA},RESOLUTION=${res} \
           attack_val_t2i_job.sh
  done

done

echo ""
echo "=============================================="
echo "All attack validation jobs submitted!"
echo "=============================================="
echo "Total jobs: $(( ${#ALGORITHMS[@]} * ${#RESOLUTIONS[@]} ))"
echo "Each job runs all attack types internally"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs: slurm_logs/attacks_t2i/"
echo "Results: results/attacks_t2i/"
