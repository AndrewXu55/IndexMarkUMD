#!/bin/bash
# ==============================================
# run_all_t2i.sh
# Master launcher for all Text-to-Image experiments
# Submits sbatch jobs for each algorithm/resolution combination.
# NOTE: Each job now generates BOTH baseline and watermarked versions automatically.
# ==============================================

# --- Experiment parameters ---
ALGORITHMS=("random" "pairwise" "clustering" "spectral" "spectral-clustering")
RESOLUTIONS=(256 512)

# --- Log setup ---
mkdir -p slurm_logs/t2i

echo "=============================================="
echo "Submitting Text-to-Image experiments"
echo "=============================================="
echo "NOTE: Each job generates BOTH baseline and watermarked versions"
echo "  - Non-pairwise:"
echo "      Baseline: {algorithm}-baseline-{resolution}"
echo "      Watermarked: {algorithm}-delta2.0-{resolution}"
echo "  - Pairwise:"
echo "      Baseline: pairwise-baseline-{resolution}"
echo "      Watermarked: pairwise-{resolution} (both use delta=0.0)"
echo ""

# --- Iterate over all algorithms and resolutions ---
for algo in "${ALGORITHMS[@]}"; do
  for res in "${RESOLUTIONS[@]}"; do

    # Pairwise uses delta=0.0 for both baseline and watermarked
    if [[ $algo == "pairwise" ]]; then
      DELTA=0.0
    else
      DELTA=2.0
    fi

    JOB_NAME="t2i_${algo}_d${DELTA}_${res}"
    LOG_FILE="slurm_logs/t2i/${JOB_NAME}.log"
    echo "Submitting: ${JOB_NAME} (generates both baseline and watermarked)"
    sbatch --job-name=${JOB_NAME} \
           --output=${LOG_FILE} \
           --export=ALGORITHM=${algo},DELTA=${DELTA},RESOLUTION=${res} \
           t2i_job.sh

  done
done

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo "Total jobs: $(( ${#ALGORITHMS[@]} * ${#RESOLUTIONS[@]} ))"
echo "Each job generates 2 directories (baseline + watermarked)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs: slurm_logs/t2i/"
