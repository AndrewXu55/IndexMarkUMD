#!/bin/bash
#SBATCH --job-name=attack_c2i
#SBATCH --output=slurm_logs/attacks_c2i/%x.out
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --time=3-00:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=4

# ==============================================
# attack_val_c2i_job.sh
# Inner SLURM script for C2I attack validation.
# Expects environment variables: ALGORITHM, DELTA, RESOLUTION
# ==============================================

echo "========================================="
echo "C2I Attack Validation Job started"
echo "Algorithm: ${ALGORITHM}"
echo "Delta: ${DELTA}"
echo "Resolution: ${RESOLUTION}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================="
source ~/.bashrc
conda activate graph_watermark
cd ~/GraphWatermark

# --- Attack types to run ---
ATTACK_TYPES=("none" "jpeg" "cropping" "blurring" "noise" "color_jitter")

# --- Paths ---
VQ_CKPT="/cmlscratch/anirudhs/hub/vq_ds16_c2i.pt"
INDEX_ENCODER="/cmlscratch/anirudhs/hub/Index_encoder_${RESOLUTION}_c2i.pt"
MAPPING_PATH="results/codebook_index_mapping_knn10_mwpm_c2i.json"
PAIRS_PATH="results/codebook_pairs_knn10_mwpm_c2i.json"
BASE_IMAGE_DIR="/cmlscratch/anirudhs/graph_watermark/images/c2i_experiments"
BASE_RESULTS_DIR="results/attacks_c2i"

# --- Determine directory names based on algorithm and delta ---
# NOT_WATERMARKED_DIR is always random-baseline for comparison
NOT_WATERMARKED_DIR="${BASE_IMAGE_DIR}/random-baseline-${RESOLUTION}"

if [[ $ALGORITHM == "pairwise" ]]; then
    # Pairwise uses just "pairwise-{resolution}" naming and no DIR_SUFFIX
    WATERMARKED_DIR="${BASE_IMAGE_DIR}/pairwise-${RESOLUTION}"
    RESULTS_DIR="${BASE_RESULTS_DIR}/${ALGORITHM}/${RESOLUTION}"
elif [[ $DELTA == "0.0" ]]; then
    WATERMARKED_DIR="${BASE_IMAGE_DIR}/${ALGORITHM}-baseline-${RESOLUTION}"
    DIR_SUFFIX="baseline"
    RESULTS_DIR="${BASE_RESULTS_DIR}/${ALGORITHM}/${DIR_SUFFIX}/${RESOLUTION}"
else
    WATERMARKED_DIR="${BASE_IMAGE_DIR}/${ALGORITHM}-delta${DELTA}-${RESOLUTION}"
    DIR_SUFFIX="delta${DELTA}"
    RESULTS_DIR="${BASE_RESULTS_DIR}/${ALGORITHM}/${DIR_SUFFIX}/${RESOLUTION}"
fi

# --- Create results directory ---
mkdir -p "${RESULTS_DIR}"

# --- Common args ---
COMMON_ARGS="--vq-ckpt ${VQ_CKPT} \
             --ft-pt-path ${INDEX_ENCODER} \
             --mapping-save-path ${MAPPING_PATH} \
             --pairs-save-path ${PAIRS_PATH} \
             --load-mapping \
             --load-pairs \
             --h 0 \
             --pvalue-threshold 0.01 \
             --algorithm ${ALGORITHM} \
             --seed 0 \
             --replacement-ratio 1.0 \
             --target-image-size ${RESOLUTION} \
             --distortion-seed 123"

# --- Loop through all attack types and strengths ---
for ATTACK_TYPE in "${ATTACK_TYPES[@]}"; do
    echo ""
    echo "========================================="
    echo "Running attack: ${ATTACK_TYPE}"
    echo "========================================="

    if [[ $ATTACK_TYPE == "none" ]]; then
        # No attack - run once
        RESULTS_FILE="${RESULTS_DIR}/attack_none.json"
        python attack_val.py \
            --Watermarked-dir "${WATERMARKED_DIR}" \
            --Not-Watermarked-dir "${NOT_WATERMARKED_DIR}" \
            --results-output "${RESULTS_FILE}" \
            ${COMMON_ARGS} \
            --chosen-attack none
        echo "Results saved to: ${RESULTS_FILE}"

    elif [[ $ATTACK_TYPE == "jpeg" ]]; then
        # JPEG: quality from 90 to 0 in increments of 10
        for quality in 90 80 70 60 50 40 30 20 10 0; do
            echo "  JPEG quality: ${quality}"
            RESULTS_FILE="${RESULTS_DIR}/attack_jpeg_q${quality}.json"
            python attack_val.py \
                --Watermarked-dir "${WATERMARKED_DIR}" \
                --Not-Watermarked-dir "${NOT_WATERMARKED_DIR}" \
                --results-output "${RESULTS_FILE}" \
                ${COMMON_ARGS} \
                --chosen-attack jpeg \
                --jpeg-attack-quality ${quality}
            echo "  Results saved to: ${RESULTS_FILE}"
        done

    elif [[ $ATTACK_TYPE == "blurring" ]]; then
        # Gaussian blur: kernel size from 1 to 19 in increments of 3
        for kernel in 1 4 7 10 13 16 19; do
            echo "  Blur kernel size: ${kernel}"
            RESULTS_FILE="${RESULTS_DIR}/attack_blur_k${kernel}.json"
            python attack_val.py \
                --Watermarked-dir "${WATERMARKED_DIR}" \
                --Not-Watermarked-dir "${NOT_WATERMARKED_DIR}" \
                --results-output "${RESULTS_FILE}" \
                ${COMMON_ARGS} \
                --chosen-attack blurring \
                --blur-kernel-size ${kernel}
            echo "  Results saved to: ${RESULTS_FILE}"
        done

    elif [[ $ATTACK_TYPE == "noise" ]]; then
        # Gaussian noise: std dev from 0.025 to 0.2 in increments of 0.025
        for noise_std in 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2; do
            echo "  Noise std fraction: ${noise_std}"
            # Replace . with _ for filename
            noise_label=$(echo ${noise_std} | sed 's/\./_/')
            RESULTS_FILE="${RESULTS_DIR}/attack_noise_std${noise_label}.json"
            python attack_val.py \
                --Watermarked-dir "${WATERMARKED_DIR}" \
                --Not-Watermarked-dir "${NOT_WATERMARKED_DIR}" \
                --results-output "${RESULTS_FILE}" \
                ${COMMON_ARGS} \
                --chosen-attack noise \
                --noise-std-fraction ${noise_std}
            echo "  Results saved to: ${RESULTS_FILE}"
        done

    elif [[ $ATTACK_TYPE == "color_jitter" ]]; then
        # Color jitter: brightness from 0.5 to 3 in increments of 0.5
        for brightness in 0.5 1.0 1.5 2.0 2.5 3.0; do
            echo "  Color jitter brightness: ${brightness}"
            # Replace . with _ for filename
            bright_label=$(echo ${brightness} | sed 's/\./_/')
            RESULTS_FILE="${RESULTS_DIR}/attack_colorjitter_b${bright_label}.json"
            python attack_val.py \
                --Watermarked-dir "${WATERMARKED_DIR}" \
                --Not-Watermarked-dir "${NOT_WATERMARKED_DIR}" \
                --results-output "${RESULTS_FILE}" \
                ${COMMON_ARGS} \
                --chosen-attack color_jitter \
                --color-jitter-brightness ${brightness}
            echo "  Results saved to: ${RESULTS_FILE}"
        done

    elif [[ $ATTACK_TYPE == "cropping" ]]; then
        # Crop: scale from 90 to 50 in increments of 10 (0.90, 0.80, 0.70, 0.60, 0.50)
        for scale_pct in 90 80 70 60 50; do
            scale=$(echo "scale=2; ${scale_pct} / 100" | bc)
            echo "  Crop scale: ${scale} (${scale_pct}%)"
            RESULTS_FILE="${RESULTS_DIR}/attack_crop_s${scale_pct}.json"
            python attack_val.py \
                --Watermarked-dir "${WATERMARKED_DIR}" \
                --Not-Watermarked-dir "${NOT_WATERMARKED_DIR}" \
                --results-output "${RESULTS_FILE}" \
                ${COMMON_ARGS} \
                --chosen-attack cropping \
                --crop-scale ${scale}
            echo "  Results saved to: ${RESULTS_FILE}"
        done
    fi
done

echo ""
echo "========================================="
echo "All attacks completed at: $(date)"
echo "========================================="
