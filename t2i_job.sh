#!/bin/bash
#SBATCH --job-name=t2i_gen
#SBATCH --output=slurm_logs/t2i/%x.out
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --time=1-00:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=4

# ==============================================
# t2i_job.sbatch
# Inner SLURM script for Class-to-Image generation.
# Expects environment variables: ALGORITHM, DELTA, RESOLUTION
# ==============================================

echo "========================================="
echo "Job started"
echo "Algorithm: ${ALGORITHM}"
echo "Delta: ${DELTA}"
echo "Resolution: ${RESOLUTION}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================="
source ~/.bashrc
conda activate graph_watermark
cd ~/GraphWatermark
# --- Paths ---
VQ_CKPT="/cmlscratch/anirudhs/hub/vq_ds16_t2i.pt"
GPT_CKPT="/cmlscratch/anirudhs/hub/t2i_XL_stage2_512.pt"
MAPPING_PATH="results/codebook_index_mapping.json"
PAIRS_PATH="results/codebook_pairs.json"
BASE_SAVE_DIR="/cmlscratch/anirudhs/graph_watermark/images/t2i_experiments"

# --- Common args ---
COMMON_ARGS="--vq-ckpt ${VQ_CKPT} \
             --num-assignments 10 \
             --gamma 0.5"

# --- Algorithm-specific args ---
EXTRA_ARGS=""
if [[ $ALGORITHM == "clustering" ]]; then
    EXTRA_ARGS="--cluster-size 512"
elif [[ $ALGORITHM == "spectral" ]]; then
    EXTRA_ARGS="--spectral-sigma 1.0"
elif [[ $ALGORITHM == "spectral-clustering" ]]; then
    EXTRA_ARGS="--spectral-families 512 --spectral-sigma 1.0"
fi

python generation.py \
    --gpt-ckpt "${GPT_CKPT}" \
    --image-size ${RESOLUTION} \
    --save-dir "${BASE_SAVE_DIR}" \
    --algorithm ${ALGORITHM} \
    --delta ${DELTA} \
    ${COMMON_ARGS} \
    ${EXTRA_ARGS}

echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
