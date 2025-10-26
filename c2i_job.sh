#!/bin/bash
#SBATCH --job-name=c2i_gen
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --time=3-00:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=4

# ==============================================
# c2i_job.sbatch
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
VQ_CKPT="/cmlscratch/anirudhs/hub/vq_ds16_c2i.pt"
GPT_CKPT="/cmlscratch/anirudhs/hub/c2i_L_384.pt"
MAPPING_PATH="results/codebook_index_mapping_knn10_mwpm_c2i.json"
PAIRS_PATH="results/codebook_pairs_knn10_mwpm_c2i.json"
BASE_SAVE_DIR="/cmlscratch/anirudhs/graph_watermark/images/c2i_experiments"

# --- Common args ---
COMMON_ARGS="--vq-ckpt ${VQ_CKPT} \
             --num-assignments 10 \
             --num-classes 1000 \
             --num-seeds-per-class 1 \
             --gamma 0.5
             --overwrite"

# --- Algorithm-specific args ---
EXTRA_ARGS=""
if [[ $ALGORITHM == "clustering" ]]; then
    EXTRA_ARGS="--cluster-size 512"
elif [[ $ALGORITHM == "spectral" ]]; then
    EXTRA_ARGS="--spectral-sigma 1.0"
elif [[ $ALGORITHM == "spectral-clustering" ]]; then
    EXTRA_ARGS="--spectral-families 512 --spectral-sigma 1.0"
fi

python generation_c2i.py \
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
