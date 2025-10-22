import sys
import torch
import os
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gc
import json
import traceback
from PIL import Image
import torchvision.transforms as T
import numpy as np
from scipy import stats
import hashlib

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from tokenizer.tokenizer_image.vq_model import VQ_models
from generation import HierarchicalCodebook


def compute_green_tokens_from_context(context_tokens, vocab_size, gamma, seed):
    """
    Compute green token list based on context using KGW watermark scheme.

    Args:
        context_tokens: List of previous h tokens
        vocab_size: Size of the vocabulary
        gamma: Fraction of tokens to mark as green
        seed: Secret key for the hash function

    Returns:
        set: Set of green token indices for this context
    """
    if not context_tokens:
        return set()

    # Convert context tokens to string for hashing
    context_str = ",".join([str(t) for t in context_tokens])
    hash_input = f"{seed}_{context_str}".encode('utf-8')
    hash_obj = hashlib.sha256(hash_input)

    # Use hash to seed numpy random generator
    hash_int = int.from_bytes(hash_obj.digest()[:4], 'big')
    rng = np.random.RandomState(hash_int)

    # Number of green tokens
    num_green = int(gamma * vocab_size)

    # Generate green token list
    all_tokens = np.arange(vocab_size)
    green_tokens = rng.choice(all_tokens, size=num_green, replace=False)

    return set(green_tokens)


def compute_pvalue_kgw(indices, h, gamma, vocab_size, watermark_seed):
    """
    Compute p-value for KGW watermark detection.

    Args:
        indices: List/array of generated token indices
        h: Context window size
        gamma: Fraction of tokens marked as green
        vocab_size: Size of vocabulary
        watermark_seed: Secret key for watermark

    Returns:
        p_value: Statistical significance (lower = more likely watermarked)
        green_count: Number of green tokens detected
        total_checked: Total tokens checked (T - h)
    """
    T = len(indices)
    if T <= h:
        return 1.0, 0, 0

    green_count = 0

    # For each token from position h+1 onwards, check if it's in the green list
    for i in range(h, T):
        # Get context (previous h tokens)
        context = indices[i-h:i]

        # Compute green list for this context
        green_set = compute_green_tokens_from_context(context, vocab_size, gamma, watermark_seed)

        # Check if current token is green
        if indices[i] in green_set:
            green_count += 1

    # Total tokens checked
    total_checked = T - h

    # Compute p-value using binomial test
    # Under H0 (no watermark), S ~ Binomial(T-h, gamma)
    # p-value = P(X >= S | X ~ Binomial(T-h, gamma))
    p_value = stats.binom.sf(green_count - 1, total_checked, gamma)

    return p_value, green_count, total_checked


def load_image_to_tensor(image_path, target_device="cpu"):
    pil_image = Image.open(image_path).convert("RGB")
    img_np_uint8 = np.array(pil_image)
    img_np_float01 = img_np_uint8.astype(np.float32) / 255.0
    img_np_float_neg1_pos1 = 2.0 * img_np_float01 - 1.0
    tensor_hwc = torch.from_numpy(img_np_float_neg1_pos1)
    tensor_chw = tensor_hwc.permute(2, 0, 1)
    tensor_nchw = tensor_chw.unsqueeze(0)
    reconstructed_tensor = tensor_nchw.to(target_device).float()
    return reconstructed_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text-to-Image AR Generation with Pairing-based Codebook Replacement"
    )

    parser.add_argument(
        "--vq-ckpt",
        type=str,
        default="/cmlscratch/anirudhs/hub/vq_ds16_t2i.pt",
        help="Path to VQ model checkpoint",
    )
    parser.add_argument(
        "--mapping-save-path",
        type=str,
        default="results/codebook_index_mapping.json",
        help="Path to save/load the index mapping JSON file",
    )
    parser.add_argument(
        "--pairs-save-path",
        type=str,
        default="results/codebook_pairs.json",
        help="Path to save/load the codebook pair list JSON file",
    )
    parser.add_argument(
        "--vq-model",
        type=str,
        choices=list(VQ_models.keys()) if VQ_models else ["VQ-16"],
        default="VQ-16",
        help="VQ model architecture",
    )
    parser.add_argument(
        "--codebook-size", type=int, default=16384, help="Codebook size for VQ"
    )
    parser.add_argument(
        "--codebook-embed-dim",
        type=int,
        default=8,
        help="Codebook embedding dimension for VQ",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["none", "fp16", "bf16"],
        help="Computation precision (none means fp32)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--replacement-ratio",
        type=float,
        default=1.0,
        help="Ratio of red-listed tokens to be replaced (0.0-1.0)",
    )
    parser.add_argument(
        "--load-mapping",
        action="store_true",
        help="Attempt to load index mapping from file instead of recomputing",
    )
    parser.add_argument(
        "--load-pairs",
        action="store_true",
        help="Attempt to load codebook pair list from file instead of recomputing",
    )
    parser.add_argument(
        "--index-encoder",
        type=str,
        default="/cmlscratch/anirudhs/hub/Index_encoder_512.pt",
    )
    parser.add_argument(
        "--image-directory", type=str, default="images/Gen_Image/100%-512"
    )
    parser.add_argument(
        "--h",
        type=int,
        default=1,
        help="Context window size for KGW watermark detection.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Fraction of vocabulary marked as green tokens (e.g., 0.5 for 50%%).",
    )
    parser.add_argument(
        "--watermark-seed",
        type=int,
        default=42,
        help="Secret key (seed) for watermark hash function.",
    )
    parser.add_argument(
        "--use-pvalue",
        action="store_true",
        help="Use p-value based detection instead of simple green token ratio.",
    )
    parser.add_argument(
        "--pvalue-threshold",
        type=float,
        default=0.01,
        help="P-value threshold for watermark detection (lower = stricter).",
    )

    args, _ = parser.parse_known_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        print(
            "Warning: BF16 precision was requested, but is not supported on this GPU. Falling back to FP32."
        )
        args.precision = "none"

    hc = None
    codebook_weights = None

    try:
        print("Loading VQ model...")
        vq_model_class = VQ_models.get(args.vq_model)
        vq_model = vq_model_class(
            codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim
        ).to(device)

        should_load_vq_weights = not (
            args.load_pairs
            and args.load_mapping
            and os.path.exists(args.pairs_save_path)
            and os.path.exists(args.mapping_save_path)
        )

        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(
            checkpoint["model"] if "model" in checkpoint else checkpoint["ema"]
        )

        if args.index_encoder != None:
            finetuned_checkpoint = torch.load(
                args.index_encoder, map_location="cpu", weights_only=False
            )
            encoder_state_dict = finetuned_checkpoint["encoder_state_dict"]
            missing_keys, unexpected_keys = vq_model.encoder.load_state_dict(
                encoder_state_dict, strict=False
            )

        codebook_weights = (
            vq_model.quantize.embedding.weight.data.cpu().numpy().astype(np.float32)
        )
        print(f"Codebook weights extracted (shape: {codebook_weights.shape})")

        hc = HierarchicalCodebook(
            codebook_vectors=codebook_weights,
            replacement_ratio=args.replacement_ratio,
            mapping_save_path=args.mapping_save_path,
            pairs_save_path=args.pairs_save_path,
            load_mapping=True,
            load_pairs=True,
        )

    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing an expected key in the checkpoint file. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during initialization:")
        traceback.print_exc()
        sys.exit(1)

    if hc:
        print("\nHierarchicalCodebook ready.")
    else:
        print("\nError: HierarchicalCodebook failed to initialize.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_ratios = []
    all_pvalues = []
    processed_image_count = 0
    image_directory = args.image_directory
    all_files = [
        f
        for f in os.listdir(image_directory)
        if os.path.isfile(os.path.join(image_directory, f))
    ]
    image_files = [
        f
        for f in all_files
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]

    print(f"\nDetection Mode: {'P-value based' if args.use_pvalue else 'Green ratio based'}")
    if args.use_pvalue:
        print(f"P-value threshold: {args.pvalue_threshold}")
        print(f"KGW parameters: h={args.h}, gamma={args.gamma}, seed={args.watermark_seed}")

    for filename in image_files:
        image_path = os.path.join(image_directory, filename)
        x_input = load_image_to_tensor(image_path, target_device=device)
        with torch.no_grad():
            latent_orig, _, [_, _, indices_orig] = vq_model.encode(x_input)
            indices_orig = indices_orig.flatten()

            if args.use_pvalue:
                # Use p-value based detection
                indices_list = indices_orig.cpu().numpy().tolist()
                p_value, green_count, total_checked = compute_pvalue_kgw(
                    indices_list,
                    args.h,
                    args.gamma,
                    args.codebook_size,
                    args.watermark_seed
                )
                all_pvalues.append(p_value)
                current_ratio = green_count / total_checked if total_checked > 0 else 0
                all_ratios.append(current_ratio)
                processed_image_count += 1
            else:
                # Use simple green ratio detection
                green = 0
                red = 0
                total_indices = 0
                for i in indices_orig:
                    idx_item = i.item()
                    if idx_item in hc.green_list:
                        green += 1
                    else:
                        red += 1
                    total_indices += 1

                if total_indices > 0:
                    current_ratio = green / total_indices
                    all_ratios.append(current_ratio)
                    processed_image_count += 1
                else:
                    print(f"Warning: No indices found for image {filename}. Skipping.")

    if processed_image_count > 0:
        average_ratio = sum(all_ratios) / processed_image_count
        print("\n--- Results ---")
        print(f"Processed {processed_image_count} images.")
        print(f"Average 'green' ratio across all processed images: {average_ratio:.8f}")

        if args.use_pvalue:
            average_pvalue = sum(all_pvalues) / processed_image_count
            print(f"Average p-value across all processed images: {average_pvalue:.10f}")
            detected_watermarks = sum(1 for p in all_pvalues if p < args.pvalue_threshold)
            print(f"Images with watermark detected (p < {args.pvalue_threshold}): {detected_watermarks}/{processed_image_count}")
    else:
        print("\nNo images were successfully processed.")
