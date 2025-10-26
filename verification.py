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


def load_assignment_json(image_path):
    """
    Load the assignment JSON file for a given image.

    Args:
        image_path: Path to the image file

    Returns:
        dict: Assignment data including algorithm, red_list, green_list, etc.
              Returns None if file not found.
    """
    # Get the directory and filename
    image_dir = os.path.dirname(image_path)
    image_filename = os.path.basename(image_path)
    image_name = os.path.splitext(image_filename)[0]

    # Look for the JSON file in assignments subdirectory
    json_path = os.path.join(image_dir, "assignments", f"{image_name}.json")

    if not os.path.exists(json_path):
        print(f"Warning: Assignment JSON not found at {json_path}")
        return None

    try:
        with open(json_path, "r") as f:
            assignment_data = json.load(f)
        return assignment_data
    except Exception as e:
        print(f"Error loading assignment JSON from {json_path}: {e}")
        return None


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
    hash_input = f"{seed}_{context_str}".encode("utf-8")
    hash_obj = hashlib.sha256(hash_input)

    # Use hash to seed numpy random generator
    hash_int = int.from_bytes(hash_obj.digest()[:4], "big")
    rng = np.random.RandomState(hash_int)

    # Number of green tokens
    num_green = int(gamma * vocab_size)

    # Generate green token list
    all_tokens = np.arange(vocab_size)
    green_tokens = rng.choice(all_tokens, size=num_green, replace=False)

    return set(green_tokens)


def compute_pvalue_kgw(indices, green_list, h, gamma):
    """
    Compute p-value for KGW watermark detection using pre-defined green list.

    Args:
        indices: List/array of generated token indices
        green_list: Set or list of green token indices
        h: Context window size (for current experiments, h=0)
        gamma: Fraction of tokens marked as green

    Returns:
        p_value: Statistical significance (lower = more likely watermarked)
        green_count: Number of green tokens detected
        total_checked: Total tokens checked (T - h)
    """
    T = len(indices)
    if T <= h:
        return 1.0, 0, 0

    green_set = set(green_list) if not isinstance(green_list, set) else green_list
    green_count = 0

    # For each token from position h onwards, check if it's in the green list
    # Note: h=0 for current experiments, so we check all tokens
    for i in range(h, T):
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
        "--image-directory",
        type=str,
        default="/cmlscratch/anirudhs/graph_watermark/images/t2i_experiments/spectral-clustering-delta2.0-512",
    )
    parser.add_argument(
        "--h",
        type=int,
        default=0,
        help="Context window size for KGW watermark detection (default: 0).",
    )
    parser.add_argument(
        "--pvalue-threshold",
        type=float,
        default=0.01,
        help="P-value threshold for watermark detection (lower = stricter).",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="spectral-clustering",
        help="Override algorithm type (otherwise read from JSON). Options: pairwise, random, clustering, spectral, spectral-clustering",
    )
    parser.add_argument(
        "--results-output",
        type=str,
        default="verification_results.json",
        help="Path to save verification results as JSON",
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
    pairwise_count = 0
    llm_watermark_count = 0

    # Store detailed results for each image
    image_results = []

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

    print(f"\nProcessing {len(image_files)} images from {image_directory}")
    print(f"P-value threshold: {args.pvalue_threshold}")
    print(f"Context window size (h): {args.h}")

    for filename in image_files:
        image_path = os.path.join(image_directory, filename)

        # Load assignment JSON to determine algorithm
        assignment_data = load_assignment_json(image_path)

        if assignment_data is None:
            print(f"Warning: No assignment data for {filename}. Skipping.")
            continue

        # Determine algorithm
        algorithm = (
            args.algorithm
            if args.algorithm
            else assignment_data.get("algorithm", "unknown")
        )

        # Extract indices from image
        x_input = load_image_to_tensor(image_path, target_device=device)
        with torch.no_grad():
            latent_orig, _, [_, _, indices_orig] = vq_model.encode(x_input)
            indices_orig = indices_orig.flatten()
            indices_list = indices_orig.cpu().numpy().tolist()

            if algorithm == "pairwise":
                # Pairwise: Use pairwise replacement then calculate p-value
                pairwise_count += 1
                green_list_data = assignment_data.get("green_list", [])
                red_list_data = assignment_data.get("red_list", [])
                pairs = assignment_data.get("pairs", [])
                replacement_ratio = assignment_data.get("parameters", {}).get(
                    "replacement_ratio", 1.0
                )

                green_list = set(green_list_data)
                red_list = set(red_list_data)

                # Create index mapping from pairs (red -> green)
                index_mapping = {}
                for pair in pairs:
                    if len(pair) == 2:
                        idx1, idx2 = pair
                        # Determine which is red and which is green
                        if idx1 in red_list and idx2 in green_list:
                            index_mapping[idx1] = idx2
                        elif idx2 in red_list and idx1 in green_list:
                            index_mapping[idx2] = idx1

                # Apply pairwise replacement probabilistically
                replaced_indices = []
                np.random.seed(args.seed)  # For reproducibility
                for idx in indices_list:
                    if idx in red_list and idx in index_mapping:
                        # Probabilistically replace red token with green pair
                        if np.random.random() < replacement_ratio:
                            replaced_indices.append(index_mapping[idx])
                        else:
                            replaced_indices.append(idx)
                    else:
                        replaced_indices.append(idx)

                # Calculate gamma from actual green/red lists
                total_vocab = len(green_list_data) + len(red_list_data)
                gamma = len(green_list_data) / total_vocab if total_vocab > 0 else 0.5

                # Compute p-value using the replaced indices
                p_value, green_count, total_checked = compute_pvalue_kgw(
                    replaced_indices, green_list, args.h, gamma
                )
                all_pvalues.append(p_value)
                current_ratio = green_count / total_checked if total_checked > 0 else 0
                all_ratios.append(current_ratio)
                processed_image_count += 1

                # Store image result
                image_results.append(
                    {
                        "filename": filename,
                        "algorithm": "pairwise",
                        "p_value": float(p_value),
                        "green_ratio": float(current_ratio),
                        "gamma": float(gamma),
                        "green_count": int(green_count),
                        "total_checked": int(total_checked),
                        "num_pairs": len(pairs),
                        "detected": (p_value < args.pvalue_threshold).item(),
                    }
                )

                print(
                    f"  {filename} [pairwise]: p-value={p_value:.6e}, green_ratio={current_ratio:.4f}, gamma={gamma:.4f}, pairs={len(pairs)}"
                )

            elif algorithm in [
                "random",
                "clustering",
                "spectral",
                "spectral-clustering",
            ]:
                # LLM watermarking: Use KGW p-value based detection with binomial test
                llm_watermark_count += 1
                green_list_data = assignment_data.get("green_list", [])
                red_list_data = assignment_data.get("red_list", [])

                # Calculate gamma from actual green/red lists
                total_vocab = len(green_list_data) + len(red_list_data)
                gamma = len(green_list_data) / total_vocab if total_vocab > 0 else 0.5

                green_list = set(green_list_data)

                # Compute p-value using the green list from JSON
                p_value, green_count, total_checked = compute_pvalue_kgw(
                    indices_list, green_list, args.h, gamma
                )
                all_pvalues.append(p_value)
                current_ratio = green_count / total_checked if total_checked > 0 else 0
                all_ratios.append(current_ratio)
                processed_image_count += 1

                # Store image result
                image_results.append(
                    {
                        "filename": filename,
                        "algorithm": algorithm,
                        "p_value": float(p_value),
                        "green_ratio": float(current_ratio),
                        "gamma": float(gamma),
                        "green_count": int(green_count),
                        "total_checked": int(total_checked),
                        "detected": (p_value < args.pvalue_threshold).item(),
                    }
                )

                print(
                    f"  {filename} [{algorithm}]: p-value={p_value:.6e}, green_ratio={current_ratio:.4f}, gamma={gamma:.4f}"
                )
            else:
                print(
                    f"Warning: Unknown algorithm '{algorithm}' for {filename}. Skipping."
                )
                continue

    # Print summary results
    if processed_image_count > 0:
        average_ratio = sum(all_ratios) / processed_image_count
        average_pvalue_neg_log = sum(-np.log(all_pvalues)) / processed_image_count
        detected_watermarks = sum(1 for p in all_pvalues if p < args.pvalue_threshold)

        print("\n" + "=" * 80)
        print("VERIFICATION RESULTS")
        print("=" * 80)
        print(f"Total images processed: {processed_image_count}")
        print(f"  - Pairwise algorithm: {pairwise_count}")
        print(f"  - LLM watermarking algorithms: {llm_watermark_count}")
        print(f"\nAverage green token ratio: {average_ratio:.8f}")
        print(f"Average p-value (-log): {float(average_pvalue_neg_log):.10f}")
        print(
            f"Images with watermark detected (p < {args.pvalue_threshold}): {detected_watermarks}/{processed_image_count}"
        )
        print(f"Detection rate: {detected_watermarks/processed_image_count*100:.2f}%")

        print("=" * 80)

        # Prepare results dictionary
        results = {
            "parameters": {
                "image_directory": args.image_directory,
                "pvalue_threshold": args.pvalue_threshold,
                "context_window_h": args.h,
                "seed": args.seed,
                "replacement_ratio": args.replacement_ratio,
                "algorithm": args.algorithm,
            },
            "summary": {
                "total_images_processed": processed_image_count,
                "pairwise_count": pairwise_count,
                "llm_watermark_count": llm_watermark_count,
                "average_green_ratio": float(average_ratio),
                "average_pvalue": float(average_pvalue_neg_log),
                "detected_watermarks": detected_watermarks,
                "detection_rate": float(detected_watermarks / processed_image_count),
            },
            "individual_results": image_results,
        }

        # Save results to JSON file
        with open(args.results_output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {args.results_output}")

    else:
        print("\nNo images were successfully processed.")

        # Save empty results
        results = {
            "parameters": {
                "image_directory": args.image_directory,
                "pvalue_threshold": args.pvalue_threshold,
                "context_window_h": args.h,
                "seed": args.seed,
                "replacement_ratio": args.replacement_ratio,
                "algorithm": args.algorithm,
            },
            "summary": {
                "total_images_processed": 0,
                "pairwise_count": 0,
                "llm_watermark_count": 0,
            },
            "individual_results": [],
        }

        with open(args.results_output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Empty results saved to: {args.results_output}")
