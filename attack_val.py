import argparse
import gc
import io
import json
import os
import random
import sys
import time
import traceback
from typing import Union, Tuple, List, Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.utils import save_image
import torchvision.transforms as transforms
from scipy import stats
import hashlib

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
# sys.path.append(current_dir)
sys.path.append("/root/autodl-tmp/LlamaGen")

from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generatearcon import generate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def image_distortion(
    img_pil,
    seed,
    attack,
    jpeg_quality=70,
    blur_kernel_size=11,
    noise_std_fraction=0.01,
    color_jitter_brightness=0.5,
    crop_scale=0.75,
):
    img_distorted = img_pil.copy()

    if attack == "jpeg":
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        img_distorted = Image.open(buffer)
        img_distorted.load()

    elif attack == "cropping":
        set_random_seed(seed)
        crop_transform = transforms.RandomResizedCrop(
            img_distorted.size, scale=(crop_scale, crop_scale), ratio=(1.0, 1.0)
        )
        img_distorted = crop_transform(img_distorted)

    elif attack == "blurring":
        # Ensure kernel size is odd
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        blur_transform = transforms.GaussianBlur(
            kernel_size=blur_kernel_size, sigma=(blur_kernel_size - 1) / 6
        )
        img_distorted = blur_transform(img_distorted)

    elif attack == "noise":
        img_np = np.array(img_distorted).astype(np.float32)
        set_random_seed(seed)
        g_noise = np.random.normal(0, noise_std_fraction * 255.0, img_np.shape)
        noisy_img_np = img_np + g_noise
        noisy_img_np = np.clip(noisy_img_np, 0, 255)
        img_distorted = Image.fromarray(noisy_img_np.astype(np.uint8))

    elif attack == "color_jitter":
        set_random_seed(seed)
        jitter_transform = transforms.ColorJitter(brightness=color_jitter_brightness)
        img_distorted = jitter_transform(img_distorted)

    elif attack == "random_erase":
        erase_area_ratio = 0.1
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()
        tensor = to_tensor(img_distorted).unsqueeze(0)  # Add batch dim -> [1, C, H, W]
        n, c, h, w = tensor.shape
        area = h * w
        target_area = area * erase_area_ratio
        w_erase = int(np.sqrt(target_area))
        h_erase = int(np.sqrt(target_area))
        if w_erase >= w:
            w_erase = w
        if h_erase >= h:
            h_erase = h
        if w_erase == 0 or h_erase == 0:
            pass
        else:
            x = np.random.randint(0, w - w_erase + 1)
            y = np.random.randint(0, h - h_erase + 1)

            erase_val = torch.rand(n, c, h_erase, w_erase, device=tensor.device)

            tensor[..., y : y + h_erase, x : x + w_erase] = erase_val

        img_distorted = to_pil(tensor.squeeze(0))
    elif attack is None or attack.lower() == "none":
        pass
    else:
        print(
            f"Warning: Unknown attack type '{attack}'. Returning original image copy."
        )

    return img_distorted


def load_pil_image_to_tensor(pil_image, target_device="cpu"):
    img_np_uint8 = np.array(pil_image)
    if img_np_uint8.ndim == 2:
        img_np_uint8 = np.stack([img_np_uint8] * 3, axis=-1)
    elif img_np_uint8.shape[2] == 4:
        img_np_uint8 = img_np_uint8[:, :, :3]
    elif img_np_uint8.shape[2] != 3:
        raise ValueError(f"Unexpected number of channels: {img_np_uint8.shape[2]}")
    img_np_float01 = img_np_uint8.astype(np.float32) / 255.0
    img_np_float_neg1_pos1 = 2.0 * img_np_float01 - 1.0
    tensor_hwc = torch.from_numpy(img_np_float_neg1_pos1)
    tensor_chw = tensor_hwc.permute(2, 0, 1)
    tensor_nchw = tensor_chw.unsqueeze(0)
    reconstructed_tensor = tensor_nchw.to(target_device).float()
    return reconstructed_tensor


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


def tensor_to_image(tensor):
    tensor = torch.clamp(127.5 * tensor + 128.0, 0, 255)
    array = tensor.permute(0, 2, 3, 1).byte().cpu().numpy()
    return array


def get_inner_bounding_box_indices(
    start_row: int,
    start_col: int,
    h_placed: int,
    w_placed: int,
    image_size: int,
    num_grids_per_dim: int,
) -> List[int]:
    if h_placed <= 0 or w_placed <= 0:
        return []

    if image_size <= 0 or num_grids_per_dim <= 0:
        raise ValueError("image_size and num_grids_per_dim must be positive.")
    if image_size % num_grids_per_dim != 0:
        raise ValueError(
            f"image_size ({image_size}) must be divisible by num_grids_per_dim ({num_grids_per_dim})"
        )

    grid_size = image_size // num_grids_per_dim

    end_row = min(start_row + h_placed - 1, image_size - 1)
    end_col = min(start_col + w_placed - 1, image_size - 1)

    clamped_start_row = max(0, start_row)
    clamped_start_col = max(0, start_col)

    min_grid_row = clamped_start_row // grid_size
    min_grid_col = clamped_start_col // grid_size
    max_grid_row = end_row // grid_size
    max_grid_col = end_col // grid_size

    min_grid_row = max(0, min(min_grid_row, num_grids_per_dim - 1))
    min_grid_col = max(0, min(min_grid_col, num_grids_per_dim - 1))
    max_grid_row = max(0, min(max_grid_row, num_grids_per_dim - 1))
    max_grid_col = max(0, min(max_grid_col, num_grids_per_dim - 1))

    inner_min_grid_row = min_grid_row + 1
    inner_max_grid_row = max_grid_row - 1
    inner_min_grid_col = min_grid_col + 1
    inner_max_grid_col = max_grid_col - 1

    inner_box_indices = []
    if (
        inner_min_grid_row <= inner_max_grid_row
        and inner_min_grid_col <= inner_max_grid_col
    ):
        for r in range(inner_min_grid_row, inner_max_grid_row + 1):
            for c in range(inner_min_grid_col, inner_max_grid_col + 1):
                linear_index = r * num_grids_per_dim + c
                inner_box_indices.append(linear_index)

    return inner_box_indices


def place_crop_and_get_indices(
    image_size: int,
    crop_img: torch.Tensor,
    device: torch.device,
    max_offset: int = 16,
    num_grids_per_dim: int = 16,
    fill_value: float = 0.5,
) -> Tuple[List[torch.Tensor], List[List[int]]]:
    if not isinstance(crop_img, torch.Tensor):
        raise TypeError("crop_img 必须是 torch.Tensor")

    if crop_img.dim() == 4:
        if crop_img.shape[0] == 1:
            crop_img = crop_img.squeeze(0)
        else:
            raise ValueError(
                f"4D crop_img batch dim must be 1, got shape {crop_img.shape}"
            )
    elif crop_img.dim() != 3:
        raise ValueError(
            f"crop_img must be 3D (C, H, W) or 4D (1, C, H, W), got shape {crop_img.shape}"
        )

    if max_offset <= 0:
        raise ValueError("max_offset must be positive.")
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if num_grids_per_dim <= 0:
        raise ValueError("num_grids_per_dim must be positive.")
    if image_size % num_grids_per_dim != 0:
        raise ValueError(
            f"image_size ({image_size}) must be divisible by num_grids_per_dim ({num_grids_per_dim})"
        )

    crop_img = crop_img.to(device)
    num_channels, h_crop, w_crop = crop_img.shape

    augmented_image_list: List[torch.Tensor] = []
    inner_indices_list: List[List[int]] = []
    total_images = max_offset * max_offset

    for r_offset in range(max_offset):
        for c_offset in range(max_offset):
            start_row = r_offset
            start_col = c_offset

            canvas = torch.full(
                (num_channels, image_size, image_size),
                fill_value,
                dtype=crop_img.dtype,
                device=device,
            )

            end_row_on_canvas = min(start_row + h_crop, image_size)
            end_col_on_canvas = min(start_col + w_crop, image_size)
            h_to_place = max(0, end_row_on_canvas - start_row)
            w_to_place = max(0, end_col_on_canvas - start_col)

            if h_to_place > 0 and w_to_place > 0:
                crop_data_to_place = crop_img[:, :h_to_place, :w_to_place]
                canvas[:, start_row:end_row_on_canvas, start_col:end_col_on_canvas] = (
                    crop_data_to_place
                )

            current_inner_indices = get_inner_bounding_box_indices(
                start_row=start_row,
                start_col=start_col,
                h_placed=h_to_place,
                w_placed=w_to_place,
                image_size=image_size,
                num_grids_per_dim=num_grids_per_dim,
            )

            augmented_image_list.append(canvas.unsqueeze(0))
            inner_indices_list.append(current_inner_indices)

    assert (
        len(augmented_image_list) == total_images
    ), f"Generated image count ({len(augmented_image_list)}) != expected ({total_images})"
    assert (
        len(inner_indices_list) == total_images
    ), f"Generated index list count ({len(inner_indices_list)}) != expected ({total_images})"

    return augmented_image_list, inner_indices_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Validation")

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
    parser.add_argument("--index-encoder", type=str, default=None)
    parser.add_argument(
        "--ft-pt-path",
        type=str,
        default="/cmlscratch/anirudhs/hub/Index_encoder_512.pt",
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
        default="pairwise",
        help="Override algorithm type (otherwise read from JSON). Options: pairwise, random, clustering, spectral, spectral-clustering",
    )
    parser.add_argument("--WATERMARK-THRESHOLD", type=float, default=0.615)
    parser.add_argument("--target-image-size", type=int, default=512)
    parser.add_argument(
        "--chosen-attack",
        type=str,
        default="blurring",
        choices=["jpeg", "cropping", "blurring", "noise", "color_jitter", "none"],
    )
    parser.add_argument("--jpeg-attack-quality", type=int, default=70)
    parser.add_argument(
        "--blur-kernel-size",
        type=int,
        default=19,
        help="Kernel size for Gaussian blur attack (must be odd)",
    )
    parser.add_argument(
        "--noise-std-fraction",
        type=float,
        default=0.01,
        help="Standard deviation fraction for Gaussian noise attack (multiplied by 255)",
    )
    parser.add_argument(
        "--color-jitter-brightness",
        type=float,
        default=0.5,
        help="Brightness parameter for color jitter attack",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=0.75,
        help="Scale for random crop attack (0.0-1.0)",
    )
    parser.add_argument(
        "--Watermarked-dir",
        type=str,
        default="/cmlscratch/anirudhs/graph_watermark/images/t2i_experiments/random-delta2.0-512",
    )
    parser.add_argument(
        "--Not-Watermarked-dir",
        type=str,
        default="/cmlscratch/anirudhs/graph_watermark/images/t2i_experiments/random-baseline-512",
    )
    parser.add_argument("--distortion-seed", type=int, default=123)
    parser.add_argument(
        "--results-output",
        type=str,
        default="attack_validation_results.json",
        help="Path to save attack validation results as JSON",
    )
    parser.add_argument(
        "--fpr-threshold",
        type=float,
        default=1.0,
        help="FPR threshold (in percentage) for computing TPR @ FPR metric (default: 1.0 for 1%%)",
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
    codebook_weights = None

    print("Loading VQ model...")
    vq_model_class = VQ_models.get(args.vq_model)
    if not vq_model_class:
        raise ValueError(f"Unsupported VQ model type: {args.vq_model}")

    vq_model = vq_model_class(
        codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim
    )

    should_load_vq_weights = not (
        args.load_pairs
        and args.load_mapping
        and os.path.exists(args.pairs_save_path)
        and os.path.exists(args.mapping_save_path)
    )

    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("ema", checkpoint))
    if state_dict is None:
        raise KeyError("Could not find 'model' or 'ema' key in the VQ checkpoint.")
    vq_model.load_state_dict(state_dict)
    del checkpoint, state_dict
    gc.collect()
    print("VQ model weights loaded.")

    if args.ft_pt_path != None:
        finetuned_checkpoint = torch.load(
            args.ft_pt_path, map_location="cpu", weights_only=False
        )
        if "encoder_state_dict" in finetuned_checkpoint:
            encoder_state_dict = finetuned_checkpoint["encoder_state_dict"]
            missing_keys, unexpected_keys = vq_model.encoder.load_state_dict(
                encoder_state_dict, strict=False
            )
            print("Loaded finetuned encoder state dict.")
            if missing_keys:
                print(f"  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys}")
        else:
            print(f"Warning: 'encoder_state_dict' not found in {args.ft_pt_path}.")
    else:
        print(
            f"Warning: Finetuned encoder checkpoint not found at {args.ft_pt_path}. Using encoder from VQ checkpoint."
        )

    vq_model.to(device)
    vq_model.eval()
    codebook_weights = (
        vq_model.quantize.embedding.weight.data.cpu().numpy().astype(np.float32)
    )
    print(f"Codebook weights extracted (shape: {codebook_weights.shape})")
    #### Begin validation

    directories_to_process = {
        args.Watermarked_dir: 1,  # Watermarked = Label 1
        args.Not_Watermarked_dir: 0,  # Not Watermarked = Label 0
    }

    all_true_labels = []
    all_predictions = []
    all_calculated_ratios = []
    processed_image_count = 0
    total_images_to_process = 0

    # Store detailed results for each image
    image_results = []

    for image_directory in directories_to_process:
        if not os.path.isdir(image_directory):
            print(f"Warning: Directory not found - {image_directory}")
            continue
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
        total_images_to_process += len(image_files)

    print(
        f"\nStarting processing for attack type: '{args.chosen_attack}' across {len(directories_to_process)} directories ({total_images_to_process} images expected)..."
    )

    for image_directory, true_label in directories_to_process.items():
        print(f"\nProcessing directory: {image_directory} (Label: {true_label})")
        if not os.path.isdir(image_directory):
            print("  Directory skipped (not found).")
            continue

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
        print(f"  Found {len(image_files)} images.")

        for idx, filename in tqdm(enumerate(image_files)):
            image_path = os.path.join(image_directory, filename)
            current_seed = args.distortion_seed + idx

            try:
                # Load assignment JSON to determine algorithm and green/red lists
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

                original_pil_image = Image.open(image_path).convert("RGB")

                # Get green/red lists from assignment data
                green_list_data = assignment_data.get("green_list", [])
                red_list_data = assignment_data.get("red_list", [])

                # Calculate gamma from actual green/red lists
                total_vocab = len(green_list_data) + len(red_list_data)
                gamma = len(green_list_data) / total_vocab if total_vocab > 0 else 0.5
                green_list = set(green_list_data)

                # Apply attack/distortion
                distorted_pil_image = image_distortion(
                    original_pil_image,
                    seed=current_seed,
                    attack=args.chosen_attack,
                    jpeg_quality=args.jpeg_attack_quality,
                    blur_kernel_size=args.blur_kernel_size,
                    noise_std_fraction=args.noise_std_fraction,
                    color_jitter_brightness=args.color_jitter_brightness,
                    crop_scale=args.crop_scale,
                )

                x_input = load_pil_image_to_tensor(
                    distorted_pil_image, target_device=device
                )

                with torch.no_grad():
                    # Extract indices from distorted image
                    latent_orig, _, misc = vq_model.encode(x_input)
                    indices_orig = misc[2]
                    indices_orig = indices_orig.flatten()
                    indices_list = indices_orig.cpu().numpy().tolist()

                    p_value, green_count, total_checked = compute_pvalue_kgw(
                        indices_list, green_list, args.h, gamma
                    )

                    current_ratio = (
                        green_count / total_checked if total_checked > 0 else 0
                    )
                    all_calculated_ratios.append(current_ratio)

                    # Use p-value threshold for prediction
                    prediction = 1 if p_value < args.pvalue_threshold else 0

                    all_true_labels.append(true_label)
                    all_predictions.append(prediction)

                    # Store detailed results
                    result_dict = {
                        "filename": filename,
                        "directory": image_directory,
                        "true_label": int(true_label),
                        "predicted_label": int(prediction),
                        "p_value": float(p_value),
                        "green_ratio": float(current_ratio),
                        "gamma": float(gamma),
                        "green_count": int(green_count),
                        "total_checked": int(total_checked),
                        "algorithm": algorithm,
                        "attack_type": args.chosen_attack,
                        "correct": bool(true_label == prediction),
                        "watermarked": bool(true_label == 1),
                        "detected_as_watermarked": bool(prediction == 1),
                    }

                    # Add attack-specific parameters
                    if args.chosen_attack == "jpeg":
                        result_dict["jpeg_quality"] = args.jpeg_attack_quality
                    elif args.chosen_attack == "blurring":
                        result_dict["blur_kernel_size"] = args.blur_kernel_size
                    elif args.chosen_attack == "noise":
                        result_dict["noise_std_fraction"] = args.noise_std_fraction
                    elif args.chosen_attack == "color_jitter":
                        result_dict["color_jitter_brightness"] = (
                            args.color_jitter_brightness
                        )
                    elif args.chosen_attack == "cropping":
                        result_dict["crop_scale"] = args.crop_scale

                    image_results.append(result_dict)

                    processed_image_count += 1

                    if processed_image_count % 100 == 0:
                        print(
                            f"  Processed {processed_image_count}/{total_images_to_process} images..."
                        )

                # Clean up memory
                if args.chosen_attack != "cropping":
                    del (
                        x_input,
                        latent_orig,
                        indices_orig,
                        misc,
                        original_pil_image,
                        distorted_pil_image,
                    )
                if device == "cuda":
                    torch.cuda.empty_cache()
            except FileNotFoundError:
                print(f"Error: Image file not found at {image_path}. Skipping.")
            except Exception as e:
                print(
                    f"Error processing image {filename} with attack '{args.chosen_attack}': {e}"
                )
                traceback.print_exc()
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    print(f"\n--- Overall Results ---")
    print(
        f"Attack Type Applied: {args.chosen_attack}"
        + (
            f" (JPEG Quality: {args.jpeg_attack_quality})"
            if args.chosen_attack == "jpeg"
            else ""
        )
    )
    print(f"Watermark Detection: p-value < {args.pvalue_threshold}")
    print(f"Context window size (h): {args.h}")
    print(
        f"Total images processed: {processed_image_count} (Expected: {total_images_to_process})"
    )

    if processed_image_count > 0 and len(all_true_labels) == len(all_predictions):
        accuracy = accuracy_score(all_true_labels, all_predictions)

        correct_predictions = sum(
            1 for true, pred in zip(all_true_labels, all_predictions) if true == pred
        )
        manual_accuracy = correct_predictions / processed_image_count

        print(
            f"\nOverall Accuracy: {accuracy:.6f} ({correct_predictions}/{processed_image_count})"
        )

        watermarked_indices = [
            i for i, label in enumerate(all_true_labels) if label == 1
        ]
        non_watermarked_indices = [
            i for i, label in enumerate(all_true_labels) if label == 0
        ]

        if watermarked_indices:
            watermarked_correct = sum(
                1 for i in watermarked_indices if all_predictions[i] == 1
            )
            watermarked_acc = watermarked_correct / len(watermarked_indices)
            print(
                f"  Accuracy on Watermarked (Label 1): {watermarked_acc:.4f} ({watermarked_correct}/{len(watermarked_indices)})"
            )

        if non_watermarked_indices:
            non_watermarked_correct = sum(
                1 for i in non_watermarked_indices if all_predictions[i] == 0
            )
            non_watermarked_acc = non_watermarked_correct / len(non_watermarked_indices)
            print(
                f"  Accuracy on Non-Watermarked (Label 0): {non_watermarked_acc:.4f} ({non_watermarked_correct}/{len(non_watermarked_indices)})"
            )

        # Calculate TPR @ FPR metric
        tpr_at_fpr = None
        if len(image_results) > 0:
            # Extract p-values and true labels from image_results
            all_pvalues = [result["p_value"] for result in image_results]
            y_true = [result["true_label"] for result in image_results]

            # Use negative p-value as score (lower p-value = higher watermark confidence)
            y_score = [-pval for pval in all_pvalues]

            # Calculate ROC curve
            fpr_array, tpr_array, thresholds = roc_curve(y_true, y_score, pos_label=1)

            # Find TPR at the specified FPR threshold (convert percentage to fraction)
            target_fpr = args.fpr_threshold / 100.0

            # Find the closest FPR value to the target
            idx = np.argmin(np.abs(fpr_array - target_fpr))
            tpr_at_fpr = float(tpr_array[idx])
            actual_fpr = float(fpr_array[idx])

            print(
                f"\nTPR @ FPR={args.fpr_threshold}%: {tpr_at_fpr:.4f} (actual FPR: {actual_fpr*100:.2f}%)"
            )

        # Prepare results dictionary
        attack_params = {}
        if args.chosen_attack == "jpeg":
            attack_params["jpeg_quality"] = args.jpeg_attack_quality
        elif args.chosen_attack == "blurring":
            attack_params["blur_kernel_size"] = args.blur_kernel_size
        elif args.chosen_attack == "noise":
            attack_params["noise_std_fraction"] = args.noise_std_fraction
        elif args.chosen_attack == "color_jitter":
            attack_params["color_jitter_brightness"] = args.color_jitter_brightness
        elif args.chosen_attack == "cropping":
            attack_params["crop_scale"] = args.crop_scale

        results = {
            "parameters": {
                "watermarked_dir": args.Watermarked_dir,
                "not_watermarked_dir": args.Not_Watermarked_dir,
                "pvalue_threshold": args.pvalue_threshold,
                "context_window_h": args.h,
                "attack_type": args.chosen_attack,
                "attack_params": attack_params,
                "distortion_seed": args.distortion_seed,
                "seed": args.seed,
                "replacement_ratio": args.replacement_ratio,
                "algorithm": args.algorithm,
                "fpr_threshold_percent": args.fpr_threshold,
            },
            "summary": {
                "total_images_processed": processed_image_count,
                "overall_accuracy": float(accuracy),
                "correct_predictions": correct_predictions,
                "average_green_ratio": (
                    float(sum(all_calculated_ratios) / len(all_calculated_ratios))
                    if all_calculated_ratios
                    else 0.0
                ),
                "tpr_at_fpr": tpr_at_fpr,
                "fpr_threshold_percent": args.fpr_threshold,
                "watermarked_stats": (
                    {
                        "total": len(watermarked_indices),
                        "correct": watermarked_correct if watermarked_indices else 0,
                        "accuracy": (
                            float(watermarked_acc) if watermarked_indices else 0.0
                        ),
                    }
                    if watermarked_indices
                    else None
                ),
                "non_watermarked_stats": (
                    {
                        "total": len(non_watermarked_indices),
                        "correct": (
                            non_watermarked_correct if non_watermarked_indices else 0
                        ),
                        "accuracy": (
                            float(non_watermarked_acc)
                            if non_watermarked_indices
                            else 0.0
                        ),
                    }
                    if non_watermarked_indices
                    else None
                ),
            },
            "individual_results": image_results,
        }

        # Save results to JSON file
        with open(args.results_output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {args.results_output}")

    else:
        print(
            "\nCould not calculate accuracy. Not enough data processed or mismatch in label/prediction counts."
        )

        # Save empty results
        attack_params = {}
        if args.chosen_attack == "jpeg":
            attack_params["jpeg_quality"] = args.jpeg_attack_quality
        elif args.chosen_attack == "blurring":
            attack_params["blur_kernel_size"] = args.blur_kernel_size
        elif args.chosen_attack == "noise":
            attack_params["noise_std_fraction"] = args.noise_std_fraction
        elif args.chosen_attack == "color_jitter":
            attack_params["color_jitter_brightness"] = args.color_jitter_brightness
        elif args.chosen_attack == "cropping":
            attack_params["crop_scale"] = args.crop_scale

        results = {
            "parameters": {
                "watermarked_dir": args.Watermarked_dir,
                "not_watermarked_dir": args.Not_Watermarked_dir,
                "pvalue_threshold": args.pvalue_threshold,
                "context_window_h": args.h,
                "attack_type": args.chosen_attack,
                "attack_params": attack_params,
                "distortion_seed": args.distortion_seed,
                "seed": args.seed,
                "replacement_ratio": args.replacement_ratio,
                "algorithm": args.algorithm,
                "fpr_threshold_percent": args.fpr_threshold,
            },
            "summary": {
                "total_images_processed": 0,
                "overall_accuracy": 0.0,
                "tpr_at_fpr": None,
                "fpr_threshold_percent": args.fpr_threshold,
            },
            "individual_results": [],
        }

        with open(args.results_output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Empty results saved to: {args.results_output}")

    print("\nScript finished.")
