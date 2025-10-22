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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.utils import save_image
import torchvision.transforms as transforms

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
# sys.path.append(current_dir)
sys.path.append("/root/autodl-tmp/LlamaGen")

from hc import HierarchicalCodebook
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


def image_distortion(img_pil, seed, attack, jpeg_quality=70):
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
            img_distorted.size, scale=(0.75, 0.75), ratio=(1.0, 1.0)
        )
        img_distorted = crop_transform(img_distorted)

    elif attack == "blurring":
        blur_transform = transforms.GaussianBlur(kernel_size=11, sigma=1.0)
        img_distorted = blur_transform(img_distorted)

    elif attack == "noise":
        img_np = np.array(img_distorted).astype(np.float32)
        set_random_seed(seed)
        noise_std_dev_fraction = 0.01
        g_noise = np.random.normal(0, noise_std_dev_fraction * 255.0, img_np.shape)
        noisy_img_np = img_np + g_noise
        noisy_img_np = np.clip(noisy_img_np, 0, 255)
        img_distorted = Image.fromarray(noisy_img_np.astype(np.uint8))

    elif attack == "color_jitter":
        set_random_seed(seed)
        jitter_transform = transforms.ColorJitter(brightness=0.5)
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


def green_check(hc, indices_orig):
    green = 0
    red = 0
    for i in indices_orig:
        if i.item() in hc.green_list:
            green = green + 1
        else:
            red = red + 1
    return green / (green + red)


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
    parser.add_argument("--WATERMARK-THRESHOLD", type=float, default=0.615)
    parser.add_argument("--target-image-size", type=int, default=512)
    parser.add_argument(
        "--chosen-attack",
        type=str,
        default="cropping",
        choices=["jpeg", "cropping", "blurring", "noise", "color_jitter", "none"],
    )
    parser.add_argument("--jpeg-attack-quality", type=int, default=70)
    parser.add_argument(
        "--Watermarked-dir", type=str, default="images/Gen_Image/100%-512"
    )
    parser.add_argument(
        "--Not-Watermarked-dir", type=str, default="images/Gen_Image/50%-512"
    )
    parser.add_argument("--distortion-seed", type=int, default=123)

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
    hc = HierarchicalCodebook(
        codebook_vectors=codebook_weights,
        replacement_ratio=args.replacement_ratio,
        mapping_save_path=args.mapping_save_path,
        pairs_save_path=args.pairs_save_path,
        load_mapping=True,
        load_pairs=True,
    )
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

        for idx, filename in enumerate(image_files):
            image_path = os.path.join(image_directory, filename)
            current_seed = args.distortion_seed + idx

            try:
                original_pil_image = Image.open(image_path).convert("RGB")

                distorted_pil_image = image_distortion(
                    original_pil_image,
                    seed=current_seed,
                    attack=args.chosen_attack,
                    jpeg_quality=args.jpeg_attack_quality,
                )

                x_input = load_pil_image_to_tensor(
                    distorted_pil_image, target_device=device
                )

                with torch.no_grad():
                    if args.chosen_attack == "cropping":
                        rate_list = []
                        crop_size = (x_input.size(2), x_input.size(3))
                        crop_list, crop_index = place_crop_and_get_indices(
                            args.target_image_size, x_input, device
                        )
                        for idx, i in enumerate(crop_list):
                            latent_orig, _, [_, _, indices_i] = vq_model.encode(i)
                            indices_orig = indices_i.flatten()
                            rate = green_check(hc, indices_i[crop_index[idx]])
                            rate_list.append(rate)
                            current_ratio = max(rate_list)
                            if current_ratio > args.WATERMARK_THRESHOLD:
                                break
                    else:
                        latent_orig, _, misc = vq_model.encode(x_input)
                        indices_orig = misc[2]
                        indices_orig = indices_orig.flatten()

                        white = 0
                        red = 0
                        total_indices = 0
                        for i in indices_orig:
                            idx_item = i.item()
                            if idx_item in hc.green_list:
                                white += 1
                            else:
                                red += 1
                            total_indices += 1

                        current_ratio = white / total_indices
                    all_calculated_ratios.append(current_ratio)

                    prediction = 1 if current_ratio > args.WATERMARK_THRESHOLD else 0

                    all_true_labels.append(true_label)
                    all_predictions.append(prediction)
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
    print(f"Watermark Prediction Threshold: > {args.WATERMARK_THRESHOLD}")
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

    else:
        print(
            "\nCould not calculate accuracy. Not enough data processed or mismatch in label/prediction counts."
        )

    print("\nScript finished.")
