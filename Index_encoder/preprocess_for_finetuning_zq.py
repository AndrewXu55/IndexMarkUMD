import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import argparse
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import tqdm
import tempfile
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from tokenizer.tokenizer_image.vq_model import VQ_models, Encoder, Decoder, VectorQuantizer, ModelArgs
except ImportError as e:
    print(f"Could not import VQ model classes: {e}")
    print("Please ensure the tokenizer.tokenizer_image.vq_model file exists and the path is correct, or adjust PROJECT_ROOT above.")
    sys.exit(1)

class CocoGroundTruthDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not self.image_files:
             raise ValueError(f"No images found in directory {image_dir}")
        print(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def center_crop_arr(self, pil_image, image_size):
        width, height = pil_image.size
        crop_size = min(width, height)
        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2
        pil_image = pil_image.crop((left, top, right, bottom))
        pil_image = pil_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        return pil_image

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            pil_image = Image.open(img_path).convert("RGB")
            pil_image = self.center_crop_arr(pil_image, self.image_size)
            img_np_uint8 = np.array(pil_image)
            img_np_float01 = img_np_uint8.astype(np.float32) / 255.0
            img_np_float_neg1_pos1 = 2.0 * img_np_float01 - 1.0
            tensor_chw = torch.from_numpy(img_np_float_neg1_pos1).permute(2, 0, 1)
            return tensor_chw
        except Exception as e:
            print(f"Error processing image {img_path}: {e}. Returning a placeholder.")
            return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)

def load_image_to_tensor(image_path, target_device='cpu'):
    try:
        pil_image = Image.open(image_path).convert('RGB')
        img_np_uint8 = np.array(pil_image)
        img_np_float01 = img_np_uint8.astype(np.float32) / 255.0
        img_np_float_neg1_pos1 = 2.0 * img_np_float01 - 1.0
        tensor_hwc = torch.from_numpy(img_np_float_neg1_pos1)
        tensor_chw = tensor_hwc.permute(2, 0, 1)
        reconstructed_tensor = tensor_chw.to(target_device).float()
        return reconstructed_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
        
def preprocess_data_for_zq_target(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading pre-trained VQ model...")
    try:
        model = VQ_models[args.vq_model](
        codebook_size=16384,
        codebook_embed_dim=8)
    except KeyError:
        print(f"Error: Model type '{args.vq_model}' is not in VQ_models. Options are: {list(VQ_models.keys())}")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating model: {e}")
        sys.exit(1)

    model.to(device)
    try:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.vq_ckpt}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    if "ema" in checkpoint: model_weight = checkpoint["ema"]
    elif "model" in checkpoint: model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint: model_weight = checkpoint["state_dict"]
    else: raise Exception("Could not find model weights in the checkpoint file")
    new_state_dict = {}
    for k, v in model_weight.items(): new_state_dict[k.replace('module.', '')] = v
    try: model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"Error loading state dict (possibly due to model architecture mismatch): {e}")
        print("Attempting to load with strict=False")
        try:
             model.load_state_dict(new_state_dict, strict=False)
        except Exception as e2:
             print(f"Loading with strict=False also failed: {e2}")
             sys.exit(1)
    del checkpoint, model_weight, new_state_dict
    model.eval()
    print("Model loaded and set to evaluation mode.")

    print("Preparing original dataset...")
    try:
        original_dataset = CocoGroundTruthDataset(image_dir=args.image_dir, image_size=args.image_size)
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        sys.exit(1)
    original_dataloader = DataLoader(original_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print("Original dataset ready.")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Preprocessed data will be saved to: {args.output_dir}")

    print("Starting preprocessing...")
    num_processed = 0
    num_skipped = 0
    original_basenames = [os.path.splitext(os.path.basename(f))[0] for f in original_dataset.image_files]

    for i, original_image_tensor in enumerate(tqdm.tqdm(original_dataloader, desc="Preprocessing Progress")):
        if torch.all(original_image_tensor == 0) and torch.numel(original_image_tensor)>0 :
            print(f"Image at index {i} failed to load (placeholder), skipping preprocessing.")
            num_skipped += 1
            continue
        if original_image_tensor.nelement() == 0:
            print(f"Skipping empty batch {i}")
            num_skipped += 1
            continue

        original_image_tensor = original_image_tensor.to(device)
        basename = original_basenames[i]

        recon_disk_save_path = os.path.join(args.output_dir, f"{basename}_recon_disk.pt")
        z_q_orig_save_path = os.path.join(args.output_dir, f"{basename}_z_q_orig.pt")

        if not args.overwrite and os.path.exists(recon_disk_save_path) and os.path.exists(z_q_orig_save_path):
            num_skipped += 1
            continue

        try:
            with torch.no_grad():
                z_e_orig = model.encoder(original_image_tensor)
                h_orig = model.quant_conv(z_e_orig)
                z_q_orig, _, _ = model.quantize(h_orig)

            reconstructed_tensor_mem = model.decode(z_q_orig)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=True, dir=args.temp_dir) as tmp_file:
                temp_path = tmp_file.name
                save_image(reconstructed_tensor_mem.squeeze(0), temp_path, normalize=True, value_range=(-1, 1))

                reconstructed_tensor_disk = load_image_to_tensor(temp_path, target_device='cpu')

            if reconstructed_tensor_disk is None:
                print(f"Failed to reload image: {basename}")
                num_skipped += 1
                continue

            z_q_orig_to_save = z_q_orig.squeeze(0).cpu()

            torch.save(reconstructed_tensor_disk, recon_disk_save_path)
            torch.save(z_q_orig_to_save, z_q_orig_save_path)

            num_processed += 1

        except Exception as e:
            print(f"\nCritical error processing image {basename} (index {i}): {e}")
            if os.path.exists(recon_disk_save_path): os.remove(recon_disk_save_path)
            if os.path.exists(z_q_orig_save_path): os.remove(z_q_orig_save_path)
            num_skipped += 1
            continue

    print(f"\nPreprocessing finished.")
    print(f"Successfully processed: {num_processed} images")
    print(f"Skipped (load failed or already exists): {num_skipped} images")
    print(f"Total: {len(original_dataset)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("VQModel Preprocessing Script - Generate (reload-from-disk image, z_q_orig) data pairs")
    parser.add_argument("--vq-model", type=str, choices=VQ_models.keys() if VQ_models else [], default="VQ-16", help="VQ model type")
    parser.add_argument("--vq-ckpt", type=str, default="/root/autodl-tmp/LlamaGen/pretrained_models/vq_ds16_t2i.pt", help="Pre-trained VQ model checkpoint")
    parser.add_argument("--codebook-size", type=int, default=16384, help="Codebook size")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="Codebook embedding dimension")
    parser.add_argument("--image-dir", type=str, default="/root/autodl-tmp/LlamaGen/coco/TREE/ground_truth", help="Directory containing original images")
    parser.add_argument("--image-size", type=int, default=256, help="Target image size")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save preprocessed data pairs (.pt files)")
    parser.add_argument("--overwrite", action='store_true', help="If set, overwrite existing preprocessed files")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes for the data loader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temp-dir", type=str, default=None, help="Directory for storing temporary image files (defaults to system temp)")

    args = parser.parse_args()

    if args.temp_dir:
        os.makedirs(args.temp_dir, exist_ok=True)

    preprocess_data_for_zq_target(args)
