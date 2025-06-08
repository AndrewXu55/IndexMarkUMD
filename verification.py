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

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from tokenizer.tokenizer_image.vq_model import VQ_models
from generation import HierarchicalCodebook


def load_image_to_tensor(image_path, target_device='cpu'):
    pil_image = Image.open(image_path).convert('RGB')
    img_np_uint8 = np.array(pil_image)
    img_np_float01 = img_np_uint8.astype(np.float32) / 255.0
    img_np_float_neg1_pos1 = 2.0 * img_np_float01 - 1.0
    tensor_hwc = torch.from_numpy(img_np_float_neg1_pos1)
    tensor_chw = tensor_hwc.permute(2, 0, 1)
    tensor_nchw = tensor_chw.unsqueeze(0)
    reconstructed_tensor = tensor_nchw.to(target_device).float()
    return reconstructed_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Image AR Generation with Pairing-based Codebook Replacement")

    parser.add_argument("--vq-ckpt", type=str, default="path/vq_ds16_t2i.pt", help="Path to VQ model checkpoint")
    parser.add_argument("--mapping-save-path", type=str, default="path/codebook_index_mapping.json", help="Path to save/load the index mapping JSON file")
    parser.add_argument("--pairs-save-path", type=str, default="path/codebook_pairs.json", help="Path to save/load the codebook pair list JSON file")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()) if VQ_models else ["VQ-16"], default="VQ-16", help="VQ model architecture")
    parser.add_argument("--codebook-size", type=int, default=16384, help="Codebook size for VQ")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="Codebook embedding dimension for VQ")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"], help="Computation precision (none means fp32)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--replacement-ratio", type=float, default=1.0, help="Ratio of red-listed tokens to be replaced (0.0-1.0)")
    parser.add_argument("--load-mapping", action='store_true', help="Attempt to load index mapping from file instead of recomputing")
    parser.add_argument("--load-pairs", action='store_true', help="Attempt to load codebook pair list from file instead of recomputing")
    parser.add_argument("--index-encoder",type=str, default=None)
    parser.add_argument("--image-directory",type=str, default='path/Gen_Image/100%-256')

    args, _ = parser.parse_known_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        print("Warning: BF16 precision was requested, but is not supported on this GPU. Falling back to FP32.")
        args.precision = "none"

    hc = None
    codebook_weights = None

    try:
        print("Loading VQ model...")
        vq_model_class = VQ_models.get(args.vq_model)
        vq_model = vq_model_class(
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim).to(device) 

        should_load_vq_weights = not (args.load_pairs and args.load_mapping and
                                      os.path.exists(args.pairs_save_path) and
                                      os.path.exists(args.mapping_save_path))

        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint["ema"])

        if args.index_encoder != None:
            finetuned_checkpoint = torch.load(args.index_encoder, map_location="cpu")
            encoder_state_dict = finetuned_checkpoint['encoder_state_dict']
            missing_keys, unexpected_keys = vq_model.encoder.load_state_dict(encoder_state_dict, strict=False)

        codebook_weights = vq_model.quantize.embedding.weight.data.cpu().numpy().astype(np.float32)
        print(f"Codebook weights extracted (shape: {codebook_weights.shape})")

        hc = HierarchicalCodebook(
            codebook_vectors=codebook_weights,
            replacement_ratio=args.replacement_ratio,
            mapping_save_path=args.mapping_save_path,
            pairs_save_path=args.pairs_save_path,
            load_mapping=True,
            load_pairs=True
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_ratios = []
    processed_image_count = 0
    image_directory = args.image_directory
    all_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    for filename in image_files:
        image_path = os.path.join(image_directory, filename)
        x_input = load_image_to_tensor(image_path, target_device=device)
        with torch.no_grad():
            latent_orig, _, [_, _, indices_orig] = vq_model.encode(x_input)
            indices_orig = indices_orig.flatten()
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
    else:
        print("\nNo images were successfully processed.")
