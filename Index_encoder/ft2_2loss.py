import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm
import glob
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from tokenizer.tokenizer_image.vq_model import VQ_models, Encoder, Decoder, VectorQuantizer, ModelArgs


class PreprocessedReconDataset(Dataset):
    def __init__(self, preprocessed_dir):
        self.preprocessed_dir = preprocessed_dir
        print(f"Searching for file pairs in: {preprocessed_dir}")
        start_time = time.time()
        self.recon_files = sorted(glob.glob(os.path.join(preprocessed_dir, "*_recon_disk.pt")))

        self.file_pairs = []
        skipped_count = 0
        for recon_f in tqdm.tqdm(self.recon_files, desc="Finding file pairs"):
            basename = os.path.basename(recon_f).replace("_recon_disk.pt", "")
            target_f = os.path.join(preprocessed_dir, f"{basename}_z_q_orig.pt")
            if os.path.exists(target_f):
                self.file_pairs.append((recon_f, target_f))
            else:
                skipped_count += 1
        end_time = time.time()
        print(f"Found {len(self.file_pairs)} file pairs.")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} files because corresponding target was missing.")
        print(f"File pair search took {end_time - start_time:.2f} seconds.")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        recon_path, target_path = self.file_pairs[idx]
        try:
            recon_tensor_disk = torch.load(recon_path, map_location='cpu')
            target_tensor = torch.load(target_path, map_location='cpu')
            return recon_tensor_disk, target_tensor
        except Exception as e:
            print(f"Error loading item at index {idx} (Paths: {recon_path}, {target_path}): {e}. Returning None.")
            return None, None


def collate_fn(batch):
    original_size = len(batch)
    batch = list(filter(lambda x: x is not None and x[0] is not None and x[1] is not None, batch))
    filtered_size = len(batch)

    if original_size > filtered_size:
        print(f"Collate Warning: Filtered out {original_size - filtered_size} items due to loading errors or None values.")

    if not batch:
        return None, None

    try:
        recon_tensors = torch.stack([item[0] for item in batch], dim=0)
        target_tensors = torch.stack([item[1] for item in batch], dim=0)
        return recon_tensors, target_tensors
    except Exception as e:
        return None, None


def fine_tune_encoder_on_preprocessed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    log_file_path = args.log_file
    print(f"Logging epoch average loss to: {log_file_path}")

    print(f"Initializing VQ model: {args.vq_model}")
    model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size, 
        codebook_embed_dim=args.codebook_embed_dim) 
    model.to(device)
    print(f"VQ model initialized and moved to {device}.")

    print(f"Loading VQ checkpoint from: {args.vq_ckpt}")
    if not os.path.exists(args.vq_ckpt):
        print(f"Error: VQ checkpoint file not found at {args.vq_ckpt}")
        sys.exit(1)

    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")

    if "ema" in checkpoint:
        print("Using 'ema' weights from checkpoint.")
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:
        print("Using 'model' weights from checkpoint.")
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        print("Using 'state_dict' weights from checkpoint.")
        model_weight = checkpoint["state_dict"]
    else:
        print("Error: Checkpoint does not contain known weight keys ('ema', 'model', 'state_dict').")
        sys.exit(1)

    finetune_state_dict = model.state_dict()
    non_encoder_weights_loaded_count = 0
    total_non_encoder_weights = 0
    loaded_keys = set()
    encoder_keys_in_ckpt = 0

    print("Loading non-encoder weights from checkpoint...")
    for k, v in model_weight.items():
        key_no_prefix = k.replace('module.', '')
        if key_no_prefix.startswith('encoder.'):
            encoder_keys_in_ckpt += 1
            continue

        total_non_encoder_weights += 1
        if key_no_prefix in finetune_state_dict:
            if finetune_state_dict[key_no_prefix].shape == v.shape:
                finetune_state_dict[key_no_prefix] = v
                non_encoder_weights_loaded_count += 1
                loaded_keys.add(key_no_prefix)
            else:
                print(f"  Shape mismatch for {key_no_prefix}: Ckpt has {v.shape}, Model needs {finetune_state_dict[key_no_prefix].shape}. Weight not loaded.")
        else:
            print(f"  Key {key_no_prefix} from checkpoint not found in the current model.")

    print(f"Loaded {non_encoder_weights_loaded_count} out of {total_non_encoder_weights} non-encoder weights found in checkpoint.")
    if encoder_keys_in_ckpt > 0:
         print(f"Skipped loading {encoder_keys_in_ckpt} encoder weights (will be fine-tuned or loaded from resume checkpoint).")

    non_encoder_keys_in_model = {k for k in finetune_state_dict.keys() if not k.startswith('encoder.')}
    missing_keys = non_encoder_keys_in_model - loaded_keys
    if missing_keys:
        print(f"Warning: The following {len(missing_keys)} non-encoder weights exist in the model but were NOT found in the checkpoint:")
        for k in sorted(list(missing_keys)):
            print(f"    - {k} (Shape: {finetune_state_dict[k].shape})")

    model.load_state_dict(finetune_state_dict, strict=False)
    print("Non-encoder weights loaded into model.")

    del checkpoint, model_weight, finetune_state_dict, loaded_keys, missing_keys

    encoder_param_count = 0
    trainable_param_count = 0
    print("Setting requires_grad for model parameters...")
    for name, param in model.named_parameters():
        if name.startswith('encoder.'):
            param.requires_grad = True
            encoder_param_count += param.numel()
            trainable_param_count += param.numel()
        else:
            param.requires_grad = False

    print(f"Total parameters in encoder: {encoder_param_count}")
    print(f"Total trainable parameters: {trainable_param_count}")
    if encoder_param_count == 0:
        print("Error: No parameters found starting with 'encoder.'. Cannot fine-tune.")
        sys.exit(1)

    model.train()
    model.encoder.train()
    model.decoder.eval()
    model.quantize.eval()
    model.quant_conv.eval()
    model.post_quant_conv.eval()
    print("Set model modes: Encoder to train, others to eval.")

    trainable_model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_model_params, lr=args.finetune_lr, betas=(0.9, 0.95))
    print(f"Optimizer: Adam, Learning Rate: {args.finetune_lr}, Betas: (0.9, 0.95)")

    start_epoch = 0
    latest_checkpoint_path = None
    checkpoint_files = glob.glob(os.path.join(args.output_dir, "encoder_finetuned_target_zq_epoch_*.pt"))
    if checkpoint_files:
        latest_epoch = -1
        for f in checkpoint_files:
            try:
                epoch_num = int(os.path.basename(f).split('_')[-1].replace('.pt', ''))
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_checkpoint_path = f
            except ValueError:
                continue
        if latest_checkpoint_path:
            print(f"Attempting to resume training from latest checkpoint: {latest_checkpoint_path}")
            try:
                checkpoint_data = torch.load(latest_checkpoint_path, map_location=device)

                if isinstance(checkpoint_data, dict) and 'encoder_state_dict' in checkpoint_data:
                    model.encoder.load_state_dict(checkpoint_data['encoder_state_dict'])
                    if 'optimizer_state_dict' in checkpoint_data:
                         optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                         print("Loaded optimizer state from checkpoint.")
                    else:
                         print("Warning: Checkpoint missing 'optimizer_state_dict'. Optimizer state not loaded.")

                    if 'epoch' in checkpoint_data:
                         start_epoch = checkpoint_data['epoch']
                         print(f"Resumed model and optimizer from epoch {start_epoch}.")
                    else:
                         try:
                             start_epoch = int(os.path.basename(latest_checkpoint_path).split('_')[-1].replace('.pt', ''))
                             print(f"Warning: Checkpoint dictionary missing 'epoch' key. Inferred start epoch {start_epoch} from filename.")
                         except ValueError:
                             print("Warning: Could not determine start epoch from checkpoint. Starting from epoch 0.")
                             start_epoch = 0
                else:
                    print("Checkpoint appears to be in the old format (only encoder weights).")
                    model.encoder.load_state_dict(checkpoint_data)
                    try:
                        start_epoch = int(os.path.basename(latest_checkpoint_path).split('_')[-1].replace('.pt', ''))
                        print(f"Resumed model state from epoch {start_epoch}. Optimizer state not loaded (old format).")
                    except ValueError:
                        print("Warning: Could not determine start epoch from old-format checkpoint filename. Starting from epoch 0.")
                        start_epoch = 0
            except Exception as e:
                print(f"Error loading checkpoint {latest_checkpoint_path}: {e}")
                print("Starting training from scratch.")
                start_epoch = 0
                trainable_model_params = filter(lambda p: p.requires_grad, model.parameters())
                optimizer = torch.optim.Adam(trainable_model_params, lr=args.finetune_lr, betas=(0.9, 0.95))
        else:
            print("No valid checkpoint files found matching the pattern. Starting training from scratch.")
    else:
        print("No checkpoint files found in the output directory. Starting training from scratch.")


    print("Initializing Dataset...")
    preprocessed_dataset = PreprocessedReconDataset(preprocessed_dir=args.preprocessed_dir)
    if len(preprocessed_dataset) == 0:
        print(f"Error: No valid data found in {args.preprocessed_dir}. Please check the directory and file naming convention.")
        sys.exit(1)

    print(f"Dataset size: {len(preprocessed_dataset)}")
    print("Initializing DataLoader...")
    dataloader = DataLoader(preprocessed_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            collate_fn=collate_fn,
                            drop_last=True)
    print(f"DataLoader initialized with Batch Size: {args.batch_size}, Num Workers: {args.num_workers}")

    criterion = nn.MSELoss()
    print("Using MSELoss for h_recon vs z_q_orig and L2 reconstruction loss.")
    loss_weight_l2 = 0.5
    print(f"Using L2 loss weight: {loss_weight_l2}")

    print(f"Starting training from epoch {start_epoch + 1} for {args.epochs} total epochs.")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        model.encoder.train()

        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch", leave=False)
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        num_valid_batches = 0
        batch_load_errors = 0

        for batch_idx, (recon_disk_batch, z_q_orig_target_batch) in enumerate(progress_bar):

            if recon_disk_batch is None or z_q_orig_target_batch is None:
                batch_load_errors += 1
                continue

            recon_disk_batch = recon_disk_batch.to(device, non_blocking=True)
            z_q_orig_target_batch = z_q_orig_target_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            try:
                z_e_recon = model.encoder(recon_disk_batch)
                h_recon = model.quant_conv(z_e_recon)

                z_q, _, _ = model.quantize(h_recon)
                recon_disk_batch_new = model.decode(z_q)

                loss1 = criterion(h_recon, z_q_orig_target_batch.detach())

                loss2 = criterion(recon_disk_batch, recon_disk_batch_new)

                loss = loss1 + loss2 * loss_weight_l2

                loss.backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        args.grad_clip
                    )

                optimizer.step()

                total_loss += loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                num_valid_batches += 1

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    loss1=f"{loss1.item():.4f}",
                    loss2=f"{loss2.item():.4f}",
                    avg_loss=f"{total_loss / num_valid_batches:.4f}"
                )

            except Exception as e:
                print(f"\nError during training step in batch {batch_idx}: {e}")
                print(f"  Input shape: {recon_disk_batch.shape if recon_disk_batch is not None else 'None'}")
                print(f"  Target shape: {z_q_orig_target_batch.shape if z_q_orig_target_batch is not None else 'None'}")
                import traceback
                traceback.print_exc()
                optimizer.zero_grad()
                continue

        progress_bar.close()

        if batch_load_errors > 0:
             print(f"Warning: Skipped {batch_load_errors} batches in epoch {epoch+1} due to loading/collation errors.")

        if num_valid_batches > 0:
            avg_epoch_loss = total_loss / num_valid_batches
            avg_epoch_loss1 = total_loss1 / num_valid_batches
            avg_epoch_loss2 = total_loss2 / num_valid_batches

            print(f"Epoch {epoch+1} finished. Processed {num_valid_batches} batches.")
            print(f"  Avg Loss: {avg_epoch_loss:.6f} | Avg Loss1 (Latent MSE): {avg_epoch_loss1:.6f} | Avg Loss2 (Recon MSE): {avg_epoch_loss2:.6f}")

            try:
                with open(log_file_path, 'a') as log_f:
                    log_f.write(f"Epoch {epoch+1}: Avg Loss: {avg_epoch_loss:.6f}, Avg Loss1: {avg_epoch_loss1:.6f}, Avg Loss2: {avg_epoch_loss2:.6f}\n")
            except Exception as e:
                print(f"Warning: Could not write to log file {log_file_path}: {e}")

        else:
            print(f"Epoch {epoch+1} finished, but no valid batches were processed. Check data loading and collation.")

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            encoder_save_path = os.path.join(args.output_dir, f"encoder_finetuned_target_zq_epoch_{epoch+1}.pt")
            try:
                save_data = {
                    'epoch': epoch + 1,
                    'encoder_state_dict': model.encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args
                }
                torch.save(save_data, encoder_save_path)
                print(f"Checkpoint saved: {encoder_save_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("VQModel Encoder Fine-tuning (Dual Loss)")

    parser.add_argument("--vq-model", type=str, choices=VQ_models.keys() if VQ_models else [], default="VQ-16", help="VQ Model architecture type")
    parser.add_argument("--vq-ckpt", type=str, default="/root/autodl-tmp/LlamaGen/pretrained_models/vq_ds16_t2i.pt", help="Path to the pre-trained VQ model checkpoint")
    parser.add_argument("--codebook-size", type=int, default=16384, help="Size of the codebook")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="Dimension of codebook embeddings")

    parser.add_argument("--preprocessed-dir", type=str, required=True, help="Directory containing preprocessed '*_recon_disk.pt' and '*_z_q_orig.pt' files")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader worker processes")

    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="Learning rate for the Adam optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value (0 to disable)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--output-dir", type=str, default="./finetuned_encoder_target_zq_dual_loss", help="Directory to save checkpoints and logs")
    parser.add_argument("--save-interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--log-file", type=str, default="training_loss_dual.log", help="Filename for logging epoch average loss (relative to output-dir)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("----- Configuration -----")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-------------------------")


    fine_tune_encoder_on_preprocessed(args)

    print("Training finished.")
