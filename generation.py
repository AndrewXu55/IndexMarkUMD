import argparse
import gc
import json
import os
import sys
import time
import traceback
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.utils import save_image

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from autoregressive.models.generatearcon import generate
from autoregressive.models.gpt import GPT_models
from language.t5 import T5Embedder
from tokenizer.tokenizer_image.vq_model import VQ_models

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)


class HierarchicalCodebook:
    def __init__(self, codebook_vectors, replacement_ratio=1.0,
                 mapping_save_path="codebook_index_mapping.json",
                 pairs_save_path="codebook_pairs.json",
                 load_mapping=False, load_pairs=False,device="cuda",
                 pairing_method="Default"):
        
        self.codebook = codebook_vectors
        self.codebook_size = codebook_vectors.shape[0]
        self.feature_dim = codebook_vectors.shape[1]
        self.replacement_ratio = replacement_ratio
        self.mapping_save_path = mapping_save_path
        self.pairs_save_path = pairs_save_path

        self.pairs = []
        self.unassigned_indices = set(range(self.codebook_size))
        self.red_list = set()
        self.green_list = set()
        self.index_mapping = {}

        loaded_pairs = False
        loaded_mapping = False

        if load_pairs:
            loaded_pairs = self.load_pairs(self.pairs_save_path)
            if loaded_pairs:
                self._reconstruct_state_from_pairs()

        if load_mapping:
            loaded_mapping = self.load_mapping(self.mapping_save_path)
            if loaded_mapping and not loaded_pairs:
                print("Warning: The mapping has been loaded, but the pairing list has not been loaded.")
                self.red_list = set(self.index_mapping.keys())
                self.green_list = set(self.index_mapping.values())
        self.red_list_tensor = torch.tensor(list(self.red_list), dtype=torch.long, device=device)

        calculation_needed = True
        if loaded_pairs and loaded_mapping:
            print("Successfully loaded the pairing list and index mapping from the file.")
            calculation_needed = False
        elif loaded_pairs and not loaded_mapping:
            print("The pairing list has been loaded, but the mapping has not been loaded. Mapping will be generated based on the loaded pairing.")
            self._create_red_green_lists()
            self.save_mapping(self.mapping_save_path)
            calculation_needed = False
        elif not loaded_pairs and loaded_mapping:
            print("The mapping has been loaded, but the pairing list has not been loaded. The loaded mapping will be used to skip pairing calculations.")
            calculation_needed = False 
        else:
            print("Calculate the pairing list and index mapping.")

        if calculation_needed:
            if pairing_method == "Default":
                self._build_pairs()             
            else:
                self.build_pairs_clustering()
            if not self.pairs:
                print("Error: Pairing build failed or resulted in an empty pairing list. Cannot continue.")
                raise RuntimeError("Failed to build codebook pairing.")
            self._create_red_green_lists()
            self.save_mapping(self.mapping_save_path)

        print("HierarchicalCodebook initialization completed.")
        print(f"  - Number of pairings: {len(self.pairs)}")
        print(f"  - Mapping size: {len(self.index_mapping)}")
        print(f"  - Unallocated index number: {len(self.unassigned_indices)}")

    def _build_pairs(self):
        print("Calculating similarity matrix ..")
        try:
            norms = np.linalg.norm(self.codebook, axis=1, keepdims=True)
            zero_norm_mask = (norms < 1e-10).flatten()
            if np.any(zero_norm_mask):
                print(f"Warning: {np.sum(zero_norm_mask)} zero norm vectors were found in the codebook. The similarity involving them will be 0.")
                norms[zero_norm_mask.reshape(-1, 1)] = 1e-10 

            norm_codebook = self.codebook / norms
            sim_matrix = cosine_similarity(norm_codebook)
            del norm_codebook, norms 
            gc.collect()
        except MemoryError:
            print("Error: Insufficient memory while calculating the complete similarity matrix.")
            raise
        np.fill_diagonal(sim_matrix, -np.inf)
        sim_matrix[zero_norm_mask, :] = -np.inf
        sim_matrix[:, zero_norm_mask] = -np.inf

        print(f"Pairing {self.codebook_size} codebook vectors ...")
        assigned_mask = np.zeros(self.codebook_size, dtype=bool) 
        num_pairs_formed = 0
        target_pairs = self.codebook_size // 2
        self.pairs = []

        for i in range(target_pairs):
            flat_idx = np.argmax(sim_matrix) 
            idx1, idx2 = np.unravel_index(flat_idx, sim_matrix.shape) 

            if sim_matrix[idx1, idx2] <= -np.inf:
                print(f"Warning: Stop pairing in round {i+1}. No more valid pairs were found (maximum similarity is - nf).")
                break

            self.pairs.append(tuple(sorted((int(idx1), int(idx2))))) 
            assigned_mask[idx1] = True
            assigned_mask[idx2] = True
            num_pairs_formed += 1

            sim_matrix[idx1, :] = -np.inf
            sim_matrix[:, idx1] = -np.inf
            sim_matrix[idx2, :] = -np.inf
            sim_matrix[:, idx2] = -np.inf

            if (num_pairs_formed % 500 == 0) or (num_pairs_formed == target_pairs):
                print(f"  Pairing has been formed: {num_pairs_formed}/{target_pairs}")

            if num_pairs_formed % 1000 == 0:
                 gc.collect()

        del sim_matrix 
        gc.collect()

        self.unassigned_indices = set(np.where(~assigned_mask)[0])

        print(f"\nPairing construction completed.")
        print(f"  Total number of pairs formed:{len(self.pairs)}")
        if self.unassigned_indices:
            print(f"  Note: There are {len(self.unassigned_indices)} indexes that have not been assigned.")
        else:
             print(f"  All {self.codebook_size} indexes have been successfully assigned to the pairing.")

        if hasattr(self, 'pairs_save_path') and self.pairs_save_path:
            self.save_pairs(self.pairs_save_path) 
        else:
            print("Warning: The pairing save path (pairs_save_path) has not been set. Skip saving the pairing list.")

        print("\nAnalyzing the similarity within the formed pairs ...")
        self._analyze_pair_similarities_direct()

    def build_pairs_clustering(self, k=1000, balance_threshold=0.05):
        """
        Build pairs using K-means clustering instead of greedy pairing using cosine similarity.
        
        Args:
            k: Number of clusters
            balance_threshold: Maximum allowed deviation from 50/50 split
        """
        
        print(f"Building pairs using clustering method with k={k} clusters...")
        
        # Step 1: Normalize vectors and handle zero-norm vectors
        try:
            norms = np.linalg.norm(self.codebook, axis=1, keepdims=True)
            zero_norm_mask = (norms < 1e-10).flatten()
            if np.any(zero_norm_mask):
                print(f"Warning: {np.sum(zero_norm_mask)} zero norm vectors found. They will be assigned randomly.")
                norms[zero_norm_mask.reshape(-1, 1)] = 1.0  # Prevent division by zero
            norm_codebook = self.codebook / norms
        except MemoryError:
            print("Error: Insufficient memory while normalizing vectors.")
            raise
        
        # Step 2: Perform K-means clustering
        print(f"Performing K-means clustering into {k} clusters...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(norm_codebook)
        
        # Step 3: Look at bin count per cluster
        cluster_sizes = np.bincount(cluster_labels, minlength=k)
        print(f"Cluster sizes - Min: {cluster_sizes.min()}, Max: {cluster_sizes.max()}, Mean: {cluster_sizes.mean():.2f}")
        
        # Step 4: Assign clusters to red/green to achieve ~50/50 balance
        print("Assigning clusters to red/green groups...")
        sorted_indices = np.argsort(cluster_sizes)[::-1]
        red_green_assignment = np.zeros(k, dtype=int)
        red_count = 0
        green_count = 0
        # Greedy assignment: assign each cluster to the group with fewer tokens
        for cluster_id in sorted_indices:
            size = cluster_sizes[cluster_id]
            if red_count <= green_count:
                red_green_assignment[cluster_id] = 0  # Red
                red_count += size
            else:
                red_green_assignment[cluster_id] = 1  # Green
                green_count += size
        token_assignments = np.array([red_green_assignment[label] for label in cluster_labels])
        
        # Step 5: Analyze the results and the balance of red and green tokens
        print(f"\nClustering-based pairing completed.")
        print(f"  Total pairs formed: {len(self.pairs)}")
        print(f"  Unassigned indices: {len(self.unassigned_indices)}")
        num_red = np.sum(token_assignments == 0)
        num_green = np.sum(token_assignments == 1)
        total = len(token_assignments)
        red_pct = num_red / total
        green_pct = num_green / total
        imbalance = abs(red_pct - 0.5)
        print(f"\nFinal token distribution:")
        print(f"- Red tokens: {num_red} ({red_pct * 100:.2f}%)")
        print(f"- Green tokens: {num_green} ({green_pct * 100:.2f}%)")
        print(f"- Imbalance: {imbalance * 100:.2f}%")
        if abs(red_pct - 0.5) > balance_threshold:
            print(f"Imbalance EXCEEDS threshold: {balance_threshold*100}%!")
        
        if hasattr(self, 'pairs_save_path') and self.pairs_save_path:
            self.save_pairs(self.pairs_save_path)
        
        print("\nAnalyzing pair similarities...")
        self._analyze_pair_similarities_direct()
        
        self.token_red_green_assignment = token_assignments
        self.cluster_labels = cluster_labels
        
        del norm_codebook, kmeans
        gc.collect()

    def build_pairs_random(self):
        pass

    def _analyze_pair_similarities_direct(self):
        """Directly using codebook vectors to analyze the similarity within the formed pairs."""
        if not self.pairs:
            print("There are no available pairs for similarity analysis.")
            return

        similarities = []
        count = 0
        print("  Start analyzing pairing similarity...") 
        for idx1, idx2 in self.pairs:
            vec1 = self.codebook[idx1]
            vec2 = self.codebook[idx2]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 1e-9 and norm2 > 1e-9:
                sim = np.dot(vec1, vec2) / (norm1 * norm2)
                similarities.append(sim)
            else:
                similarities.append(0.0)
            count += 1
            if count % 2000 == 0:
                 print(f"    Analyzed the similarity between {count}/{len(self.pairs)} pairs...")

        similarities = np.array(similarities)
        print("  Pairing similarity analysis completed.")

        print(f"\n===== Pairing similarity analysis results =====")
        if len(similarities) > 0:
            print(f"  The number of pairs analyzed: {len(similarities)}")
            print(f"  Average similarity: {np.mean(similarities):.4f}")
            print(f"  Minimum similarity: {np.min(similarities):.4f}")
            print(f"  Similarity measure : {np.max(similarities):.4f}")
            print(f"  Similarity standard deviation: {np.std(similarities):.4f}")
        else:
            print("  There are no valid similarity values available for analysis.")

    def _create_red_green_lists(self):
        """Create a red/green list and mapping based on the self.pairs list."""
        self.red_list.clear()
        self.green_list.clear()
        self.index_mapping.clear()

        if not self.pairs:
            print("Warning: Unable to create red/green list because the pairing list is empty.")
            return

        print("Creating red/green list based on pairing...")
        for idx1, idx2 in self.pairs:
            if np.random.random() < 0.5:
                red_idx, green_idx = idx1, idx2
            else:
                red_idx, green_idx = idx2, idx1
            self.red_list.add(red_idx)
            self.green_list.add(green_idx)
            self.index_mapping[int(red_idx)] = int(green_idx)

        print(f"\n===== Red/Green List Statistics =====")
        print(f"  The number of pairs processed: {len(self.pairs)}")
        print(f"  Red List Size: {len(self.red_list)}")
        print(f"  Greenlist size: {len(self.green_list)}")
        expected_total = len(self.pairs) * 2
        actual_total = len(self.red_list) + len(self.green_list)
        if actual_total != expected_total:
             print(f"  Warning: The total number of red/green lists ({actual_total}) does not match the expected ({expected_total}).")

    def get_random_replacement(self, original_idx, log_ratio_value, threshold):
        """Replacement logic: Consider confidence level."""
        try:
            log_ratio_val = float(log_ratio_value)
            threshold_val = float(threshold)
        except (ValueError, TypeError):
            print(f"Warning: Invalid log_ratio-value ({log_ratio_value}) or threshold ({threshold}). Skip replacement check.")
            return original_idx

        if (original_idx in self.red_list and
            np.random.random() < self.replacement_ratio and
            log_ratio_val <= threshold_val):
            return self.index_mapping.get(original_idx, original_idx)
        return original_idx

    def save_mapping(self, filepath):
        """Save the index_mapping dictionary to a JSON file."""
        print(f"Saving index mapping to {filepath}...")
        if not self.index_mapping:
            print("Warning: Index mapping is empty, no need to save.")
            return
        try:
            serializable_mapping = {int(k): int(v) for k, v in self.index_mapping.items()}
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_mapping, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved {len(self.index_mapping)} mapping entries.")
        except TypeError as e:
            print(f"Error: The serialization index mapping failed. Please check the data type. Error: {e}")
        except Exception as e:
            print(f"Error: {e}")

    def load_mapping(self, filepath):
        """Load the index_mapping dictionary from the JSON file."""
        print(f"Attempting to load index mapping from {filepath}...")
        if not os.path.exists(filepath):
            print(f"Information: Mapping file {filepath} not found.")
            return False
        try:
            with open(filepath, 'r', encoding='utf-8') as f: 
                loaded_mapping_str_keys = json.load(f)
            self.index_mapping = {int(k): int(v) for k, v in loaded_mapping_str_keys.items()}
            self.red_list = set(self.index_mapping.keys())
            self.green_list = set(self.index_mapping.values())
            print(f"Successfully loaded {len(self.index_mapping)} mapping entries from {filepath}.")
            return True
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON file {filepath}. The file may be corrupted. Error: {e}")
            self.index_mapping = {} 
            return False
        except Exception as e:
            print(f"Error: Failed to load index mapping from {filepath}. Error: {e}")
            self.index_mapping = {} 
            return False

    def save_pairs(self, filepath):
        """Save the self.pairs list to a JSON file."""
        print(f"Saving the pairs list to {filepath}...")
        if not self.pairs:
            print("Warning: The pairs list is empty, no need to save.")
            return
        try:
            list_of_lists = [list(pair) for pair in self.pairs]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(list_of_lists, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved {len(self.pairs)} pairs.")
        except TypeError as e:
             print(f"Error: Failed to serialize the pairs list. Please check the data types. Error: {e}")
        except Exception as e:
            print(f"Error: Failed to save the pairs list to {filepath}. Error: {e}")

    def load_pairs(self, filepath):
        """Load the self.pairs list from a JSON file."""
        print(f"Attempting to load the pairs list from {filepath}...")
        if not os.path.exists(filepath):
            print(f"Info: Pairs file {filepath} not found.")
            return False
        try:
            with open(filepath, 'r', encoding='utf-8') as f: 
                loaded_list = json.load(f)
            self.pairs = [tuple(map(int, pair)) for pair in loaded_list if isinstance(pair, list) and len(pair) == 2]
            print(f"Successfully loaded {len(self.pairs)} pairs from {filepath}.")
            return True
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON file {filepath}. The file may be corrupted. Error: {e}")
            self.pairs = [] 
            return False
        except Exception as e:
            print(f"Error: Failed to load the pairs list from {filepath}. Error: {e}")
            self.pairs = [] 
            return False

    def _reconstruct_state_from_pairs(self):
        """Update assigned_mask and unassigned_indices based on self.pairs."""
        if not self.pairs:
            print("Warning: Unable to rebuild state from an empty pairing list.")
            return
        print("Rebuilding allocation status based on loaded pairings ..")
        assigned_mask = np.zeros(self.codebook_size, dtype=bool)
        valid_pairs_count = 0
        for idx1, idx2 in self.pairs:
            if 0 <= idx1 < self.codebook_size and 0 <= idx2 < self.codebook_size:
                assigned_mask[idx1] = True
                assigned_mask[idx2] = True
                valid_pairs_count += 1
            else:
                print(f"Warning: The loaded pair {(idx1, idx2)} contains an index that is out of bounds for the codebook size {self.codebook_size}. This pairing has been skipped.")
        self.unassigned_indices = set(np.where(~assigned_mask)[0])
        print(f"State reconstruction completed. Effective number of pairs used: {valid_pairs_count}. Unallocated index number: {len(self.unassigned_indices)}")


def load_prompts_from_csv(filename):
    """Loads prompts from a single-column CSV file."""
    prompts_list = []
    if not os.path.exists(filename):
        print(f"Error: Prompt file '{filename}' not found.")
        return None 

    print(f"Loading prompts from '{filename}'...")
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for row_number, row in enumerate(reader):
                if row:
                    prompt_text = row[0].strip() 
                    if prompt_text:
                        prompts_list.append(prompt_text)
                    else:
                        print(f"Warning: Skipping empty prompt at row {row_number + 1}.")
                else:
                     print(f"Warning: Skipping empty row {row_number + 1}.")
        print(f"Loaded {len(prompts_list)} prompts.")
        return prompts_list
    except Exception as e:
        print(f"Error reading prompts from '{filename}': {e}")
        return None


def mscocojson2list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    captions = []
    ids = []
    for i in range(len(data)):
        captions.append(data[i]["caption"])
        ids.append(data[i]["image_id"])
    return captions,ids


def vectorized_replacement_process(index_sample, log_ratio_flat, log_threshold, hc_instance):
    device = index_sample.device
    new_indices = index_sample.clone() 

    if hc_instance.red_list_tensor.device != device:
        hc_instance.red_list_tensor = hc_instance.red_list_tensor.to(device)

    original_shape = new_indices.shape
    new_indices_flat = new_indices.view(-1)
    if log_ratio_flat.shape[0] != new_indices_flat.shape[0]:
        raise ValueError(f"Shape mismatch: log_ratio_flat ({log_ratio_flat.shape}) vs new_indices_flat ({new_indices_flat.shape})")

    if hc_instance.red_list_tensor.numel() > 0: 
        condition1_mask = torch.isin(new_indices_flat, hc_instance.red_list_tensor)
    else:
        condition1_mask = torch.zeros_like(new_indices_flat, dtype=torch.bool, device=device)

    if not isinstance(log_ratio_flat, torch.Tensor):
         log_ratio_flat = torch.from_numpy(log_ratio_flat).to(device) 
    log_threshold_tensor = torch.tensor(log_threshold, device=device, dtype=log_ratio_flat.dtype)
    condition3_mask = log_ratio_flat <= log_threshold_tensor

    candidate_mask = condition1_mask & condition3_mask

    replacement_count = 0
    if candidate_mask.any(): 
        candidate_positions_flat = torch.where(candidate_mask)[0]
        
        original_token_ids_at_candidates = new_indices_flat[candidate_positions_flat]

        mapped_token_ids_list = []
        for token_id_tensor in original_token_ids_at_candidates:
            token_id = token_id_tensor.item() 
            mapped_token_ids_list.append(hc_instance.index_mapping.get(token_id, token_id))

        mapped_token_ids_tensor = torch.tensor(mapped_token_ids_list,
                                               dtype=new_indices_flat.dtype,
                                               device=device)
        actually_changed_mask_for_candidates = (mapped_token_ids_tensor != original_token_ids_at_candidates)

        final_indices_to_change = candidate_positions_flat[actually_changed_mask_for_candidates]
        final_values_to_set = mapped_token_ids_tensor[actually_changed_mask_for_candidates]

        if final_indices_to_change.numel() > 0:
            new_indices_flat[final_indices_to_change] = final_values_to_set
            replacement_count = final_indices_to_change.numel()

    new_indices_final = new_indices_flat.view(original_shape)

    return new_indices_final, replacement_count


def main(args, hc, vq_model, start_index, filename, prompts=None):
    filename_list = filename
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    print(f"Loading GPT checkpoint from: {args.gpt_ckpt}")
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")

    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        print("Warning: Standard keys ('model', 'module', 'state_dict') not found. Attempting to load entire checkpoint as state_dict.")
        model_weight = checkpoint

    if all(key.startswith('module.') for key in model_weight.keys()):
         model_weight = {k.replace('module.', ''): v for k, v in model_weight.items()}

    try:
        load_result = gpt_model.load_state_dict(model_weight, strict=False)
        print(f"GPT load results: {load_result}")
        if load_result.missing_keys:
             print(f"Warning: Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
             print(f"Warning: Unexpected keys: {load_result.unexpected_keys}")

    except Exception as e:
        print(f"Error loading state dict: {e}")
    gpt_model.eval()
    del checkpoint, model_weight # Free up memory
    print(f"GPT model loaded successfully.")

    if args.compile:
        print(f"Compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) 
        print("Model compiled.")
    else:
        print(f"Model compilation disabled.")

    print(f"Loading T5 model from: {args.t5_path} ({args.t5_model_type})")
    assert os.path.exists(args.t5_path), f"T5 path not found: {args.t5_path}"
    t5_model = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=args.t5_path,
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    print("T5 model loaded.")

    if prompts is None:
        print("Warning: No prompts provided, using default list.")
        prompts = [
           "A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grassin front of the Sydney Opera House holding a sign on the chest that says Welcome Friends!",
           "A photo of an astronaut riding a horse in the forest. There is a river in front of them with water lilies.",
        ]

    save_base_dir = args.save_dir
    os.makedirs(save_base_dir, exist_ok=True)
    print(f"Saving images to base directory: {save_base_dir}")

    total_start_time = time.time()

    for prompt_idx, prompt in enumerate(prompts):
        absolute_index = start_index + prompt_idx
        print(f"\n--- Processing Prompt {prompt_idx+1}/{len(prompts)} (Absolute Index: {absolute_index}): '{prompt[:80]}...' ---")
        prompt_start_time = time.time()

        # 1. Get T5 Embeddings for the current prompt
        current_prompts = [prompt] # Process one prompt at a time
        caption_embs, emb_masks = t5_model.get_text_embeddings(current_prompts)

        if not args.no_left_padding:
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs_list = []
            for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)): # Loop will run once
                valid_num = int(emb_mask.sum().item())
                new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                new_caption_embs_list.append(new_caption_emb)
            new_caption_embs = torch.stack(new_caption_embs_list)
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks

        c_indices = new_caption_embs * new_emb_masks[:,:, None]
        c_emb_masks = new_emb_masks

        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]

        # 2. Generate initial indices and confidence scores using GPT (once per prompt)
        print("  Generating initial indices with GPT...")
        t1 = time.time()
        if hasattr(hc, 'index_mapping'):
            index_map_to_pass = hc.index_mapping
        else:
            index_map_to_pass = None 
            print("Warning: hc object does not have 'index_mapping' attribute.")

        index_sample, con = generate(
            gpt_model, c_indices, latent_size ** 2,
            c_emb_masks,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True,
            index_mapping = index_map_to_pass # Pass the mapping or None
        )
        sampling_time = time.time() - t1
        print(f"  GPT sampling took {sampling_time:.2f} seconds.")

        conf_pairs = con
        original_conf = conf_pairs[:, :, 0].cpu().numpy().flatten()
        paired_conf = conf_pairs[:, :, 1].cpu().numpy().flatten()

        ratio = np.divide(
            original_conf,
            paired_conf,
            out=np.ones_like(original_conf) * np.inf, 
            where=paired_conf != 0
        )
        log_ratio = np.log10(ratio, out=np.full_like(ratio, -np.inf), where=ratio > 0)
        log_ratio_flat = log_ratio.flatten()
        valid_log_ratio = log_ratio_flat[np.isfinite(log_ratio_flat)] 

        # 3. Loop through percentiles for replacement, decoding, and saving
        for k, percentile_val in enumerate(args.percentile):
            title = args.TITLES[k]
            percentile_start_time = time.time()
            print(f"    Processing for percentile {percentile_val}% (Folder: {title}-{args.image_size})...")

            current_save_dir = os.path.join(save_base_dir, f"{title}-{args.image_size}")
            os.makedirs(current_save_dir, exist_ok=True)

            if len(valid_log_ratio) > 0:
                 log_threshold = np.percentile(valid_log_ratio, percentile_val)
            else:
                 log_threshold = -np.inf


            new_indices, count = vectorized_replacement_process(
                                        index_sample,
                                        log_ratio_flat,
                                        log_threshold,
                                        hc)

            t2 = time.time()
            replaced_samples = vq_model.decode_code(new_indices, qzshape)
            decoder_time = time.time() - t2

            im_idx = filename_list[prompt_idx]
            save_path = os.path.join(current_save_dir, f"{im_idx}.png")
            # --- End Modification ---

            save_image(replaced_samples, save_path, nrow=1, normalize=True, value_range=(-1, 1))
            percentile_end_time = time.time()

        prompt_end_time = time.time()
        print(f"--- Prompt {prompt_idx+1} finished in {prompt_end_time - prompt_start_time:.2f} seconds ---")

    total_end_time = time.time()
    print(f"\n=== Total generation finished in {total_end_time - total_start_time:.2f} seconds ===")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Text to image AR generation, using pairing based codebook replacement.")
    # --- Path parameters ---
    parser.add_argument("--t5-path", type=str, default='path/t5-ckpt', help="Path to T5 checkpoint directory.")
    parser.add_argument("--gpt-ckpt", type=str, default="path/t2i_XL_stage2_512.pt", help="Path to GPT checkpoint file.")
    parser.add_argument("--vq-ckpt", type=str, default="path/vq_ds16_t2i.pt", help="VQ model checkpoint path.")
    parser.add_argument("--mapping-save-path", type=str, default="path/codebook_index_mapping.json", help="Save/load the path of the index mapping JSON file.")
    parser.add_argument("--pairs-save-path", type=str, default="path/codebook_pairs.json", help="Path to save/load codebook pairing list JSON file.")
    parser.add_argument("--TITLES", type=str, nargs='+', default=["50%", "60%", "70%", "80%", "90%", "100%"])
    parser.add_argument("--save-dir", type=str, default="path/Gen_Image", help="Base directory to save generated images.")
    # --- Model configuration parameters ---
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl', help="Specific T5 model type (e.g., 'flan-t5-xl', 't5-v1_1-xxl').")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL", help="GPT model architecture.")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="GPT task type: class-to-image or text-to-image.")

    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()) if VQ_models else ["VQ-16"], default="VQ-16") 
    parser.add_argument("--codebook-size", type=int, default=16384, help="The codebook size of VQ")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="The embedding dimension of VQ codebook.")
    # --- Generate/process parameters ---
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512, help="Target image resolution.")
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16, help="VAE downsampling factor.")
    parser.add_argument("--t5-feature-max-len", type=int, default=120, help="Max sequence length for T5 features.")
    parser.add_argument("--cls-token-num", type=int, default=120, help="Max token number for condition input in GPT.")
    parser.add_argument("--cfg-scale", type=float, default=7.5, help="Classifier-Free Guidance scale.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="Top-k sampling parameter.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--percentile", type=int, nargs='+', default=[0, 20, 40, 60, 80, 100], help="List of percentiles for confidence-based replacement.")

    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"], help="Computation precision.")
    parser.add_argument("--compile", action='store_true', default=False, help="Enable torch.compile for the GPT model (requires PyTorch 2.0+).")
    parser.add_argument("--no-left-padding", action='store_true', default=False, help="Disable left-padding for T5 embeddings.")
    parser.add_argument("--replacement-ratio", type=float, default=1.0)
    parser.add_argument("--load-mapping", action='store_true', help="Attempt to load index mapping from file instead of recalculating.")
    parser.add_argument("--load-pairs", action='store_true', help="Attempt to load the codebook pairing list from the file instead of recalculating.")

    args, _ = parser.parse_known_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    hc = None
    codebook_weights = None

    print("Load VQ model...")
    vq_model_class = VQ_models.get(args.vq_model)
    vq_model = vq_model_class(
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)

    should_load_vq_weights = not (args.load_pairs and args.load_mapping and
                                  os.path.exists(args.pairs_save_path) and
                                  os.path.exists(args.mapping_save_path))

    if should_load_vq_weights:
        print(f"Loading VQ model weights from {args.vq_ckpt}...")
        if not os.path.exists(args.vq_ckpt):
            raise FileNotFoundError(f"VQ checkpoint file not found: {args.vq_ckpt}")

        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")

        state_dict = checkpoint.get("model", checkpoint.get("ema", checkpoint))
        if state_dict is None:
            raise KeyError("Cannot find 'model' or 'ema' key in VQ checkpoint.")
        vq_model.load_state_dict(state_dict)
        del checkpoint, state_dict 
        gc.collect()
        print("The VQ model weights have been loaded.")
        vq_model.eval() 
        codebook_weights = vq_model.quantize.embedding.weight.data.cpu().numpy().astype(np.float32)
        print(f"The codebook weights have been extracted (shape:{codebook_weights.shape})")
    else:
        print("Detected attempting to load pairing and mapping from file, skipping loading VQ model weights.")
        codebook_weights = np.zeros((args.codebook_size, args.codebook_embed_dim), dtype=np.float32)
        print(f"Initialize using placeholder codebook weights with the shape: {codebook_weights.shape}.")

    hc = HierarchicalCodebook(
        codebook_vectors=codebook_weights,
        replacement_ratio=args.replacement_ratio,
        mapping_save_path=args.mapping_save_path,
        pairs_save_path=args.pairs_save_path,
        load_mapping=True,
        load_pairs=True
    )

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    ).to(device) 
    vq_model.eval()
    checkpoint_vq = torch.load(args.vq_ckpt, map_location="cpu")

    if "model" in checkpoint_vq:
        vq_state_dict = checkpoint_vq["model"]
    elif "state_dict" in checkpoint_vq:
         vq_state_dict = checkpoint_vq["state_dict"]
    elif "ema" in checkpoint_vq: 
         print("Loading EMA weights for VQ model.")
         vq_state_dict = checkpoint_vq["ema"]
    else:
        print("Warning: Standard keys ('model', 'state_dict', 'ema') not found in VQ checkpoint. Attempting to load entire checkpoint.")
        vq_state_dict = checkpoint_vq

    if all(key.startswith('module.') for key in vq_state_dict.keys()):
         vq_state_dict = {k.replace('module.', ''): v for k, v in vq_state_dict.items()}

    load_result_vq = vq_model.load_state_dict(vq_state_dict, strict=False)
    print(f"VQ load results: {load_result_vq}")
    if load_result_vq.missing_keys:
         print(f"Warning: VQ Missing keys: {load_result_vq.missing_keys}")
    if load_result_vq.unexpected_keys:
         print(f"Warning: VQ Unexpected keys: {load_result_vq.unexpected_keys}")

    del checkpoint_vq, vq_state_dict
    print("VQ model loaded.")
    
    captions_list,ids = mscocojson2list("/root/autodl-tmp/LlamaGen/coco/meta_data.json")
    captions_list = captions_list[:5]
    ids = ids[:5]
    main(args, hc, vq_model, start_index=0,filename = ids, prompts=captions_list)