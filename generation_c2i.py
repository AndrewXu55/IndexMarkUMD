import sys
import torch
import os
import time
import argparse
import numpy as np
import gc
import json 
import traceback 

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from torchvision.utils import save_image
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generatearconc2i import generate 

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false" 


class HierarchicalCodebook:
    def __init__(self, codebook_vectors, replacement_ratio=1.0,
                 mapping_save_path="codebook_index_mapping.json",
                 pairs_save_path="codebook_pairs.json",
                 load_mapping=False, load_pairs=False):
        
        if not isinstance(codebook_vectors, np.ndarray):
            raise TypeError("codebook_vectors must be a NumPy array")
        if codebook_vectors.ndim != 2:
            if not (load_mapping and load_pairs and codebook_vectors.shape[0] > 0):
                 raise ValueError("codebook_vectors must be a 2D array (size, dim) unless loading mapping and pairs.")

        print("Initializing HierarchicalCodebook (Pairing Mode)...")
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
                 max_index_in_pairs = 0
                 if self.pairs:
                      max_index_in_pairs = max(max(p) for p in self.pairs)
                 if self.codebook_size <= max_index_in_pairs or self.codebook_size == 0:
                      print(f"Warning: Adjusting internal codebook size based on loaded pairs to {max_index_in_pairs + 1}")
                      self.codebook_size = max_index_in_pairs + 1
                      self.unassigned_indices = set(range(self.codebook_size))

                 self._reconstruct_state_from_pairs() 


        if load_mapping:
            loaded_mapping = self.load_mapping(self.mapping_save_path)
            if loaded_mapping and not loaded_pairs:
                print("Warning: Mapping loaded, but pair list was not. Red/Green lists populated, but pairing info missing.")
                self.red_list = set(self.index_mapping.keys())
                self.green_list = set(self.index_mapping.values())
                max_index_in_mapping = 0
                if self.index_mapping:
                     all_indices = list(self.index_mapping.keys()) + list(self.index_mapping.values())
                     max_index_in_mapping = max(all_indices) if all_indices else -1
                if self.codebook_size <= max_index_in_mapping or self.codebook_size == 0:
                     print(f"Warning: Adjusting internal codebook size based on loaded mapping to {max_index_in_mapping + 1}")
                     self.codebook_size = max_index_in_mapping + 1
                     self.unassigned_indices = set(range(self.codebook_size)) - self.red_list - self.green_list
        self.red_list_tensor = torch.tensor(list(self.red_list), dtype=torch.long, device=device)
        
        calculation_needed = False
        if loaded_pairs and loaded_mapping:
            print("Successfully loaded pair list and index mapping from files.")
        elif loaded_pairs and not loaded_mapping:
            print("Loaded pair list, but not mapping. Will generate mapping from loaded pairs.")
            self._create_red_green_lists() 
            if self.index_mapping:
                 self.save_mapping(self.mapping_save_path)
            else:
                 print("Warning: Mapping generation from loaded pairs failed or resulted in empty mapping. Not saving.")
        elif not loaded_pairs and loaded_mapping:
            print("Loaded mapping, but not pair list. Will use loaded mapping and skip pair calculation.")
        else:
            if self.codebook_size == 0 or self.feature_dim == 0 or np.all(self.codebook == 0):
                 print("Error: Cannot compute pairs/mapping. Codebook vectors are missing or zero (and loading failed or was not requested).")
                 raise ValueError("Codebook vectors required for calculation but not provided or loaded.")
            else:
                 print("Will compute pair list and index mapping.")
                 calculation_needed = True

        if calculation_needed:
            self._build_pairs()             
            if not self.pairs:
                print("Error: Pair building failed or resulted in an empty pair list. Cannot proceed.")
                raise RuntimeError("Failed to build codebook pairs.")
            self._create_red_green_lists() 
            if self.index_mapping: 
                self.save_mapping(self.mapping_save_path)
            else:
                print("Warning: Mapping generation from computed pairs failed or resulted in empty mapping. Not saving.")

        print("HierarchicalCodebook initialization complete.")
        print(f"  - Codebook size used: {self.codebook_size}")
        print(f"  - Number of pairs: {len(self.pairs)}")
        print(f"  - Mapping size: {len(self.index_mapping)}")
        print(f"  - Unassigned indices: {len(self.unassigned_indices)}")

    def _build_pairs(self):
        print("Calculating similarity matrix...")
        if self.codebook is None or self.codebook.shape[0] == 0:
             raise ValueError("Cannot build pairs: Codebook vectors not available.")
        try:
            norms = np.linalg.norm(self.codebook, axis=1, keepdims=True)
            zero_norm_mask = (norms < 1e-10).flatten()
            if np.any(zero_norm_mask):
                print(f"Warning: Found {np.sum(zero_norm_mask)} zero-norm vectors in the codebook. Similarities involving them will be 0.")
                norms[zero_norm_mask.reshape(-1, 1)] = 1e-10 

            norm_codebook = self.codebook / norms
            sim_matrix = cosine_similarity(norm_codebook)
            del norm_codebook, norms 
            gc.collect()
        except MemoryError:
            print("Error: Out of memory while computing the full similarity matrix.")
            raise
        np.fill_diagonal(sim_matrix, -np.inf)
        sim_matrix[zero_norm_mask, :] = -np.inf
        sim_matrix[:, zero_norm_mask] = -np.inf

        print(f"Pairing {self.codebook_size} codebook vectors...")
        assigned_mask = np.zeros(self.codebook_size, dtype=bool)
        num_pairs_formed = 0
        target_pairs = self.codebook_size // 2
        self.pairs = []

        for i in range(target_pairs):
            valid_entries_mask = np.isfinite(sim_matrix)
            if not np.any(valid_entries_mask):
                print(f"Warning: Stopping pairing at round {i+1}. No more finite similarity values found.")
                break

            max_val = np.max(sim_matrix[valid_entries_mask])
            candidates = np.argwhere(sim_matrix == max_val)
            if len(candidates) == 0:
                 print(f"Warning: Stopping pairing at round {i+1}. Could not find index for max value {max_val}.")
                 break
            idx1, idx2 = candidates[0]

            if sim_matrix[idx1, idx2] <= -np.inf:
                print(f"Warning: Stopping pairing at round {i+1}. Max similarity found is -inf (Shouldn't happen here).")
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
                print(f"  Pairs formed: {num_pairs_formed}/{target_pairs}")

            if num_pairs_formed % 1000 == 0:
                 gc.collect()

        del sim_matrix 
        gc.collect()

        self.unassigned_indices = set(np.where(~assigned_mask)[0])

        print(f"\nPair building completed.")
        print(f"  Total pairs formed: {len(self.pairs)}")
        if self.unassigned_indices:
            print(f"  Note: {len(self.unassigned_indices)} indices remain unassigned (expected if codebook size is odd).")
        else:
             print(f"  All {self.codebook_size} indices successfully assigned to pairs.")

        if hasattr(self, 'pairs_save_path') and self.pairs_save_path:
            self.save_pairs(self.pairs_save_path) 
        else:
            print("Warning: Pair save path (pairs_save_path) not set. Skipping saving the pair list.")

        print("\nAnalyzing similarities within formed pairs...")
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

    def _analyze_pair_similarities_direct(self):
        if not self.pairs:
            print("No pairs available for similarity analysis.")
            return
        if self.codebook is None or np.all(self.codebook == 0):
             print("Codebook vectors not available, skipping similarity analysis.")
             return

        similarities = []
        count = 0
        print("  Starting pair similarity analysis...")
        for idx1, idx2 in self.pairs:
            if idx1 >= self.codebook.shape[0] or idx2 >= self.codebook.shape[0]:
                print(f"Warning: Pair ({idx1}, {idx2}) contains out-of-bounds index for codebook shape {self.codebook.shape}. Skipping similarity calculation for this pair.")
                continue
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
                 print(f"    Analyzed similarities for {count}/{len(self.pairs)} pairs...")

        similarities = np.array(similarities)
        print("  Pair similarity analysis complete.")

        print(f"\n===== Pair Similarity Analysis Results =====")
        if len(similarities) > 0:
            print(f"  Pairs analyzed: {len(similarities)}")
            print(f"  Mean similarity: {np.mean(similarities):.4f}")
            print(f"  Min similarity: {np.min(similarities):.4f}")
            print(f"  Max similarity: {np.max(similarities):.4f}")
            print(f"  Std Dev of similarity: {np.std(similarities):.4f}")
        else:
            print("  No valid similarity values to analyze.")

    def _create_red_green_lists(self):
        self.red_list.clear()
        self.green_list.clear()
        self.index_mapping.clear()

        if not self.pairs:
            print("Warning: Cannot create red/green lists as the pair list is empty.")
            return

        print("Creating red/green lists from pairs...")
        rng_state = np.random.get_state()
        np.random.seed(42)
        for idx1, idx2 in self.pairs:
            if np.random.random() < 0.5:
                red_idx, green_idx = idx1, idx2
            else:
                red_idx, green_idx = idx2, idx1
            self.red_list.add(red_idx)
            self.green_list.add(green_idx)
            self.index_mapping[int(red_idx)] = int(green_idx)
        np.random.set_state(rng_state)

        print(f"\n===== Red/Green List Statistics =====")
        print(f"  Pairs processed: {len(self.pairs)}")
        print(f"  Red list size: {len(self.red_list)}")
        print(f"  Green list size: {len(self.green_list)}")
        expected_total = len(self.pairs) * 2
        actual_total = len(self.red_list) + len(self.green_list)
        if actual_total != expected_total:
             print(f"  Warning: Red/Green list total ({actual_total}) does not match expected ({expected_total}). May be due to duplicate indices in loaded pairs or mapping.")

    def get_random_replacement_org(self, original_idx):
        if original_idx in self.red_list and np.random.random() < self.replacement_ratio:
            return self.index_mapping.get(original_idx, original_idx)
        return original_idx 

    def get_random_replacement(self, original_idx, log_ratio_value, threshold):
        try:
            log_ratio_val = float(log_ratio_value)
            threshold_val = float(threshold)
        except (ValueError, TypeError):
            print(f"Warning: Invalid log_ratio_value ({log_ratio_value}) or threshold ({threshold}). Skipping replacement check.")
            return original_idx

        replace_decision = np.random.random() < self.replacement_ratio

        if (original_idx in self.red_list and
            replace_decision and 
            log_ratio_val <= threshold_val):
            replacement = self.index_mapping.get(original_idx)
            if replacement is not None and replacement != original_idx:
                 return replacement
            else:
                 return original_idx 
        return original_idx 

    def save_mapping(self, filepath):
        print(f"Saving index mapping to {filepath}...")
        if not self.index_mapping:
            print("Warning: Index mapping is empty, nothing to save.")
            return
        try:
            serializable_mapping = {int(k): int(v) for k, v in self.index_mapping.items()}
            os.makedirs(os.path.dirname(filepath), exist_ok=True) 
            with open(filepath, 'w', encoding='utf-8') as f: 
                json.dump(serializable_mapping, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved {len(self.index_mapping)} mapping entries.")
        except TypeError as e:
            print(f"Error: Failed to serialize index mapping. Check data types. Error: {e}")
        except Exception as e:
            print(f"Error: Failed to save index mapping to {filepath}. Error: {e}")

    def load_mapping(self, filepath):
        print(f"Attempting to load index mapping from {filepath}...")
        if not os.path.exists(filepath):
            print(f"Info: Mapping file {filepath} not found.")
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
            print(f"Error: Failed to parse JSON file {filepath}. File might be corrupt. Error: {e}")
            self.index_mapping = {} 
            return False
        except Exception as e:
            print(f"Error: Failed to load index mapping from {filepath}. Error: {e}")
            self.index_mapping = {} 
            return False

    def save_pairs(self, filepath):
        """Saves the self.pairs list to a JSON file."""
        print(f"Saving pair list to {filepath}...")
        if not self.pairs:
            print("Warning: Pair list is empty, nothing to save.")
            return
        try:
            list_of_lists = [list(pair) for pair in self.pairs]
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f: 
                json.dump(list_of_lists, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved {len(self.pairs)} pairs.")
        except TypeError as e:
             print(f"Error: Error serializing pair list. Check data types. Error: {e}")
        except Exception as e:
            print(f"Error: Failed to save pair list to {filepath}. Error: {e}")

    def load_pairs(self, filepath):
        print(f"Attempting to load pair list from {filepath}...")
        if not os.path.exists(filepath):
            print(f"Info: Pair file {filepath} not found.")
            return False
        try:
            with open(filepath, 'r', encoding='utf-8') as f: 
                loaded_list = json.load(f)
            self.pairs = [tuple(map(int, pair)) for pair in loaded_list if isinstance(pair, list) and len(pair) == 2]
            print(f"Successfully loaded {len(self.pairs)} pairs from {filepath}.")
            return True
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON file {filepath}. File might be corrupt. Error: {e}")
            self.pairs = []
            return False
        except Exception as e:
            print(f"Error: Failed to load pair list from {filepath}. Error: {e}")
            self.pairs = []
            return False

    def _reconstruct_state_from_pairs(self):
        if not self.pairs:
            print("Warning: Cannot reconstruct state from empty pair list.")
            return
        print("Reconstructing assignment state from loaded pairs...")
        max_index_in_pairs = 0
        if self.pairs:
            max_index_in_pairs = max(max(p) for p in self.pairs) if self.pairs else -1
        if self.codebook_size <= max_index_in_pairs or self.codebook_size == 0:
            print(f"Note: Adjusting internal codebook size to {max_index_in_pairs + 1} based on loaded pairs for state reconstruction.")
            self.codebook_size = max_index_in_pairs + 1

        assigned_mask = np.zeros(self.codebook_size, dtype=bool)
        valid_pairs_count = 0
        indices_seen = set()
        duplicates_found = False
        for idx1, idx2 in self.pairs:
            if 0 <= idx1 < self.codebook_size and 0 <= idx2 < self.codebook_size:
                if idx1 in indices_seen or idx2 in indices_seen:
                     duplicates_found = True
                else:
                    assigned_mask[idx1] = True
                    assigned_mask[idx2] = True
                    indices_seen.add(idx1)
                    indices_seen.add(idx2)
                valid_pairs_count += 1 
            else:
                print(f"Warning: Loaded pair {(idx1, idx2)} contains out-of-bounds index for codebook size {self.codebook_size}. Skipping this pair.")
        if duplicates_found:
            print("Warning: Duplicate indices found across different pairs in the loaded file. This might affect red/green list generation.")
        self.unassigned_indices = set(range(self.codebook_size)) - indices_seen
        print(f"State reconstruction complete. Valid pairs processed: {valid_pairs_count}. Indices assigned: {len(indices_seen)}. Unassigned indices: {len(self.unassigned_indices)}")


def vectorized_replacement_process(index_sample, log_ratio_flat, log_threshold, hc_instance):
    device = index_sample.device
    new_indices = index_sample.clone()

    if hc_instance.red_list_tensor.device != device:
        hc_instance.red_list_tensor = hc_instance.red_list_tensor.to(device)

    original_shape = new_indices.shape
    new_indices_flat = new_indices.view(-1)
    if log_ratio_flat.shape[0] != new_indices_flat.shape[0]:
        raise ValueError(f"log_ratio_flat ({log_ratio_flat.shape}) vs new_indices_flat ({new_indices_flat.shape})")

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


def main(args, hc, vq_model_main, gpt_model): 
    torch.set_grad_enabled(False)
    device = next(gpt_model.parameters()).device 
    print(f"Using device: {device}")

    total_classes = args.num_classes 
    all_class_labels = list(range(total_classes))
    num_seeds_per_class = args.num_seeds_per_class
    batch_size = args.batch_size
    base_seed = args.seed 

    latent_size = args.image_size // args.downsample_size
    save_base_dir = args.save_dir 
    os.makedirs(save_base_dir, exist_ok=True)
    print(f"Saving images to base directory: {save_base_dir}")
    print(f"Generating {num_seeds_per_class} samples for each of {total_classes} classes.")
    print(f"Processing in batches of size {batch_size}.")

    total_images_generated = 0
    overall_start_time = time.time()

    for i in range(0, total_classes, batch_size):
        batch_start_time = time.time()
        current_batch_labels = all_class_labels[i:min(i + batch_size, total_classes)]
        current_batch_size = len(current_batch_labels)
        if current_batch_size == 0:
            continue

        print(f"\n--- Processing Batch {i // batch_size + 1}/{(total_classes + batch_size - 1) // batch_size} (Classes {current_batch_labels[0]} to {current_batch_labels[-1]}) ---")
        c_indices = torch.tensor(current_batch_labels, device=device)

        for seed_idx in range(num_seeds_per_class):
            seed_start_time = time.time()
            current_seed = base_seed + (i * num_seeds_per_class) + seed_idx 
            torch.manual_seed(current_seed)
            np.random.seed(current_seed) 
            print(f"  Seed {seed_idx + 1}/{num_seeds_per_class} (Global Seed: {current_seed})")

            # 1. Generate initial indices and confidence scores using GPT (once per batch/seed)
            t1 = time.time()
            index_sample, con = generate(
                    gpt_model,
                    c_indices, # Pass current batch class indices
                    latent_size ** 2,
                    cfg_scale=args.cfg_scale,
                    cfg_interval=args.cfg_interval,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    sample_logits=True,
                    index_mapping=hc.index_mapping if hc and hasattr(hc, 'index_mapping') else None # Pass HC mapping
            )
            sampling_time = time.time() - t1

            conf_pairs = con 
            original_conf = conf_pairs[:, :, 0].cpu().numpy()
            paired_conf = conf_pairs[:, :, 1].cpu().numpy()  

            epsilon = 1e-9
            ratio = np.divide(
                original_conf,
                paired_conf + epsilon
            )
            log_ratio = np.log10(np.maximum(ratio, epsilon))
            log_ratio_flat = log_ratio.flatten()
            valid_log_ratio = log_ratio_flat[np.isfinite(log_ratio_flat)] 

            # 2. Loop through percentiles for replacement, decoding, and saving
            for k, percentile_val in enumerate(args.percentile):
                title = args.TITLES[k] if k < len(args.TITLES) else f"P{percentile_val}"
                percentile_start_time = time.time()

                current_save_dir = os.path.join(save_base_dir, title)
                os.makedirs(current_save_dir, exist_ok=True)

                if len(valid_log_ratio) > 0:
                     log_threshold = np.percentile(valid_log_ratio, percentile_val)
                else:
                     log_threshold = -np.inf 
                new_indices = index_sample.clone()
                new_indices, count = vectorized_replacement_process(
                                        index_sample,
                                        log_ratio_flat,
                                        log_threshold,
                                        hc)

                t2 = time.time()
                qzshape = [current_batch_size, args.codebook_embed_dim, latent_size, latent_size]
                replaced_samples = vq_model_main.decode_code(new_indices, qzshape)
                decoder_time = time.time() - t2

                for img_idx_in_batch in range(current_batch_size):
                    actual_class_label = current_batch_labels[img_idx_in_batch]
                    filename = f"class{actual_class_label:04d}_seed{seed_idx:02d}_{title}.png"
                    save_path = os.path.join(current_save_dir, filename)
                    save_image(replaced_samples[img_idx_in_batch], save_path, normalize=True, value_range=(-1, 1))
                    total_images_generated += 1

            seed_end_time = time.time()
            print(f"    Seed {seed_idx + 1} processed in {seed_end_time - seed_start_time:.2f}s.")
            gc.collect()

        batch_end_time = time.time()
        print(f"--- Batch {i // batch_size + 1} finished in {batch_end_time - batch_start_time:.2f}s ---")
        gc.collect()

    overall_end_time = time.time()
    print(f"\n=== Total generation finished ===")
    print(f"Generated {total_images_generated} images in total.")
    print(f"Total time: {overall_end_time - overall_start_time:.2f} seconds ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-to-Image AR generation with HC-based codebook replacement (Multi-Seed)")

    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-L", help="GPT model architecture.")
    parser.add_argument("--gpt-ckpt", type=str, default="path/c2i_L_256.pt", help="Path to C2I GPT checkpoint file.")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="GPT task type (should be c2i).")
    parser.add_argument("--from-fsdp", action='store_true', help="Load checkpoint saved with FSDP.")
    parser.add_argument("--cls-token-num", type=int, default=1, help="Max token number of condition input (typically 1 for class).")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes for Imagenet (or total to generate).")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256, help="Target image resolution.")
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16, help="VAE downsampling factor.")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-Free Guidance scale for C2I.")
    parser.add_argument("--cfg-interval", type=float, default=10, help="CFG interval frequency.") 
    parser.add_argument("--TITLES", type=str, nargs='+', default=["50%", "60%", "70%", "80%", "90%", "100%"])

    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()) if VQ_models else ["VQ-16"], default="VQ-16", help="VQ model architecture.")
    parser.add_argument("--vq-ckpt", type=str, default="path/vq_ds16_c2i.pt", help="Path to C2I VQ model checkpoint.")
    parser.add_argument("--codebook-size", type=int, default=16384, help="VQ codebook size.")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="VQ codebook embedding dimension.")

    parser.add_argument("--seed", type=int, default=0, help="Base random seed for reproducibility.")
    parser.add_argument("--top-k", type=int, default=2000, help="Top-k sampling parameter.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter.")

    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"], help="Computation precision.")
    parser.add_argument("--compile", action='store_true', default=False, help="Enable torch.compile for the GPT model (requires PyTorch 2.0+).")

    parser.add_argument("--percentile", type=int, nargs='+', default=[0, 20, 40, 60, 80, 100], help="List of percentiles for confidence-based replacement.")
    parser.add_argument("--save-dir", type=str, default="path/Gen_Image_c2i", help="Base directory to save generated images.")
    parser.add_argument("--replacement-ratio", type=float, default=1.0, help="Proportion of red-list tokens eligible for replacement (0.0-1.0).")
    parser.add_argument("--mapping-save-path", type=str, default="path/codebook_index_mapping_knn10_mwpm_c2i.json", help="Path to load index mapping JSON file for C2I VQ.")
    parser.add_argument("--pairs-save-path", type=str, default="path/codebook_pairs_knn10_mwpm_c2i.json", help="Path to load codebook pair list JSON file for C2I VQ.")
    parser.add_argument("--load-mapping", action='store_true', default=True, help="Attempt to load index mapping from file instead of recomputing.")
    parser.add_argument("--load-pairs", action='store_true', default=True, help="Attempt to load codebook pair list from file instead of recomputing.")
    parser.add_argument("--no-load-mapping", action="store_false", dest="load_mapping", help="Disable loading index mapping.") 
    parser.add_argument("--no-load-pairs", action="store_false", dest="load_pairs", help="Disable loading codebook pairs.") 
    parser.add_argument("--num-seeds-per-class", type=int, default=10, help="Number of different images (seeds) to generate per class.")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of classes to process in parallel per generation step.")

    args = parser.parse_args() 

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False) 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Primary device: {device}")
    if device == "cuda":
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")

    hc = None 
    codebook_weights = None

    needs_hc_calculation = False
    if args.load_pairs and args.load_mapping:
        if not os.path.exists(args.pairs_save_path):
            print(f"Info: Pair file {args.pairs_save_path} not found, will need calculation if mapping load also fails or is disabled.")
            needs_hc_calculation = True # Need calculation if file is missing
        if not os.path.exists(args.mapping_save_path):
             print(f"Info: Mapping file {args.mapping_save_path} not found, will need calculation if pair load also fails or is disabled.")
             needs_hc_calculation = True # Need calculation if file is missing
        if os.path.exists(args.pairs_save_path) and os.path.exists(args.mapping_save_path):
             print("Found existing pair and mapping files. Will attempt to load.")
             needs_hc_calculation = False
        else:
             needs_hc_calculation = True 

    elif args.load_pairs: 
         if not os.path.exists(args.pairs_save_path):
             print(f"Error: Pair file {args.pairs_save_path} not found, and mapping loading not requested. Calculation needed but codebook vectors may be missing.")
             needs_hc_calculation = True 
         else:
             print("Found pair file, will load pairs and calculate mapping.")
             needs_hc_calculation = False 

    elif args.load_mapping:
         if not os.path.exists(args.mapping_save_path):
             print(f"Error: Mapping file {args.mapping_save_path} not found, and pair loading not requested. Calculation needed.")
             needs_hc_calculation = True # Need calculation
         else:
              print("Found mapping file, will load mapping and skip pair calculation.")
              needs_hc_calculation = False # Don't need weights or pair calc

    else:
        print("Loading of pairs and mapping disabled. Calculation is required.")
        needs_hc_calculation = True


    if needs_hc_calculation:
        print("Loading VQ model weights temporarily to extract codebook for HC calculation...")
        try:
            temp_vq_model = VQ_models[args.vq_model](
                codebook_size=args.codebook_size,
                codebook_embed_dim=args.codebook_embed_dim
            )
            if not os.path.exists(args.vq_ckpt):
                raise FileNotFoundError(f"C2I VQ checkpoint file not found for HC calculation: {args.vq_ckpt}")

            checkpoint_vq_temp = torch.load(args.vq_ckpt, map_location="cpu")
            if "model" in checkpoint_vq_temp: state_dict_vq = checkpoint_vq_temp["model"]
            elif "state_dict" in checkpoint_vq_temp: state_dict_vq = checkpoint_vq_temp["state_dict"]
            elif "ema" in checkpoint_vq_temp: state_dict_vq = checkpoint_vq_temp["ema"] # Added ema key
            else: state_dict_vq = checkpoint_vq_temp

            if all(key.startswith('module.') for key in state_dict_vq.keys()):
                state_dict_vq = {k.replace('module.', ''): v for k, v in state_dict_vq.items()}

            if any(key.startswith('vq.') for key in state_dict_vq.keys()):
                 state_dict_vq = {k.replace('vq.', ''): v for k, v in state_dict_vq.items() if k.startswith('vq.')}
                 if 'quantize.embedding.weight' not in state_dict_vq:
                      if 'embedding.weight' in state_dict_vq:
                           state_dict_vq['quantize.embedding.weight'] = state_dict_vq.pop('embedding.weight')
                      else:
                           raise KeyError("Could not find codebook embedding weights ('quantize.embedding.weight' or 'embedding.weight') after stripping 'vq.' prefix.")

            load_res = temp_vq_model.load_state_dict(state_dict_vq, strict=False)
            print(f"Temporary VQ load result for HC: {load_res}")
            if 'quantize.embedding.weight' in load_res.missing_keys and 'embedding.weight' not in state_dict_vq:
                original_keys = checkpoint_vq_temp.get("model", checkpoint_vq_temp.get("state_dict", checkpoint_vq_temp.get("ema", checkpoint_vq_temp))).keys()
                print("Available top-level keys in VQ checkpoint:", checkpoint_vq_temp.keys())
                print("Available keys in VQ state_dict:", original_keys)

                raise KeyError(f"Could not find codebook embedding weights key 'quantize.embedding.weight' or similar in VQ checkpoint {args.vq_ckpt}. Checkpoint structure might be unexpected.")
                
            if hasattr(temp_vq_model, 'quantize') and hasattr(temp_vq_model.quantize, 'embedding'):
                codebook_weights = temp_vq_model.quantize.embedding.weight.data.cpu().numpy().astype(np.float32)
            elif hasattr(temp_vq_model, 'embedding'): # Simpler VQ structure?
                 codebook_weights = temp_vq_model.embedding.weight.data.cpu().numpy().astype(np.float32)
            else:
                 raise AttributeError("Could not find embedding layer in the temporary VQ model structure.")

            print(f"Codebook weights extracted (shape: {codebook_weights.shape})")
            if codebook_weights.shape != (args.codebook_size, args.codebook_embed_dim):
                print(f"Warning: Extracted codebook shape {codebook_weights.shape} does not match arguments ({args.codebook_size}, {args.codebook_embed_dim}). Using extracted shape.")
                args.codebook_size = codebook_weights.shape[0] # Update args to reflect reality
                print(f"Warning: Make sure --codebook-size ({args.codebook_size}) matches the GPT model's vocab_size.")


            del temp_vq_model, checkpoint_vq_temp, state_dict_vq # Free memory
            gc.collect()
        except Exception as e:
            print(f"ERROR: Failed to load VQ weights for HC calculation: {e}")
            traceback.print_exc()
            print("Cannot proceed without codebook vectors for HC calculation.")
            sys.exit(1)
    else:
        print("Attempting to load HC pairs/mapping from files, skipping VQ weight extraction.")
        codebook_weights = np.zeros((args.codebook_size, args.codebook_embed_dim), dtype=np.float32)
        print(f"Using placeholder codebook weights (shape: {codebook_weights.shape}) for HC initialization.")

    try:
        hc = HierarchicalCodebook(
            codebook_vectors=codebook_weights, 
            replacement_ratio=args.replacement_ratio,
            mapping_save_path=args.mapping_save_path, 
            pairs_save_path=args.pairs_save_path,    
            load_mapping=args.load_mapping,           
            load_pairs=args.load_pairs              
        )
        if args.load_mapping and (not hc.index_mapping or len(hc.index_mapping) == 0):
             print("WARNING: Loading mapping was requested, but the index mapping is empty after initialization.")
             if not os.path.exists(args.mapping_save_path):
                  print(f"         Reason: Mapping file '{args.mapping_save_path}' not found.")
             else:
                  print(f"         Reason: Mapping file '{args.mapping_save_path}' might be empty, corrupt, or failed to load.")
             if not needs_hc_calculation and not args.load_pairs: # No fallback possible
                  print("         Cannot proceed without a valid mapping.")
                  sys.exit(1)
             elif needs_hc_calculation:
                  print("         Proceeding with calculation based on extracted weights.")
             elif args.load_pairs:
                  print("         Will attempt to derive mapping from loaded pairs.")

    except Exception as e:
        print(f"FATAL: Failed to initialize HierarchicalCodebook: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Load Main VQ Model (for decoding) ---
    print("\nLoading main VQ model for decoding...")
    vq_model_main = VQ_models[args.vq_model](
        codebook_size=args.codebook_size, # Use potentially updated codebook size
        codebook_embed_dim=args.codebook_embed_dim
    ).to(device)
    vq_model_main.eval()

    if not os.path.exists(args.vq_ckpt):
        raise FileNotFoundError(f"Main C2I VQ checkpoint file not found: {args.vq_ckpt}")
    checkpoint_vq_main = torch.load(args.vq_ckpt, map_location="cpu")

    if "model" in checkpoint_vq_main: vq_state_dict_main = checkpoint_vq_main["model"]
    elif "state_dict" in checkpoint_vq_main: vq_state_dict_main = checkpoint_vq_main["state_dict"]
    elif "ema" in checkpoint_vq_main: vq_state_dict_main = checkpoint_vq_main["ema"]
    else: vq_state_dict_main = checkpoint_vq_main

    if all(key.startswith('module.') for key in vq_state_dict_main.keys()):
         vq_state_dict_main = {k.replace('module.', ''): v for k, v in vq_state_dict_main.items()}

    if any(key.startswith('vq.') for key in vq_state_dict_main.keys()):
         vq_state_dict_main = {k.replace('vq.', ''): v for k, v in vq_state_dict_main.items() if k.startswith('vq.')}

    load_result_vq_main = vq_model_main.load_state_dict(vq_state_dict_main, strict=False)
    print(f"Main VQ load results: {load_result_vq_main}")
    if load_result_vq_main.missing_keys:
         print(f"Warning: Main VQ Missing keys: {load_result_vq_main.missing_keys}")
    if load_result_vq_main.unexpected_keys:
         print(f"Warning: Main VQ Unexpected keys: {load_result_vq_main.unexpected_keys}")

    del checkpoint_vq_main, vq_state_dict_main # Free memory
    gc.collect()
    print("Main VQ model loaded and ready on device.")

    print("\nLoading main GPT model for generation...")
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes, 
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type, 
    ).to(device=device, dtype=precision)

    print(f"Loading GPT checkpoint from: {args.gpt_ckpt}")
    if not os.path.exists(args.gpt_ckpt):
         raise FileNotFoundError(f"GPT checkpoint not found: {args.gpt_ckpt}")
    checkpoint_gpt = torch.load(args.gpt_ckpt, map_location="cpu")

    if args.from_fsdp: model_weight = checkpoint_gpt
    elif "model" in checkpoint_gpt: model_weight = checkpoint_gpt["model"]
    elif "module" in checkpoint_gpt: model_weight = checkpoint_gpt["module"]
    elif "state_dict" in checkpoint_gpt: model_weight = checkpoint_gpt["state_dict"]
    else: model_weight = checkpoint_gpt

    if all(key.startswith('module.') for key in model_weight.keys()):
         model_weight = {k.replace('module.', ''): v for k, v in model_weight.items()}

    try:
        load_result_gpt = gpt_model.load_state_dict(model_weight, strict=False)
        print(f"GPT load results: {load_result_gpt}")
        mismatched_pos_emb = False
        if 'pos_embed' in load_result_gpt.missing_keys or 'pos_embed' in load_result_gpt.unexpected_keys:
             print("Warning: Positional embedding mismatch detected. This might happen if image size/downsampling changed.")
             mismatched_pos_emb = True # Flag it
        if load_result_gpt.missing_keys and not mismatched_pos_emb: # Report other missing keys
             print(f"Warning: GPT Missing keys: {load_result_gpt.missing_keys}")
        if load_result_gpt.unexpected_keys and not mismatched_pos_emb: # Report other unexpected keys
             print(f"Warning: GPT Unexpected keys: {load_result_gpt.unexpected_keys}")

    except Exception as e:
        print(f"Error loading GPT state dict: {e}")
        traceback.print_exc()
        raise e

    gpt_model.eval()
    del checkpoint_gpt, model_weight # Free up memory
    gc.collect()
    print(f"GPT model loaded successfully.")

    if args.compile:
        print(f"Compiling the GPT model...")
        try:
            gpt_model = torch.compile(
                gpt_model,
                mode="reduce-overhead",
                fullgraph=True
            )
            print("GPT model compiled.")
        except Exception as e:
            print(f"Warning: GPT Model compilation failed: {e}. Running without compilation.")
    else:
        print(f"GPT model compilation disabled.")

    print("\nStarting main generation process...")
    main(args, hc, vq_model_main, gpt_model)

    print("\nScript finished.")