import argparse
import gc
import json
import os
import sys
import time
import traceback
from typing import Union
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
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
torch.set_float32_matmul_precision("high")

setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


class HierarchicalCodebook:
    def __init__(
        self,
        codebook_vectors,
        replacement_ratio=1.0,
        mapping_save_path="codebook_index_mapping.json",
        pairs_save_path="codebook_pairs.json",
        load_mapping=False,
        load_pairs=False,
        device="cuda",
    ):

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
                print(
                    "Warning: The mapping has been loaded, but the pairing list has not been loaded."
                )
                self.red_list = set(self.index_mapping.keys())
                self.green_list = set(self.index_mapping.values())

        calculation_needed = True
        if loaded_pairs and loaded_mapping:
            print(
                "Successfully loaded the pairing list and index mapping from the file."
            )
            calculation_needed = False
        elif loaded_pairs and not loaded_mapping:
            print(
                "The pairing list has been loaded, but the mapping has not been loaded. Mapping will be generated based on the loaded pairing."
            )
            self._create_red_green_lists()
            self.save_mapping(self.mapping_save_path)
            calculation_needed = False
        elif not loaded_pairs and loaded_mapping:
            print(
                "The mapping has been loaded, but the pairing list has not been loaded. The loaded mapping will be used to skip pairing calculations."
            )
            calculation_needed = False
        else:
            print("Calculate the pairing list and index mapping.")

        if calculation_needed:
            self._build_pairs()
            if not self.pairs:
                print(
                    "Error: Pairing build failed or resulted in an empty pairing list. Cannot continue."
                )
                raise RuntimeError("Failed to build codebook pairing.")
            self._create_red_green_lists()
            self.save_mapping(self.mapping_save_path)

        # Create red_list_tensor AFTER red_list is populated
        self.red_list_tensor = torch.tensor(
            list(self.red_list), dtype=torch.long, device=device
        )
        self.green_list_tensor = torch.tensor(
            list(self.green_list), dtype=torch.long, device=device
        )

        print("HierarchicalCodebook initialization completed.")
        print(f"  - Number of pairings: {len(self.pairs)}")
        print(f"  - Mapping size: {len(self.index_mapping)}")
        print(f"  - Red list size: {len(self.red_list)}")
        print(f"  - Green list size: {len(self.green_list)}")
        print(f"  - Unallocated index number: {len(self.unassigned_indices)}")

    def _build_pairs(self, device="cuda"):
        print("Calculating similarity matrix ..")
        codebook_tensor = torch.tensor(
            self.codebook, dtype=torch.float32, device=device
        )

        norms = torch.norm(codebook_tensor, dim=1, keepdim=True)
        zero_norm_mask = (norms < 1e-10).squeeze()
        if zero_norm_mask.any():
            print(
                f"Warning: {zero_norm_mask.sum().item()} zero norm vectors were found in the codebook. The similarity involving them will be 0."
            )
            norms[zero_norm_mask] = 1e-10

        norm_codebook = codebook_tensor / norms

        sim_matrix = norm_codebook @ norm_codebook.T

        sim_matrix.fill_diagonal_(-float("inf"))
        sim_matrix[zero_norm_mask, :] = -float("inf")
        sim_matrix[:, zero_norm_mask] = -float("inf")

        del norm_codebook, norms, codebook_tensor
        gc.collect()

        print(f"Pairing {self.codebook_size} codebook vectors ...")
        assigned_mask = torch.zeros(self.codebook_size, dtype=torch.bool, device=device)
        num_pairs_formed = 0
        target_pairs = self.codebook_size // 2
        self.pairs = []

        for i in tqdm(range(target_pairs)):

            flat_idx = torch.argmax(sim_matrix).item()
            idx1, idx2 = divmod(flat_idx, sim_matrix.size(1))

            if sim_matrix[idx1, idx2] <= -float("inf"):
                print(
                    f"Warning: Stop pairing in round {i+1}. No more valid pairs were found (maximum similarity is -inf)."
                )
                break

            self.pairs.append(tuple(sorted((int(idx1), int(idx2)))))
            assigned_mask[idx1] = True
            assigned_mask[idx2] = True
            num_pairs_formed += 1

            sim_matrix[idx1, :] = -float("inf")
            sim_matrix[:, idx1] = -float("inf")
            sim_matrix[idx2, :] = -float("inf")
            sim_matrix[:, idx2] = -float("inf")

            if (num_pairs_formed % 500 == 0) or (num_pairs_formed == target_pairs):
                print(f"  Pairing has been formed: {num_pairs_formed}/{target_pairs}")
            if num_pairs_formed % 1000 == 0:
                gc.collect()

        del sim_matrix
        gc.collect()

        self.unassigned_indices = set(
            torch.nonzero(~assigned_mask, as_tuple=True)[0].cpu().numpy()
        )

        print(f"\nPairing construction completed.")
        print(f"  Total number of pairs formed: {len(self.pairs)}")
        if self.unassigned_indices:
            print(
                f"  Note: There are {len(self.unassigned_indices)} indexes that have not been assigned."
            )
        else:
            print(
                f"  All {self.codebook_size} indexes have been successfully assigned to the pairing."
            )

        if hasattr(self, "pairs_save_path") and self.pairs_save_path:
            self.save_pairs(self.pairs_save_path)
        else:
            print(
                "Warning: The pairing save path (pairs_save_path) has not been set. Skip saving the pairing list."
            )

        print("\nAnalyzing the similarity within the formed pairs ...")
        self._analyze_pair_similarities_direct()

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
                print(
                    f"    Analyzed the similarity between {count}/{len(self.pairs)} pairs..."
                )

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
            print(
                "Warning: Unable to create red/green list because the pairing list is empty."
            )
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
            print(
                f"  Warning: The total number of red/green lists ({actual_total}) does not match the expected ({expected_total})."
            )

    def get_random_replacement(self, original_idx, log_ratio_value, threshold):
        """Replacement logic: Consider confidence level."""
        try:
            log_ratio_val = float(log_ratio_value)
            threshold_val = float(threshold)
        except (ValueError, TypeError):
            print(
                f"Warning: Invalid log_ratio-value ({log_ratio_value}) or threshold ({threshold}). Skip replacement check."
            )
            return original_idx

        if (
            original_idx in self.red_list
            and np.random.random() < self.replacement_ratio
            and log_ratio_val <= threshold_val
        ):
            return self.index_mapping.get(original_idx, original_idx)
        return original_idx

    def save_mapping(self, filepath):
        """Save the index_mapping dictionary to a JSON file."""
        print(f"Saving index mapping to {filepath}...")
        if not self.index_mapping:
            print("Warning: Index mapping is empty, no need to save.")
            return
        try:
            serializable_mapping = {
                int(k): int(v) for k, v in self.index_mapping.items()
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_mapping, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved {len(self.index_mapping)} mapping entries.")
        except TypeError as e:
            print(
                f"Error: The serialization index mapping failed. Please check the data type. Error: {e}"
            )
        except Exception as e:
            print(f"Error: {e}")

    def load_mapping(self, filepath):
        """Load the index_mapping dictionary from the JSON file."""
        print(f"Attempting to load index mapping from {filepath}...")
        if not os.path.exists(filepath):
            print(f"Information: Mapping file {filepath} not found.")
            return False
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded_mapping_str_keys = json.load(f)
            self.index_mapping = {
                int(k): int(v) for k, v in loaded_mapping_str_keys.items()
            }
            self.red_list = set(self.index_mapping.keys())
            self.green_list = set(self.index_mapping.values())
            print(
                f"Successfully loaded {len(self.index_mapping)} mapping entries from {filepath}."
            )
            return True
        except json.JSONDecodeError as e:
            print(
                f"Error: Failed to parse JSON file {filepath}. The file may be corrupted. Error: {e}"
            )
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
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(list_of_lists, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved {len(self.pairs)} pairs.")
        except TypeError as e:
            print(
                f"Error: Failed to serialize the pairs list. Please check the data types. Error: {e}"
            )
        except Exception as e:
            print(f"Error: Failed to save the pairs list to {filepath}. Error: {e}")

    def load_pairs(self, filepath):
        """Load the self.pairs list from a JSON file."""
        print(f"Attempting to load the pairs list from {filepath}...")
        if not os.path.exists(filepath):
            print(f"Info: Pairs file {filepath} not found.")
            return False
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded_list = json.load(f)
            self.pairs = [
                tuple(map(int, pair))
                for pair in loaded_list
                if isinstance(pair, list) and len(pair) == 2
            ]
            print(f"Successfully loaded {len(self.pairs)} pairs from {filepath}.")
            return True
        except json.JSONDecodeError as e:
            print(
                f"Error: Failed to parse JSON file {filepath}. The file may be corrupted. Error: {e}"
            )
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
                print(
                    f"Warning: The loaded pair {(idx1, idx2)} contains an index that is out of bounds for the codebook size {self.codebook_size}. This pairing has been skipped."
                )
        self.unassigned_indices = set(np.where(~assigned_mask)[0])
        print(
            f"State reconstruction completed. Effective number of pairs used: {valid_pairs_count}. Unallocated index number: {len(self.unassigned_indices)}"
        )


def load_prompts_from_csv(filename):
    """Loads prompts from a single-column CSV file."""
    prompts_list = []
    if not os.path.exists(filename):
        print(f"Error: Prompt file '{filename}' not found.")
        return None

    print(f"Loading prompts from '{filename}'...")
    try:
        with open(filename, mode="r", newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            for row_number, row in enumerate(reader):
                if row:
                    prompt_text = row[0].strip()
                    if prompt_text:
                        prompts_list.append(prompt_text)
                    else:
                        print(
                            f"Warning: Skipping empty prompt at row {row_number + 1}."
                        )
                else:
                    print(f"Warning: Skipping empty row {row_number + 1}.")
        print(f"Loaded {len(prompts_list)} prompts.")
        return prompts_list
    except Exception as e:
        print(f"Error reading prompts from '{filename}': {e}")
        return None


def mscocojson2list(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    captions = []
    ids = []
    for i in range(len(data)):
        captions.append(data[i]["caption"])
        ids.append(data[i]["image_id"])
    return captions, ids


def clustering_based_assignment(codebook_vectors, n_clusters, seed=42, device="cuda"):
    """
    Perform K-means clustering on codebook vectors and assign colors.

    Args:
        codebook_vectors: Codebook weight matrix (codebook_size x embedding_dim)
        n_clusters: Number of clusters for K-means
        seed: Random seed for reproducibility
        device: Device to create tensors on (default: 'cuda')

    Returns:
        cluster_to_color: Dictionary mapping cluster_id -> 'red' or 'green'
        token_to_cluster: Array mapping token_id -> cluster_id
        red_tokens: Tensor of token indices assigned to red
        green_tokens: Tensor of token indices assigned to green
    """
    print(f"Running K-means clustering with {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(codebook_vectors)

    np.random.seed(seed)
    cluster_ids = np.arange(n_clusters)
    np.random.shuffle(cluster_ids)

    n_red_clusters = n_clusters // 2
    red_clusters = set(cluster_ids[:n_red_clusters])
    green_clusters = set(cluster_ids[n_red_clusters:])

    cluster_to_color = {}
    for cluster_id in range(n_clusters):
        if cluster_id in red_clusters:
            cluster_to_color[cluster_id] = "red"
        else:
            cluster_to_color[cluster_id] = "green"

    red_tokens_list = []
    green_tokens_list = []

    for token_id, cluster_id in enumerate(cluster_labels):
        if cluster_to_color[cluster_id] == "red":
            red_tokens_list.append(token_id)
        else:
            green_tokens_list.append(token_id)

    # Convert to tensors for efficiency
    red_tokens = torch.tensor(red_tokens_list, dtype=torch.long, device=device)
    green_tokens = torch.tensor(green_tokens_list, dtype=torch.long, device=device)

    print(f"Clustering complete:")
    print(f"  Red clusters: {len(red_clusters)}, Green clusters: {len(green_clusters)}")
    print(f"  Red tokens: {len(red_tokens)}, Green tokens: {len(green_tokens)}")
    print(f"  Red ratio: {len(red_tokens) / (len(red_tokens) + len(green_tokens)):.2%}")

    return cluster_to_color, cluster_labels, red_tokens, green_tokens


def compute_spectral_graph_laplacian(
    codebook_vectors, sigma=1.0, k=100, epsilon=1e-12, device="cuda"
):
    """
    Compute the normalized graph Laplacian from codebook vectors using KNN.

    Args:
        codebook_vectors: Codebook weight matrix (N x D), numpy array
        sigma: Scaling parameter for Gaussian kernel
        k: Number of nearest neighbors for KNN
        epsilon: Small number for numerical stability
        device: Device to run computation on

    Returns:
        L_sym: Symmetric normalized Laplacian (torch tensor on device)
        N: Number of nodes
    """
    N = codebook_vectors.shape[0]
    C = torch.from_numpy(codebook_vectors).float().to(device)

    # Compute pairwise squared distances
    with torch.no_grad():
        norms_sq = torch.sum(C**2, dim=1, keepdim=True)
        distances_sq = norms_sq + norms_sq.T - 2 * torch.mm(C, C.T)
        distances_sq = torch.clamp(distances_sq, min=0)

        # For each row, keep only k nearest neighbors
        knn_mask = torch.zeros_like(distances_sq, dtype=torch.bool)
        topk = torch.topk(
            distances_sq, k=k + 1, largest=False
        )  # +1 because self-distance is zero
        knn_mask.scatter_(1, topk.indices, True)

        # Symmetrize the mask (i connected to j if either i->j or j->i)
        knn_mask = knn_mask | knn_mask.T

        # Apply RBF only to k-NN entries
        W = torch.zeros_like(distances_sq)
        W[knn_mask] = torch.exp(-distances_sq[knn_mask] / (2 * sigma**2))
        W.fill_diagonal_(0)
        del distances_sq, knn_mask

    # Normalized Laplacian: L_sym = I - D^(-1/2) W D^(-1/2)
    d = torch.sum(W, dim=1)
    d_inv_sqrt = torch.pow(d + epsilon, -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)

    L_sym = torch.eye(N, device=device) - torch.mm(D_inv_sqrt, torch.mm(W, D_inv_sqrt))
    return L_sym, N


def assign_families_to_colors(
    cluster_labels, n_clusters, seed, gamma=0.5, device="cuda"
):
    """
    Assign cluster/family labels to red/green colors randomly.

    Args:
        cluster_labels: Array of cluster assignments for each token
        n_clusters: Number of clusters/families
        seed: Random seed for reproducibility
        gamma: Target fraction of tokens to be green (default: 0.5)
        device: Device to create tensors on (default: 'cuda')

    Returns:
        red_tokens: Tensor of token indices assigned to red
        green_tokens: Tensor of token indices assigned to green
    """
    # Count tokens in each family
    families = [[] for _ in range(n_clusters)]
    for token_id, family_id in enumerate(cluster_labels):
        families[family_id].append(token_id)

    # Sort families by size for better balance
    family_sizes = [(i, len(families[i])) for i in range(n_clusters)]
    family_sizes.sort(key=lambda x: x[1], reverse=True)

    # Assign families to green/red to achieve target gamma ratio
    total_tokens = len(cluster_labels)
    target_green = int(total_tokens * gamma)

    family_rng = np.random.RandomState(seed)
    # Shuffle to add randomness while maintaining balance
    indices = list(range(n_clusters))
    family_rng.shuffle(indices)

    green_count = 0
    green_families = set()
    for idx in indices:
        if green_count < target_green:
            green_families.add(idx)
            green_count += len(families[idx])
        else:
            break

    red_tokens_list = []
    green_tokens_list = []
    for token_id, cluster_id in enumerate(cluster_labels):
        if cluster_id in green_families:
            green_tokens_list.append(token_id)
        else:
            red_tokens_list.append(token_id)

    # Convert to tensors for efficiency
    red_tokens = torch.tensor(red_tokens_list, dtype=torch.long, device=device)
    green_tokens = torch.tensor(green_tokens_list, dtype=torch.long, device=device)

    return red_tokens, green_tokens


def spectral_clustering_assignment(
    codebook_vectors,
    n_clusters=100,
    sigma=1.0,
    k=100,
    epsilon=1e-12,
    seed=42,
    gamma=0.5,
    device="cuda",
):
    """
    High-accuracy spectral clustering for watermark families.
    Partitions N codebook tokens into M robust families, then assigns families
    to red/green with gamma fraction as green tokens. Uses GPU acceleration with KNN.

    Args:
        codebook_vectors: Codebook weight matrix (N x D), numpy array
        n_clusters: Number of families/clusters (default: 100)
        sigma: Scaling parameter for Gaussian kernel (default: 1.0)
        k: Number of nearest neighbors for KNN (default: 100)
        epsilon: Small number for numerical stability (default: 1e-12)
        seed: Random seed for family color assignment (default: 42)
        gamma: Fraction of tokens to mark as green (default: 0.5)
        device: Device to run computation on (default: 'cuda')

    Returns:
        red_tokens: Tensor of token indices assigned to red
        green_tokens: Tensor of token indices assigned to green
    """
    print(
        f"Running spectral clustering assignment (GPU, M={n_clusters}, sigma={sigma}, k={k}, gamma={gamma})..."
    )

    M = n_clusters

    print("  Phase 1: Computing KNN-based pairwise similarity matrix on GPU...")
    print("  Phase 2: Computing normalized Laplacian on GPU...")

    L_sym, N = compute_spectral_graph_laplacian(
        codebook_vectors, sigma, k, epsilon, device
    )

    print(f"    Dense graph constructed on GPU: {N} nodes")

    print(f"  Phase 3: Computing spectral embedding (finding {M} eigenvectors)...")

    # Use torch.linalg.eigh to compute all eigenvalues/eigenvectors on GPU
    # then select the M smallest
    eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)

    # Take the first M eigenvectors (corresponding to M smallest eigenvalues)
    U = eigenvectors[:, :M].cpu().numpy()
    eigenvalues_np = eigenvalues[:M].cpu().numpy()

    print(f"    Smallest {min(5, M)} eigenvalues: {eigenvalues_np[:min(5, M)]}")

    print(f"  Phase 4: Running K-Means with {M} clusters...")

    kmeans = KMeans(n_clusters=M, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(U)

    families = [[] for _ in range(M)]
    for token_idx, family_id in enumerate(labels):
        families[family_id].append(token_idx)

    print(f"    K-Means complete. {M} families created.")
    print(
        f"    Family sizes - min: {min(len(f) for f in families)}, max: {max(len(f) for f in families)}, avg: {N/M:.1f}"
    )

    print(f"  Phase 5: Assigning families to red/green (target gamma={gamma})...")

    family_indices = sorted(range(M), key=lambda i: len(families[i]), reverse=True)

    np.random.seed(seed)

    # Shuffle families for randomness
    block_size = max(1, M // 10)
    for block_start in range(0, M, block_size):
        block_end = min(block_start + block_size, M)
        block = family_indices[block_start:block_end]
        np.random.shuffle(block)
        family_indices[block_start:block_end] = block

    red_tokens_list = []
    green_tokens_list = []
    green_count = 0
    target_green = int(N * gamma)

    for family_idx in family_indices:
        family_size = len(families[family_idx])

        if green_count < target_green:
            green_tokens_list.extend(families[family_idx])
            green_count += family_size
        else:
            red_tokens_list.extend(families[family_idx])

    # Convert to tensors for efficiency
    red_tokens = torch.tensor(red_tokens_list, dtype=torch.long, device=device)
    green_tokens = torch.tensor(green_tokens_list, dtype=torch.long, device=device)

    print(f"Spectral clustering complete:")
    print(f"  Families: {M}")
    print(f"  Red tokens: {len(red_tokens)}, Green tokens: {len(green_tokens)}")
    print(
        f"  Split: {len(red_tokens)/(len(red_tokens)+len(green_tokens)):.2%} red, {len(green_tokens)/(len(red_tokens)+len(green_tokens)):.2%} green"
    )

    return red_tokens, green_tokens


def spectral_bisection_assignment(
    codebook_vectors, sigma=1.0, k=100, epsilon=1e-12, gamma=0.5, device="cuda"
):
    """
    Production-grade spectral bisection for robust watermark assignment.
    Partitions N codebook tokens into two sets (red and green) with gamma fraction as green
    using graph spectral methods with KNN-based pairwise distances on GPU.

    Args:
        codebook_vectors: Codebook weight matrix (N x D), numpy array
        sigma: Scaling parameter for Gaussian kernel (default: 1.0)
        k: Number of nearest neighbors for KNN (default: 100)
        epsilon: Small number for numerical stability (default: 1e-12)
        gamma: Fraction of tokens to mark as green (default: 0.5)
        device: Device to run computation on (default: 'cuda')

    Returns:
        red_tokens: Tensor of token indices assigned to red
        green_tokens: Tensor of token indices assigned to green
    """
    print(
        f"Running spectral bisection assignment (GPU, KNN-based, sigma={sigma}, k={k}, gamma={gamma})..."
    )

    print("  Phase 1: Computing KNN-based pairwise distance matrix on GPU...")
    print("  Phase 2: Computing normalized Laplacian on GPU...")

    L_sym, N = compute_spectral_graph_laplacian(
        codebook_vectors, sigma, k, epsilon, device
    )

    print(f"    Full graph constructed on GPU: {N} nodes, {N*(N-1)} potential edges")

    print("  Phase 3: Computing Fiedler vector...")

    # Use torch.linalg.eigh to compute all eigenvalues/eigenvectors on GPU
    # then select the first 2 (corresponding to smallest eigenvalues)
    eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)

    # Take the second eigenvector (Fiedler vector)
    fiedler_vector = eigenvectors[:, 1].cpu().numpy()
    eigenvalues_np = eigenvalues[:2].cpu().numpy()

    print(f"    Smallest eigenvalues: {eigenvalues_np[0]:.6f}, {eigenvalues_np[1]:.6f}")

    print(f"  Phase 4: Partitioning by percentile (target gamma={gamma})...")

    # Use percentile instead of median to achieve target gamma
    percentile = (1 - gamma) * 100
    threshold_val = np.percentile(fiedler_vector, percentile)

    color_map = np.where(fiedler_vector >= threshold_val, 1, 0)

    num_green = np.sum(color_map)
    target_green = int(N * gamma)

    if num_green != target_green:
        print(f"    Balancing: {num_green} green -> {target_green} green")

        distance_to_median = np.abs(fiedler_vector - np.median(fiedler_vector))

        if num_green > target_green:

            green_indices = np.where(color_map == 1)[0]
            green_distances = distance_to_median[green_indices]

            sorted_idx = np.argsort(green_distances)

            num_to_flip = num_green - target_green
            flip_indices = green_indices[sorted_idx[:num_to_flip]]
            color_map[flip_indices] = 0
        else:

            red_indices = np.where(color_map == 0)[0]
            red_distances = distance_to_median[red_indices]
            sorted_idx = np.argsort(red_distances)
            num_to_flip = target_green - num_green
            flip_indices = red_indices[sorted_idx[:num_to_flip]]
            color_map[flip_indices] = 1

    # Convert to tensors for efficiency
    red_tokens = torch.tensor(
        np.where(color_map == 0)[0], dtype=torch.long, device=device
    )
    green_tokens = torch.tensor(
        np.where(color_map == 1)[0], dtype=torch.long, device=device
    )

    print(f"Spectral bisection complete:")
    print(f"  Red tokens: {len(red_tokens)}, Green tokens: {len(green_tokens)}")
    actual_gamma = len(green_tokens) / (len(red_tokens) + len(green_tokens))
    print(f"  Achieved gamma: {actual_gamma:.4f} (target: {gamma:.4f})")

    return red_tokens, green_tokens


def generate_multiple_assignments(
    algorithm,
    codebook_vectors,
    num_assignments,
    cluster_size=1000,
    spectral_sigma=1.0,
    spectral_knn=100,
    spectral_families=100,
    base_seed=42,
    gamma=0.5,
    device="cuda",
):
    """
    Generate multiple red/green assignments in batch.

    Args:
        algorithm: One of ["pairwise", "random", "clustering", "spectral", "spectral-clustering"]
        codebook_vectors: Codebook weight vectors
        num_assignments: Number of different assignments to generate
        cluster_size: Cluster size for clustering algorithm
        spectral_sigma: Sigma parameter for spectral methods
        spectral_knn: Number of nearest neighbors for spectral methods
        spectral_families: Number of families for spectral clustering
        base_seed: Base random seed
        gamma: Fraction of tokens to mark as green (e.g., 0.5 for 50%)
        device: Device to use for computation

    Returns:
        List of tuples (red_tokens, green_tokens), one for each assignment
    """
    print(f"\n{'='*80}")
    print(f"Generating {num_assignments} assignment(s) using algorithm: {algorithm}")
    print(f"{'='*80}\n")

    assignments = []

    if algorithm == "pairwise":
        # Pairwise uses hierarchical codebook's green/red lists
        print(f"Pairwise algorithm - using HierarchicalCodebook lists...")
        if hc is None:
            raise ValueError(
                "HierarchicalCodebook (hc) must be provided for pairwise algorithm"
            )
        red_tokens = hc.red_list_tensor
        green_tokens = hc.green_list_tensor
        for i in range(num_assignments):
            assignments.append((red_tokens.clone(), green_tokens.clone()))
        print(
            f"  Created {num_assignments} identical assignments with {len(red_tokens)} red and {len(green_tokens)} green tokens"
        )

    elif algorithm == "spectral":
        # Deterministic spectral bisection - compute once and duplicate
        print(f"Spectral bisection is deterministic, computing once and duplicating...")
        red_tokens, green_tokens = spectral_bisection_assignment(
            codebook_vectors,
            sigma=spectral_sigma,
            k=spectral_knn,
            gamma=gamma,
            device=device,
        )
        print(f"  Spectral bisection: {len(red_tokens)} red, {len(green_tokens)} green")
        for i in range(num_assignments):
            assignments.append((red_tokens.clone(), green_tokens.clone()))
        print(f"  Created {num_assignments} identical assignments")

    elif algorithm == "random":
        # Random assignments - generate num_assignments different random splits
        print(f"Generating {num_assignments} different random assignments...")
        for i in range(num_assignments):
            seed = base_seed + i
            np.random.seed(seed)
            all_tokens = np.arange(codebook_vectors.shape[0])
            np.random.shuffle(all_tokens)
            split_idx = int(len(all_tokens) * gamma)
            green_tokens = torch.tensor(
                all_tokens[:split_idx], dtype=torch.long, device=device
            )
            red_tokens = torch.tensor(
                all_tokens[split_idx:], dtype=torch.long, device=device
            )
            assignments.append((red_tokens, green_tokens))
            print(
                f"  Assignment {i+1}/{num_assignments} (seed={seed}): {len(red_tokens)} red, {len(green_tokens)} green"
            )

    elif algorithm == "clustering":
        # K-means clustering - compute clusters once, then vary family assignment
        print(f"Computing K-means clusters once...")
        n_clusters = max(1, codebook_vectors.shape[0] // cluster_size)

        # Run K-means once with base_seed
        kmeans = KMeans(n_clusters=n_clusters, random_state=base_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(codebook_vectors)

        print(f"  K-means complete: {n_clusters} clusters")
        print(f"Generating {num_assignments} different family-to-color assignments...")

        # Generate different family assignments
        for i in range(num_assignments):
            seed = base_seed + i
            red_tokens, green_tokens = assign_families_to_colors(
                cluster_labels, n_clusters, seed, gamma=gamma, device=device
            )
            assignments.append((red_tokens, green_tokens))
            print(
                f"  Assignment {i+1}/{num_assignments} (seed={seed}): {len(red_tokens)} red, {len(green_tokens)} green"
            )

    elif algorithm == "spectral-clustering":
        # Spectral clustering - compute spectral embedding once, then vary family assignment
        print(f"Computing spectral embedding once...")

        # Phase 1-2: Compute graph Laplacian
        print(f"  Phase 1: Computing KNN-based pairwise distance matrix...")
        print(f"  Phase 2: Computing normalized Laplacian...")
        L_sym, N = compute_spectral_graph_laplacian(
            codebook_vectors, spectral_sigma, spectral_knn, device=device
        )

        # Phase 3: Eigendecomposition
        print(
            f"  Phase 3: Computing spectral embedding (finding {spectral_families} eigenvectors)..."
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
        U = eigenvectors[:, :spectral_families].cpu().numpy()
        eigenvalues_np = eigenvalues[:spectral_families].cpu().numpy()
        print(
            f"    Smallest {min(5, spectral_families)} eigenvalues: {eigenvalues_np[:min(5, spectral_families)]}"
        )

        # Phase 4: K-means on spectral embedding (once)
        print(f"  Phase 4: Running K-means on spectral embedding...")
        kmeans = KMeans(n_clusters=spectral_families, random_state=base_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(U)
        print(f"    K-means complete: {spectral_families} families")

        # Phase 5: Generate different family-to-color assignments
        print(f"Generating {num_assignments} different family-to-color assignments...")
        for i in range(num_assignments):
            seed = base_seed + i
            red_tokens, green_tokens = assign_families_to_colors(
                cluster_labels, spectral_families, seed, gamma=gamma, device=device
            )
            assignments.append((red_tokens, green_tokens))
            print(
                f"  Assignment {i+1}/{num_assignments} (seed={seed}): {len(red_tokens)} red, {len(green_tokens)} green"
            )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"\n{'='*80}")
    print(f"All {num_assignments} assignment(s) generated successfully!")
    print(f"{'='*80}\n")

    return assignments


def vectorized_replacement_process(
    index_sample, log_ratio_flat, log_threshold, hc_instance
):
    device = index_sample.device
    new_indices = index_sample.clone()

    if hc_instance.red_list_tensor.device != device:
        hc_instance.red_list_tensor = hc_instance.red_list_tensor.to(device)

    original_shape = new_indices.shape
    new_indices_flat = new_indices.view(-1)
    if log_ratio_flat.shape[0] != new_indices_flat.shape[0]:
        raise ValueError(
            f"Shape mismatch: log_ratio_flat ({log_ratio_flat.shape}) vs new_indices_flat ({new_indices_flat.shape})"
        )

    if hc_instance.red_list_tensor.numel() > 0:
        condition1_mask = torch.isin(new_indices_flat, hc_instance.red_list_tensor)
    else:
        condition1_mask = torch.zeros_like(
            new_indices_flat, dtype=torch.bool, device=device
        )

    if not isinstance(log_ratio_flat, torch.Tensor):
        log_ratio_flat = torch.from_numpy(log_ratio_flat).to(device)
    log_threshold_tensor = torch.tensor(
        log_threshold, device=device, dtype=log_ratio_flat.dtype
    )
    condition3_mask = log_ratio_flat <= log_threshold_tensor

    candidate_mask = condition1_mask & condition3_mask

    replacement_count = 0
    if candidate_mask.any():
        candidate_positions_flat = torch.where(candidate_mask)[0]

        original_token_ids_at_candidates = new_indices_flat[candidate_positions_flat]

        mapped_token_ids_list = []
        for token_id_tensor in original_token_ids_at_candidates:
            token_id = token_id_tensor.item()
            mapped_token_ids_list.append(
                hc_instance.index_mapping.get(token_id, token_id)
            )

        mapped_token_ids_tensor = torch.tensor(
            mapped_token_ids_list, dtype=new_indices_flat.dtype, device=device
        )
        actually_changed_mask_for_candidates = (
            mapped_token_ids_tensor != original_token_ids_at_candidates
        )

        final_indices_to_change = candidate_positions_flat[
            actually_changed_mask_for_candidates
        ]
        final_values_to_set = mapped_token_ids_tensor[
            actually_changed_mask_for_candidates
        ]

        if final_indices_to_change.numel() > 0:
            new_indices_flat[final_indices_to_change] = final_values_to_set
            replacement_count = final_indices_to_change.numel()

    new_indices_final = new_indices_flat.view(original_shape)

    return new_indices_final, replacement_count


def main(
    args,
    hc,
    vq_model,
    start_index,
    filename,
    prompts=None,
    algorithm="pairwise",
    codebook_weights=None,
    num_assignments=1,
    batch_size=1,
):
    """
    Main generation function.

    Args:
        algorithm: Strategy for token assignment to red/green lists.
                  - "pairwise": Use confidence-based assignment (for 100% watermark with pairwise replacement)
                  - "random": Use random assignment with logit boosting
                  - "baseline": No watermark (0% - keeps original token assignments)
                  - "clustering": Use K-means clustering with logit boosting
                  - "spectral": Use spectral bisection with logit boosting
                  - "spectral-clustering": Use spectral clustering with logit boosting
        codebook_weights: Codebook weight matrix (needed for clustering/spectral algorithms)
        num_assignments: Number of different red/green assignments to generate (one will be randomly selected per image)
        batch_size: Number of images to generate in parallel
    """
    filename_list = filename
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size**2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    print(f"Loading GPT checkpoint from: {args.gpt_ckpt}")
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")

    if "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        print(
            "Warning: Standard keys ('model', 'module', 'state_dict') not found. Attempting to load entire checkpoint as state_dict."
        )
        model_weight = checkpoint

    if all(key.startswith("module.") for key in model_weight.keys()):
        model_weight = {k.replace("module.", ""): v for k, v in model_weight.items()}

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
    del checkpoint, model_weight
    print(f"GPT model loaded successfully.")

    if args.compile:
        print(f"Compiling the model...")
        gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)
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

    # Pre-generate all assignments before image generation
    print(f"\n{'='*80}")
    print(f"PRE-GENERATING ASSIGNMENTS")
    print(f"{'='*80}\n")

    assignments_pool = generate_multiple_assignments(
        algorithm=algorithm,
        codebook_vectors=codebook_weights,
        num_assignments=num_assignments,
        cluster_size=args.cluster_size,
        spectral_sigma=args.spectral_sigma,
        spectral_knn=args.spectral_knn,
        spectral_families=args.spectral_families,
        base_seed=args.seed,
        gamma=args.gamma,
        device=device,
    )

    total_start_time = time.time()
    # Process prompts in batches
    num_prompts = len(prompts)
    for batch_start in range(0, num_prompts, batch_size):
        batch_end = min(batch_start + batch_size, num_prompts)
        current_batch_size = batch_end - batch_start
        batch_prompts = prompts[batch_start:batch_end]
        batch_filenames = filename_list[batch_start:batch_end]

        # Check if all images in this batch already exist (check both baseline and watermarked)
        baseline_save_dir = os.path.join(save_base_dir, f"{algorithm}-baseline-{args.image_size}")

        # For pairwise, watermarked uses delta=0.0 (not args.delta), so use special naming
        if algorithm == "pairwise":
            watermarked_save_dir = os.path.join(save_base_dir, f"{algorithm}-{args.image_size}")
        else:
            watermarked_save_dir = os.path.join(save_base_dir, f"{algorithm}-delta{args.delta}-{args.image_size}")

        all_exist = True
        for i in range(current_batch_size):
            im_idx = batch_filenames[i]
            baseline_path = os.path.join(baseline_save_dir, f"{im_idx}.png")
            watermarked_path = os.path.join(watermarked_save_dir, f"{im_idx}.png")
            if not (os.path.exists(baseline_path) and os.path.exists(watermarked_path)):
                all_exist = False
                break

        if all_exist and not args.overwrite:
            print(
                f"    Skipped batch {batch_start//batch_size + 1}/{(num_prompts + batch_size - 1)//batch_size}: All images already exist"
            )
            continue

        print(
            f"\n{'='*80}\n--- Processing Batch {batch_start//batch_size + 1}/{(num_prompts + batch_size - 1)//batch_size} (Prompts {batch_start+1}-{batch_end}/{num_prompts}, Batch Size: {current_batch_size}) ---\n{'='*80}"
        )
        batch_start_time = time.time()

        # Get embeddings for all prompts in batch
        caption_embs, emb_masks = t5_model.get_text_embeddings(batch_prompts)

        if not args.no_left_padding:
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs_list = []
            for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                valid_num = int(emb_mask.sum().item())
                new_caption_emb = torch.cat(
                    [caption_emb[valid_num:], caption_emb[:valid_num]]
                )
                new_caption_embs_list.append(new_caption_emb)
            new_caption_embs = torch.stack(new_caption_embs_list)
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks

        c_indices = new_caption_embs * new_emb_masks[:, :, None]
        c_emb_masks = new_emb_masks

        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]

        # Randomly select assignments for each image in the batch (seeded for reproducibility)
        # Note: For batched generation with different assignments per image, we'll use the same assignment
        # for all images in the batch for simplicity. Per-image assignment variation can be done across batches.
        batch_prompt_idx = batch_start
        selection_rng = np.random.RandomState(args.seed + batch_prompt_idx)
        assignment_idx = selection_rng.randint(0, len(assignments_pool))
        red_tokens, green_tokens = assignments_pool[assignment_idx]

        print(
            f"  Using assignment {assignment_idx + 1}/{len(assignments_pool)} for this batch (seed={args.seed + batch_prompt_idx})"
        )

        # Generate baseline and watermarked versions for each algorithm
        print(f"    Processing with algorithm: {algorithm}")

        if algorithm == "pairwise":
            # For pairwise: generate ONCE with delta=0.0, then create both baseline and watermarked
            print("  Generating initial indices with GPT (delta=0.0 for pairwise)...")
            t1 = time.time()

            index_sample, con = generate(
                gpt_model,
                c_indices,
                latent_size**2,
                c_emb_masks,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                sample_logits=True,
                index_mapping=None,  # No watermarking during generation
                green_list=None,
                delta=0.0,
            )
            sampling_time = time.time() - t1
            print(f"  GPT sampling took {sampling_time:.2f} seconds.")

            # Create baseline version (no replacement)
            baseline_indices = index_sample.clone()

            # Create watermarked version (replace ALL red tokens with green pairs)
            print(f"    Creating pairwise watermarked version (100% red->green replacement)")
            watermarked_indices = baseline_indices.clone()
            watermarked_indices_flat = watermarked_indices.view(-1)

            # Replace all red tokens with their green pairs
            replacement_count = 0
            if hc and hasattr(hc, 'red_list_tensor') and hasattr(hc, 'index_mapping'):
                is_red = torch.isin(watermarked_indices_flat, hc.red_list_tensor)
                red_positions = torch.where(is_red)[0]

                for pos in red_positions:
                    token_id = watermarked_indices_flat[pos].item()
                    green_token = hc.index_mapping.get(token_id, token_id)
                    if green_token != token_id:
                        watermarked_indices_flat[pos] = green_token
                        replacement_count += 1

            print(f"    Replaced {replacement_count} red tokens with green pairs")

            # Decode both versions
            print("  Decoding baseline with VQ model...")
            t2 = time.time()
            baseline_samples = vq_model.decode_code(baseline_indices, qzshape)
            print(f"  VQ decoding (baseline) took {time.time() - t2:.2f} seconds.")

            print("  Decoding watermarked with VQ model...")
            t2 = time.time()
            watermarked_samples = vq_model.decode_code(watermarked_indices, qzshape)
            print(f"  VQ decoding (watermarked) took {time.time() - t2:.2f} seconds.")

        else:
            # For other algorithms: generate TWICE (baseline with delta=0.0, watermarked with delta=2.0)
            print("  Generating baseline indices with GPT (delta=0.0)...")
            t1 = time.time()

            baseline_indices, _ = generate(
                gpt_model,
                c_indices,
                latent_size**2,
                c_emb_masks,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                sample_logits=True,
                index_mapping=None,
                green_list=None,
                delta=0.0,
            )
            baseline_sampling_time = time.time() - t1
            print(f"  GPT sampling (baseline) took {baseline_sampling_time:.2f} seconds.")

            print(f"  Generating watermarked indices with GPT (delta={args.delta})...")
            t1 = time.time()

            watermarked_indices, _ = generate(
                gpt_model,
                c_indices,
                latent_size**2,
                c_emb_masks,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                sample_logits=True,
                index_mapping=None,
                green_list=green_tokens,
                delta=args.delta,
            )
            watermarked_sampling_time = time.time() - t1
            print(f"  GPT sampling (watermarked) took {watermarked_sampling_time:.2f} seconds.")

            # Decode both versions
            print("  Decoding baseline with VQ model...")
            t2 = time.time()
            baseline_samples = vq_model.decode_code(baseline_indices, qzshape)
            print(f"  VQ decoding (baseline) took {time.time() - t2:.2f} seconds.")

            print("  Decoding watermarked with VQ model...")
            t2 = time.time()
            watermarked_samples = vq_model.decode_code(watermarked_indices, qzshape)
            print(f"  VQ decoding (watermarked) took {time.time() - t2:.2f} seconds.")

        # Save both baseline and watermarked images
        print("  Saving images...")
        baseline_save_dir = os.path.join(save_base_dir, f"{algorithm}-baseline-{args.image_size}")

        # For pairwise, watermarked uses delta=0.0 (not args.delta), so use special naming
        if algorithm == "pairwise":
            watermarked_save_dir = os.path.join(save_base_dir, f"{algorithm}-{args.image_size}")
        else:
            watermarked_save_dir = os.path.join(save_base_dir, f"{algorithm}-delta{args.delta}-{args.image_size}")

        os.makedirs(baseline_save_dir, exist_ok=True)
        os.makedirs(watermarked_save_dir, exist_ok=True)
        os.makedirs(os.path.join(baseline_save_dir, "assignments"), exist_ok=True)
        os.makedirs(os.path.join(watermarked_save_dir, "assignments"), exist_ok=True)

        for i in range(current_batch_size):
            im_idx = batch_filenames[i]

            # Save baseline image
            baseline_path = os.path.join(baseline_save_dir, f"{im_idx}.png")
            if not (os.path.exists(baseline_path) and not args.overwrite):
                save_image(
                    baseline_samples[i : i + 1], baseline_path, nrow=1, normalize=True, value_range=(-1, 1)
                )
                print(f"    Saved baseline {batch_start + i + 1}/{num_prompts}: {baseline_path}")

            # Save watermarked image
            watermarked_path = os.path.join(watermarked_save_dir, f"{im_idx}.png")
            if not (os.path.exists(watermarked_path) and not args.overwrite):
                save_image(
                    watermarked_samples[i : i + 1], watermarked_path, nrow=1, normalize=True, value_range=(-1, 1)
                )
                print(f"    Saved watermarked {batch_start + i + 1}/{num_prompts}: {watermarked_path}")

            # Save assignment JSONs for both versions
            if algorithm == "pairwise":
                if hc and hasattr(hc, "pairs"):
                    # Baseline assignment
                    baseline_assignment = {
                        "image_id": im_idx,
                        "algorithm": "pairwise",
                        "version": "baseline",
                        "pairs": hc.pairs,
                        "red_list": list(hc.red_list),
                        "green_list": list(hc.green_list),
                        "num_pairs": len(hc.pairs),
                        "parameters": {
                            "replacement_ratio": 0.0,
                            "delta": 0.0,
                        },
                    }
                    baseline_json_path = os.path.join(baseline_save_dir, "assignments", f"{im_idx}.json")
                    with open(baseline_json_path, "w") as f:
                        json.dump(baseline_assignment, f, indent=2)

                    # Watermarked assignment
                    watermarked_assignment = {
                        "image_id": im_idx,
                        "algorithm": "pairwise",
                        "version": "watermarked",
                        "pairs": hc.pairs,
                        "red_list": list(hc.red_list),
                        "green_list": list(hc.green_list),
                        "num_pairs": len(hc.pairs),
                        "parameters": {
                            "replacement_ratio": 1.0,  # 100% replacement for watermarked
                            "delta": 0.0,
                        },
                    }
                    watermarked_json_path = os.path.join(watermarked_save_dir, "assignments", f"{im_idx}.json")
                    with open(watermarked_json_path, "w") as f:
                        json.dump(watermarked_assignment, f, indent=2)

            else:
                # Baseline assignment for other algorithms
                baseline_assignment = {
                    "image_id": im_idx,
                    "assignment_id": assignment_idx,
                    "algorithm": algorithm,
                    "version": "baseline",
                    "red_list": red_tokens.cpu().tolist(),
                    "green_list": green_tokens.cpu().tolist(),
                    "num_red": len(red_tokens),
                    "num_green": len(green_tokens),
                    "parameters": {
                        "gamma": args.gamma,
                        "seed": args.seed + assignment_idx,
                        "delta": 0.0,
                    },
                }
                if algorithm == "clustering":
                    baseline_assignment["parameters"]["cluster_size"] = args.cluster_size
                elif algorithm in ["spectral", "spectral-clustering"]:
                    baseline_assignment["parameters"]["spectral_sigma"] = args.spectral_sigma
                    if algorithm == "spectral-clustering":
                        baseline_assignment["parameters"]["spectral_families"] = args.spectral_families

                baseline_json_path = os.path.join(baseline_save_dir, "assignments", f"{im_idx}.json")
                with open(baseline_json_path, "w") as f:
                    json.dump(baseline_assignment, f, indent=2)

                # Watermarked assignment
                watermarked_assignment = {
                    "image_id": im_idx,
                    "assignment_id": assignment_idx,
                    "algorithm": algorithm,
                    "version": "watermarked",
                    "red_list": red_tokens.cpu().tolist(),
                    "green_list": green_tokens.cpu().tolist(),
                    "num_red": len(red_tokens),
                    "num_green": len(green_tokens),
                    "parameters": {
                        "gamma": args.gamma,
                        "seed": args.seed + assignment_idx,
                        "delta": args.delta,
                    },
                }
                if algorithm == "clustering":
                    watermarked_assignment["parameters"]["cluster_size"] = args.cluster_size
                elif algorithm in ["spectral", "spectral-clustering"]:
                    watermarked_assignment["parameters"]["spectral_sigma"] = args.spectral_sigma
                    if algorithm == "spectral-clustering":
                        watermarked_assignment["parameters"]["spectral_families"] = args.spectral_families

                watermarked_json_path = os.path.join(watermarked_save_dir, "assignments", f"{im_idx}.json")
                with open(watermarked_json_path, "w") as f:
                    json.dump(watermarked_assignment, f, indent=2)

        batch_end_time = time.time()
        print(
            f"--- Batch finished in {batch_end_time - batch_start_time:.2f} seconds ({current_batch_size} images, {(batch_end_time - batch_start_time)/current_batch_size:.2f}s per image) ---"
        )

    total_end_time = time.time()
    print(
        f"\n=== Total generation finished in {total_end_time - total_start_time:.2f} seconds ==="
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Text to image AR generation, using pairing based codebook replacement."
    )

    parser.add_argument(
        "--t5-path",
        type=str,
        default="/cmlscratch/anirudhs/hub",
        help="Path to T5 checkpoint directory.",
    )
    parser.add_argument(
        "--gpt-ckpt",
        type=str,
        default="/cmlscratch/anirudhs/hub/t2i_XL_stage2_512.pt",
        help="Path to GPT checkpoint file.",
    )
    parser.add_argument(
        "--vq-ckpt",
        type=str,
        default="/cmlscratch/anirudhs/hub/vq_ds16_t2i.pt",
        help="VQ model checkpoint path.",
    )
    parser.add_argument(
        "--mapping-save-path",
        type=str,
        default="results/codebook_index_mapping.json",
        help="Save/load the path of the index mapping JSON file.",
    )
    parser.add_argument(
        "--pairs-save-path",
        type=str,
        default="results/codebook_pairs.json",
        help="Path to save/load codebook pairing list JSON file.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/cmlscratch/anirudhs/graph_watermark/images/t2i_experiments",
        help="Base directory to save generated images.",
    )

    parser.add_argument(
        "--t5-model-type",
        type=str,
        default="t5-v1_1-xl",
        help="Specific T5 model type (e.g., 'flan-t5-xl', 't5-v1_1-xxl').",
    )
    parser.add_argument(
        "--gpt-model",
        type=str,
        choices=list(GPT_models.keys()),
        default="GPT-XL",
        help="GPT model architecture.",
    )
    parser.add_argument(
        "--gpt-type",
        type=str,
        choices=["c2i", "t2i"],
        default="t2i",
        help="GPT task type: class-to-image or text-to-image.",
    )

    parser.add_argument(
        "--vq-model",
        type=str,
        choices=list(VQ_models.keys()) if VQ_models else ["VQ-16"],
        default="VQ-16",
    )
    parser.add_argument(
        "--codebook-size", type=int, default=16384, help="The codebook size of VQ"
    )
    parser.add_argument(
        "--codebook-embed-dim",
        type=int,
        default=8,
        help="The embedding dimension of VQ codebook.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        choices=[256, 384, 512],
        default=512,
        help="Target image resolution.",
    )
    parser.add_argument(
        "--downsample-size",
        type=int,
        choices=[8, 16],
        default=16,
        help="VAE downsampling factor.",
    )
    parser.add_argument(
        "--t5-feature-max-len",
        type=int,
        default=120,
        help="Max sequence length for T5 features.",
    )
    parser.add_argument(
        "--cls-token-num",
        type=int,
        default=120,
        help="Max token number for condition input in GPT.",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=7.5, help="Classifier-Free Guidance scale."
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--top-k", type=int, default=1000, help="Top-k sampling parameter."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter."
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["none", "fp16", "bf16"],
        help="Computation precision.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable torch.compile for the GPT model (requires PyTorch 2.0+).",
    )
    parser.add_argument(
        "--no-left-padding",
        action="store_true",
        default=False,
        help="Disable left-padding for T5 embeddings.",
    )
    parser.add_argument("--replacement-ratio", type=float, default=1.0)
    parser.add_argument(
        "--load-mapping",
        action="store_true",
        help="Attempt to load index mapping from file instead of recalculating.",
    )
    parser.add_argument(
        "--load-pairs",
        action="store_true",
        help="Attempt to load the codebook pairing list from the file instead of recalculating.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="pairwise",
        choices=[
            "pairwise",
            "random",
            "clustering",
            "spectral",
            "spectral-clustering",
        ],
        help="Algorithm for token assignment: 'pairwise' uses confidence-based assignment for 100%% watermark, 'random' uses random assignment, 'baseline' is for 0%% non-watermarked baseline, 'clustering' uses K-means clustering, 'spectral' uses spectral bisection for perfect 50/50 split, 'spectral-clustering' uses spectral clustering with families.",
    )
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=157,
        help="Number of clusters for K-means clustering algorithm (only used when algorithm='clustering').",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="Delta value for logit boosting of green tokens (used for clustering, random, and spectral algorithms).",
    )
    parser.add_argument(
        "--spectral-sigma",
        type=float,
        default=1.0,
        help="Gaussian kernel scaling parameter for spectral methods (used for 'spectral' and 'spectral-clustering').",
    )
    parser.add_argument(
        "--spectral-families",
        type=int,
        default=157,
        help="Number of families/clusters for spectral clustering (only used when algorithm='spectral-clustering').",
    )
    parser.add_argument(
        "--spectral-knn",
        type=int,
        default=100,
        help="Number of nearest neighbors (k) for KNN-based spectral methods (used for 'spectral' and 'spectral-clustering').",
    )
    parser.add_argument(
        "--num-assignments",
        type=int,
        default=10,
        help="Number of different red/green assignments to generate. For each image, one assignment will be randomly selected from this pool.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images to generate in parallel. Higher values increase speed but require more GPU memory.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Fraction of vocabulary to mark as green tokens (e.g., 0.5 for 50%%, 0.25 for 25%%). Used for partitioning in clustering/spectral algorithms.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set, overwrite existing images. If not set (default), skip generation for images that already exist.",
    )

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
        codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim
    )

    should_load_vq_weights = not (
        args.load_pairs
        and args.load_mapping
        and os.path.exists(args.pairs_save_path)
        and os.path.exists(args.mapping_save_path)
    )

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
        codebook_weights = (
            vq_model.quantize.embedding.weight.data.cpu().numpy().astype(np.float32)
        )
        print(
            f"The codebook weights have been extracted (shape:{codebook_weights.shape})"
        )
    else:
        print(
            "Detected attempting to load pairing and mapping from file, skipping loading VQ model weights."
        )
        codebook_weights = np.zeros(
            (args.codebook_size, args.codebook_embed_dim), dtype=np.float32
        )
        print(
            f"Initialize using placeholder codebook weights with the shape: {codebook_weights.shape}."
        )

    hc = HierarchicalCodebook(
        codebook_vectors=codebook_weights,
        replacement_ratio=args.replacement_ratio,
        mapping_save_path=args.mapping_save_path,
        pairs_save_path=args.pairs_save_path,
        load_mapping=True,
        load_pairs=True,
    )

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim
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
        print(
            "Warning: Standard keys ('model', 'state_dict', 'ema') not found in VQ checkpoint. Attempting to load entire checkpoint."
        )
        vq_state_dict = checkpoint_vq

    if all(key.startswith("module.") for key in vq_state_dict.keys()):
        vq_state_dict = {k.replace("module.", ""): v for k, v in vq_state_dict.items()}

    load_result_vq = vq_model.load_state_dict(vq_state_dict, strict=False)
    print(f"VQ load results: {load_result_vq}")
    if load_result_vq.missing_keys:
        print(f"Warning: VQ Missing keys: {load_result_vq.missing_keys}")
    if load_result_vq.unexpected_keys:
        print(f"Warning: VQ Unexpected keys: {load_result_vq.unexpected_keys}")

    del checkpoint_vq, vq_state_dict
    print("VQ model loaded.")

    captions_list, ids = mscocojson2list("mscoco_1000/coco_1000.json")
    # captions_list = captions_list[:5]
    # ids = ids[:5]
    main(
        args,
        hc,
        vq_model,
        start_index=0,
        filename=ids,
        prompts=captions_list,
        algorithm=args.algorithm,
        codebook_weights=codebook_weights,
        num_assignments=args.num_assignments,
        batch_size=args.batch_size,
    )
