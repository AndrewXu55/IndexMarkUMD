import sys
import torch
import os
import time
import argparse
import numpy as np
import gc
import json
import traceback
from tqdm import tqdm

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

        if not isinstance(codebook_vectors, np.ndarray):
            raise TypeError("codebook_vectors must be a NumPy array")
        if codebook_vectors.ndim != 2:
            if not (load_mapping and load_pairs and codebook_vectors.shape[0] > 0):
                raise ValueError(
                    "codebook_vectors must be a 2D array (size, dim) unless loading mapping and pairs."
                )

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
                    print(
                        f"Warning: Adjusting internal codebook size based on loaded pairs to {max_index_in_pairs + 1}"
                    )
                    self.codebook_size = max_index_in_pairs + 1
                    self.unassigned_indices = set(range(self.codebook_size))

                self._reconstruct_state_from_pairs()

        if load_mapping:
            loaded_mapping = self.load_mapping(self.mapping_save_path)
            if loaded_mapping and not loaded_pairs:
                print(
                    "Warning: Mapping loaded, but pair list was not. Red/Green lists populated, but pairing info missing."
                )
                self.red_list = set(self.index_mapping.keys())
                self.green_list = set(self.index_mapping.values())
                max_index_in_mapping = 0
                if self.index_mapping:
                    all_indices = list(self.index_mapping.keys()) + list(
                        self.index_mapping.values()
                    )
                    max_index_in_mapping = max(all_indices) if all_indices else -1
                if (
                    self.codebook_size <= max_index_in_mapping
                    or self.codebook_size == 0
                ):
                    print(
                        f"Warning: Adjusting internal codebook size based on loaded mapping to {max_index_in_mapping + 1}"
                    )
                    self.codebook_size = max_index_in_mapping + 1
                    self.unassigned_indices = (
                        set(range(self.codebook_size)) - self.red_list - self.green_list
                    )

        calculation_needed = False
        if loaded_pairs and loaded_mapping:
            print("Successfully loaded pair list and index mapping from files.")
        elif loaded_pairs and not loaded_mapping:
            print(
                "Loaded pair list, but not mapping. Will generate mapping from loaded pairs."
            )
            self._create_red_green_lists()
            if self.index_mapping:
                self.save_mapping(self.mapping_save_path)
            else:
                print(
                    "Warning: Mapping generation from loaded pairs failed or resulted in empty mapping. Not saving."
                )
        elif not loaded_pairs and loaded_mapping:
            print(
                "Loaded mapping, but not pair list. Will use loaded mapping and skip pair calculation."
            )
        else:
            if (
                self.codebook_size == 0
                or self.feature_dim == 0
                or np.all(self.codebook == 0)
            ):
                print(
                    "Error: Cannot compute pairs/mapping. Codebook vectors are missing or zero (and loading failed or was not requested)."
                )
                raise ValueError(
                    "Codebook vectors required for calculation but not provided or loaded."
                )
            else:
                print("Will compute pair list and index mapping.")
                calculation_needed = True

        if calculation_needed:
            self._build_pairs()
            if not self.pairs:
                print(
                    "Error: Pair building failed or resulted in an empty pair list. Cannot proceed."
                )
                raise RuntimeError("Failed to build codebook pairs.")
            self._create_red_green_lists()
            if self.index_mapping:
                self.save_mapping(self.mapping_save_path)
            else:
                print(
                    "Warning: Mapping generation from computed pairs failed or resulted in empty mapping. Not saving."
                )

        # Create red_list_tensor and green_list_tensor AFTER red_list/green_list are populated
        self.red_list_tensor = torch.tensor(
            list(self.red_list), dtype=torch.long, device=device
        )
        self.green_list_tensor = torch.tensor(
            list(self.green_list), dtype=torch.long, device=device
        )

        print("HierarchicalCodebook initialization complete.")
        print(f"  - Codebook size used: {self.codebook_size}")
        print(f"  - Number of pairs: {len(self.pairs)}")
        print(f"  - Mapping size: {len(self.index_mapping)}")
        print(f"  - Red list size: {len(self.red_list)}")
        print(f"  - Green list size: {len(self.green_list)}")
        print(f"  - Unassigned indices: {len(self.unassigned_indices)}")

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
                print(
                    f"Warning: Pair ({idx1}, {idx2}) contains out-of-bounds index for codebook shape {self.codebook.shape}. Skipping similarity calculation for this pair."
                )
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
                print(
                    f"    Analyzed similarities for {count}/{len(self.pairs)} pairs..."
                )

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
            print(
                f"  Warning: Red/Green list total ({actual_total}) does not match expected ({expected_total}). May be due to duplicate indices in loaded pairs or mapping."
            )

    def get_random_replacement_org(self, original_idx):
        if (
            original_idx in self.red_list
            and np.random.random() < self.replacement_ratio
        ):
            return self.index_mapping.get(original_idx, original_idx)
        return original_idx

    def get_random_replacement(self, original_idx, log_ratio_value, threshold):
        try:
            log_ratio_val = float(log_ratio_value)
            threshold_val = float(threshold)
        except (ValueError, TypeError):
            print(
                f"Warning: Invalid log_ratio_value ({log_ratio_value}) or threshold ({threshold}). Skipping replacement check."
            )
            return original_idx

        replace_decision = np.random.random() < self.replacement_ratio

        if (
            original_idx in self.red_list
            and replace_decision
            and log_ratio_val <= threshold_val
        ):
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
            serializable_mapping = {
                int(k): int(v) for k, v in self.index_mapping.items()
            }
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_mapping, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved {len(self.index_mapping)} mapping entries.")
        except TypeError as e:
            print(
                f"Error: Failed to serialize index mapping. Check data types. Error: {e}"
            )
        except Exception as e:
            print(f"Error: Failed to save index mapping to {filepath}. Error: {e}")

    def load_mapping(self, filepath):
        print(f"Attempting to load index mapping from {filepath}...")
        if not os.path.exists(filepath):
            print(f"Info: Mapping file {filepath} not found.")
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
                f"Error: Failed to parse JSON file {filepath}. File might be corrupt. Error: {e}"
            )
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
            with open(filepath, "w", encoding="utf-8") as f:
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
                f"Error: Failed to parse JSON file {filepath}. File might be corrupt. Error: {e}"
            )
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
            print(
                f"Note: Adjusting internal codebook size to {max_index_in_pairs + 1} based on loaded pairs for state reconstruction."
            )
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
                print(
                    f"Warning: Loaded pair {(idx1, idx2)} contains out-of-bounds index for codebook size {self.codebook_size}. Skipping this pair."
                )
        if duplicates_found:
            print(
                "Warning: Duplicate indices found across different pairs in the loaded file. This might affect red/green list generation."
            )
        self.unassigned_indices = set(range(self.codebook_size)) - indices_seen
        print(
            f"State reconstruction complete. Valid pairs processed: {valid_pairs_count}. Indices assigned: {len(indices_seen)}. Unassigned indices: {len(self.unassigned_indices)}"
        )


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
    d_inv_sqrt = 1.0 / torch.sqrt(d + epsilon)
    D_inv_sqrt_W = d_inv_sqrt.unsqueeze(1) * W
    L_sym = torch.eye(N, device=device) - D_inv_sqrt_W * d_inv_sqrt.unsqueeze(0)

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

        median_val = np.median(fiedler_vector)
        distance_to_median = np.abs(fiedler_vector - median_val)

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
    hc=None,
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
            f"log_ratio_flat ({log_ratio_flat.shape}) vs new_indices_flat ({new_indices_flat.shape})"
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
    vq_model_main,
    gpt_model,
    algorithm="pairwise",
    codebook_weights=None,
    num_assignments=1,
):
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
    print(
        f"Generating {num_seeds_per_class} samples for each of {total_classes} classes."
    )
    print(f"Processing in batches of size {batch_size}.")

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
        hc=hc,
    )

    total_images_generated = 0
    overall_start_time = time.time()

    # Determine save directory and watermark label early
    if algorithm == "pairwise":
        watermark_label = "pairwise"
    elif algorithm in ["random", "clustering", "spectral", "spectral-clustering"]:
        if args.delta == 0.0:
            watermark_label = f"{algorithm}-baseline"
        else:
            watermark_label = f"{algorithm}-delta{args.delta}"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    current_save_dir = os.path.join(save_base_dir, f"{watermark_label}-{args.image_size}")
    os.makedirs(current_save_dir, exist_ok=True)
    os.makedirs(os.path.join(current_save_dir, "assignments"), exist_ok=True)

    for i in range(0, total_classes, batch_size):
        batch_start_time = time.time()
        current_batch_labels = all_class_labels[i : min(i + batch_size, total_classes)]
        current_batch_size = len(current_batch_labels)
        if current_batch_size == 0:
            continue

        print(
            f"\n--- Processing Batch {i // batch_size + 1}/{(total_classes + batch_size - 1) // batch_size} (Classes {current_batch_labels[0]} to {current_batch_labels[-1]}) ---"
        )
        c_indices = torch.tensor(current_batch_labels, device=device)

        for seed_idx in range(num_seeds_per_class):
            seed_start_time = time.time()
            current_seed = base_seed + (i * num_seeds_per_class) + seed_idx
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            print(
                f"  Seed {seed_idx + 1}/{num_seeds_per_class} (Global Seed: {current_seed})"
            )

            # Check if all images in this batch/seed already exist
            all_exist = True
            for img_idx_in_batch in range(current_batch_size):
                actual_class_label = current_batch_labels[img_idx_in_batch]
                filename = f"class{actual_class_label:04d}_seed{seed_idx:02d}.png"
                save_path = os.path.join(current_save_dir, filename)
                if not os.path.exists(save_path):
                    all_exist = False
                    break

            if all_exist and not args.overwrite:
                print(
                    f"    Skipped seed {seed_idx + 1}/{num_seeds_per_class}: All images already exist"
                )
                continue

            # Randomly select assignment for this batch/seed (seeded for reproducibility)
            assignment_idx_seed = current_seed
            selection_rng = np.random.RandomState(assignment_idx_seed)
            assignment_idx = selection_rng.randint(0, len(assignments_pool))
            red_tokens, green_tokens = assignments_pool[assignment_idx]

            print(
                f"    Using assignment {assignment_idx + 1}/{len(assignments_pool)} for this seed"
            )

            # Configure generation parameters based on algorithm
            if algorithm == "pairwise":
                # Pairwise uses confidence-based replacement with index_mapping
                index_map_to_pass = (
                    hc.index_mapping if hc and hasattr(hc, "index_mapping") else None
                )
                green_list_to_pass = None
                delta_to_pass = 0.0
            elif algorithm in [
                "random",
                "clustering",
                "spectral",
                "spectral-clustering",
            ]:
                # These algorithms use logit boosting with green_list
                # delta=0.0 means no watermarking (baseline), delta=2.0 means watermarking
                index_map_to_pass = None
                if args.delta > 0:
                    green_list_to_pass = green_tokens
                else:
                    # When delta=0, no watermarking (baseline mode for this algorithm)
                    green_list_to_pass = None
                delta_to_pass = args.delta
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            # 1. Generate initial indices and confidence scores using GPT (once per batch/seed)
            t1 = time.time()
            index_sample, con = generate(
                gpt_model,
                c_indices,  # Pass current batch class indices
                latent_size**2,
                cfg_scale=args.cfg_scale,
                cfg_interval=args.cfg_interval,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                sample_logits=True,
                index_mapping=index_map_to_pass,
                green_list=green_list_to_pass,
                delta=delta_to_pass,
            )
            sampling_time = time.time() - t1
            print(f"    GPT sampling took {sampling_time:.2f} seconds.")

            print(f"    Processing with algorithm: {algorithm}")

            # For pairwise algorithm, do confidence-based replacement at 100%
            # For other algorithms, watermarking is already applied during generation via logit boosting
            if algorithm == "pairwise":
                conf_pairs = con
                original_conf = conf_pairs[:, :, 0].cpu().numpy()
                paired_conf = conf_pairs[:, :, 1].cpu().numpy()

                epsilon = 1e-9
                ratio = np.divide(original_conf, paired_conf + epsilon)
                log_ratio = np.log10(np.maximum(ratio, epsilon))
                log_ratio_flat = log_ratio.flatten()
                valid_log_ratio = log_ratio_flat[np.isfinite(log_ratio_flat)]

                # Apply 100% watermarking
                if len(valid_log_ratio) > 0:
                    log_threshold = np.percentile(valid_log_ratio, 100)
                else:
                    log_threshold = -np.inf
                new_indices = index_sample.clone()
                new_indices, count = vectorized_replacement_process(
                    index_sample, log_ratio_flat, log_threshold, hc
                )
                print(f"      Replaced {count} tokens (100% watermarking)")

                t2 = time.time()
                qzshape = [
                    current_batch_size,
                    args.codebook_embed_dim,
                    latent_size,
                    latent_size,
                ]
                replaced_samples = vq_model_main.decode_code(new_indices, qzshape)
                decoder_time = time.time() - t2
                print(f"      VQ decoding took {decoder_time:.2f} seconds.")

                for img_idx_in_batch in range(current_batch_size):
                    actual_class_label = current_batch_labels[img_idx_in_batch]
                    filename = f"class{actual_class_label:04d}_seed{seed_idx:02d}.png"
                    save_path = os.path.join(current_save_dir, filename)

                    save_image(
                        replaced_samples[img_idx_in_batch],
                        save_path,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    total_images_generated += 1

                    # Save assignment JSON for pairwise algorithm
                    if hc and hasattr(hc, "pairs"):
                        assignment_data = {
                            "image_id": f"class{actual_class_label:04d}_seed{seed_idx:02d}",
                            "algorithm": "pairwise",
                            "pairs": hc.pairs,
                            "red_list": list(hc.red_list),
                            "green_list": list(hc.green_list),
                            "num_pairs": len(hc.pairs),
                            "parameters": {
                                "replacement_ratio": args.replacement_ratio,
                            },
                        }
                        json_path = os.path.join(
                            current_save_dir,
                            "assignments",
                            f"class{actual_class_label:04d}_seed{seed_idx:02d}.json",
                        )
                        with open(json_path, "w") as f:
                            json.dump(assignment_data, f, indent=2)
            else:
                # For non-pairwise algorithms, watermarking is already applied via logit boosting
                new_indices = index_sample.clone()

                t2 = time.time()
                qzshape = [
                    current_batch_size,
                    args.codebook_embed_dim,
                    latent_size,
                    latent_size,
                ]
                replaced_samples = vq_model_main.decode_code(new_indices, qzshape)
                decoder_time = time.time() - t2
                print(f"      VQ decoding took {decoder_time:.2f} seconds.")

                for img_idx_in_batch in range(current_batch_size):
                    actual_class_label = current_batch_labels[img_idx_in_batch]
                    filename = f"class{actual_class_label:04d}_seed{seed_idx:02d}.png"
                    save_path = os.path.join(current_save_dir, filename)

                    save_image(
                        replaced_samples[img_idx_in_batch],
                        save_path,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    total_images_generated += 1

                    # Save assignment JSON for non-pairwise algorithms
                    assignment_data = {
                        "image_id": f"class{actual_class_label:04d}_seed{seed_idx:02d}",
                        "assignment_id": assignment_idx,
                        "algorithm": algorithm,
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

                    # Add algorithm-specific parameters
                    if algorithm == "clustering":
                        assignment_data["parameters"][
                            "cluster_size"
                        ] = args.cluster_size
                    elif algorithm in ["spectral", "spectral-clustering"]:
                        assignment_data["parameters"][
                            "spectral_sigma"
                        ] = args.spectral_sigma
                        if algorithm == "spectral-clustering":
                            assignment_data["parameters"][
                                "spectral_families"
                            ] = args.spectral_families

                    json_path = os.path.join(
                        current_save_dir,
                        "assignments",
                        f"class{actual_class_label:04d}_seed{seed_idx:02d}.json",
                    )
                    with open(json_path, "w") as f:
                        json.dump(assignment_data, f, indent=2)

            seed_end_time = time.time()
            print(
                f"    Seed {seed_idx + 1} processed in {seed_end_time - seed_start_time:.2f}s."
            )
            gc.collect()

        batch_end_time = time.time()
        print(
            f"--- Batch {i // batch_size + 1} finished in {batch_end_time - batch_start_time:.2f}s ---"
        )
        gc.collect()

    overall_end_time = time.time()
    print(f"\n=== Total generation finished ===")
    print(f"Generated {total_images_generated} images in total.")
    print(f"Total time: {overall_end_time - overall_start_time:.2f} seconds ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Class-to-Image AR generation with HC-based codebook replacement (Multi-Seed)"
    )

    parser.add_argument(
        "--gpt-model",
        type=str,
        choices=list(GPT_models.keys()),
        default="GPT-L",
        help="GPT model architecture.",
    )
    parser.add_argument(
        "--gpt-ckpt",
        type=str,
        default="/cmlscratch/anirudhs/hub/c2i_L_256.pt",
        help="Path to C2I GPT checkpoint file.",
    )
    parser.add_argument(
        "--gpt-type",
        type=str,
        choices=["c2i", "t2i"],
        default="c2i",
        help="GPT task type (should be c2i).",
    )
    parser.add_argument(
        "--from-fsdp", action="store_true", help="Load checkpoint saved with FSDP."
    )
    parser.add_argument(
        "--cls-token-num",
        type=int,
        default=1,
        help="Max token number of condition input (typically 1 for class).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes for Imagenet (or total to generate).",
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
        "--cfg-scale",
        type=float,
        default=4.0,
        help="Classifier-Free Guidance scale for C2I.",
    )
    parser.add_argument(
        "--cfg-interval", type=float, default=10, help="CFG interval frequency."
    )

    parser.add_argument(
        "--vq-model",
        type=str,
        choices=list(VQ_models.keys()) if VQ_models else ["VQ-16"],
        default="VQ-16",
        help="VQ model architecture.",
    )
    parser.add_argument(
        "--vq-ckpt",
        type=str,
        default="/cmlscratch/anirudhs/hub/vq_ds16_c2i.pt",
        help="Path to C2I VQ model checkpoint.",
    )
    parser.add_argument(
        "--codebook-size", type=int, default=16384, help="VQ codebook size."
    )
    parser.add_argument(
        "--codebook-embed-dim",
        type=int,
        default=8,
        help="VQ codebook embedding dimension.",
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Base random seed for reproducibility."
    )
    parser.add_argument(
        "--top-k", type=int, default=2000, help="Top-k sampling parameter."
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
        "--save-dir",
        type=str,
        default="images/Gen_Image_c2i",
        help="Base directory to save generated images.",
    )
    parser.add_argument(
        "--replacement-ratio",
        type=float,
        default=1.0,
        help="Proportion of red-list tokens eligible for replacement (0.0-1.0).",
    )
    parser.add_argument(
        "--mapping-save-path",
        type=str,
        default="results/codebook_index_mapping_knn10_mwpm_c2i.json",
        help="Path to load index mapping JSON file for C2I VQ.",
    )
    parser.add_argument(
        "--pairs-save-path",
        type=str,
        default="results/codebook_pairs_knn10_mwpm_c2i.json",
        help="Path to load codebook pair list JSON file for C2I VQ.",
    )
    parser.add_argument(
        "--load-mapping",
        action="store_true",
        default=True,
        help="Attempt to load index mapping from file instead of recomputing.",
    )
    parser.add_argument(
        "--load-pairs",
        action="store_true",
        default=True,
        help="Attempt to load codebook pair list from file instead of recomputing.",
    )
    parser.add_argument(
        "--no-load-mapping",
        action="store_false",
        dest="load_mapping",
        help="Disable loading index mapping.",
    )
    parser.add_argument(
        "--no-load-pairs",
        action="store_false",
        dest="load_pairs",
        help="Disable loading codebook pairs.",
    )
    parser.add_argument(
        "--num-seeds-per-class",
        type=int,
        default=10,
        help="Number of different images (seeds) to generate per class.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of classes to process in parallel per generation step.",
    )

    # New algorithm arguments
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
        help="Algorithm for token assignment: 'pairwise' uses confidence-based assignment for 100%% watermark, 'random' uses random assignment, 'clustering' uses K-means clustering, 'spectral' uses spectral bisection for perfect 50/50 split, 'spectral-clustering' uses spectral clustering with families. Use --delta 0.0 for baseline (no watermarking) or --delta 2.0 for watermarking.",
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

    # Determine if we need codebook weights for the algorithm
    needs_codebook_weights = args.algorithm in [
        "random",
        "clustering",
        "spectral",
        "spectral-clustering",
    ]

    # Determine if we need HC (pairwise) calculation
    needs_hc_calculation = False
    if args.algorithm == "pairwise":
        # Only check HC files for pairwise algorithm
        if args.load_pairs and args.load_mapping:
            if not os.path.exists(args.pairs_save_path):
                print(
                    f"Info: Pair file {args.pairs_save_path} not found, will need calculation if mapping load also fails or is disabled."
                )
                needs_hc_calculation = True  # Need calculation if file is missing
            if not os.path.exists(args.mapping_save_path):
                print(
                    f"Info: Mapping file {args.mapping_save_path} not found, will need calculation if pair load also fails or is disabled."
                )
                needs_hc_calculation = True  # Need calculation if file is missing
            if os.path.exists(args.pairs_save_path) and os.path.exists(
                args.mapping_save_path
            ):
                print("Found existing pair and mapping files. Will attempt to load.")
                needs_hc_calculation = False
            else:
                needs_hc_calculation = True

        elif args.load_pairs:
            if not os.path.exists(args.pairs_save_path):
                print(
                    f"Error: Pair file {args.pairs_save_path} not found, and mapping loading not requested. Calculation needed but codebook vectors may be missing."
                )
                needs_hc_calculation = True
            else:
                print("Found pair file, will load pairs and calculate mapping.")
                needs_hc_calculation = False

        elif args.load_mapping:
            if not os.path.exists(args.mapping_save_path):
                print(
                    f"Error: Mapping file {args.mapping_save_path} not found, and pair loading not requested. Calculation needed."
                )
                needs_hc_calculation = True  # Need calculation
            else:
                print(
                    "Found mapping file, will load mapping and skip pair calculation."
                )
                needs_hc_calculation = False  # Don't need weights or pair calc

        else:
            print("Loading of pairs and mapping disabled. Calculation is required.")
            needs_hc_calculation = True

    # Load codebook weights if needed (for HC calculation OR for clustering/spectral algorithms)
    if needs_hc_calculation or needs_codebook_weights:
        if needs_hc_calculation:
            print(
                "Loading VQ model weights temporarily to extract codebook for HC calculation..."
            )
        else:
            print(f"Loading VQ model weights for {args.algorithm} algorithm...")
        try:
            temp_vq_model = VQ_models[args.vq_model](
                codebook_size=args.codebook_size,
                codebook_embed_dim=args.codebook_embed_dim,
            )
            if not os.path.exists(args.vq_ckpt):
                raise FileNotFoundError(
                    f"C2I VQ checkpoint file not found for HC calculation: {args.vq_ckpt}"
                )

            checkpoint_vq_temp = torch.load(args.vq_ckpt, map_location="cpu")
            if "model" in checkpoint_vq_temp:
                state_dict_vq = checkpoint_vq_temp["model"]
            elif "state_dict" in checkpoint_vq_temp:
                state_dict_vq = checkpoint_vq_temp["state_dict"]
            elif "ema" in checkpoint_vq_temp:
                state_dict_vq = checkpoint_vq_temp["ema"]  # Added ema key
            else:
                state_dict_vq = checkpoint_vq_temp

            if all(key.startswith("module.") for key in state_dict_vq.keys()):
                state_dict_vq = {
                    k.replace("module.", ""): v for k, v in state_dict_vq.items()
                }

            if any(key.startswith("vq.") for key in state_dict_vq.keys()):
                state_dict_vq = {
                    k.replace("vq.", ""): v
                    for k, v in state_dict_vq.items()
                    if k.startswith("vq.")
                }
                if "quantize.embedding.weight" not in state_dict_vq:
                    if "embedding.weight" in state_dict_vq:
                        state_dict_vq["quantize.embedding.weight"] = state_dict_vq.pop(
                            "embedding.weight"
                        )
                    else:
                        raise KeyError(
                            "Could not find codebook embedding weights ('quantize.embedding.weight' or 'embedding.weight') after stripping 'vq.' prefix."
                        )

            load_res = temp_vq_model.load_state_dict(state_dict_vq, strict=False)
            print(f"Temporary VQ load result for HC: {load_res}")
            if (
                "quantize.embedding.weight" in load_res.missing_keys
                and "embedding.weight" not in state_dict_vq
            ):
                original_keys = checkpoint_vq_temp.get(
                    "model",
                    checkpoint_vq_temp.get(
                        "state_dict", checkpoint_vq_temp.get("ema", checkpoint_vq_temp)
                    ),
                ).keys()
                print(
                    "Available top-level keys in VQ checkpoint:",
                    checkpoint_vq_temp.keys(),
                )
                print("Available keys in VQ state_dict:", original_keys)

                raise KeyError(
                    f"Could not find codebook embedding weights key 'quantize.embedding.weight' or similar in VQ checkpoint {args.vq_ckpt}. Checkpoint structure might be unexpected."
                )

            if hasattr(temp_vq_model, "quantize") and hasattr(
                temp_vq_model.quantize, "embedding"
            ):
                codebook_weights = (
                    temp_vq_model.quantize.embedding.weight.data.cpu()
                    .numpy()
                    .astype(np.float32)
                )
            elif hasattr(temp_vq_model, "embedding"):  # Simpler VQ structure?
                codebook_weights = (
                    temp_vq_model.embedding.weight.data.cpu().numpy().astype(np.float32)
                )
            else:
                raise AttributeError(
                    "Could not find embedding layer in the temporary VQ model structure."
                )

            print(f"Codebook weights extracted (shape: {codebook_weights.shape})")
            if codebook_weights.shape != (args.codebook_size, args.codebook_embed_dim):
                print(
                    f"Warning: Extracted codebook shape {codebook_weights.shape} does not match arguments ({args.codebook_size}, {args.codebook_embed_dim}). Using extracted shape."
                )
                args.codebook_size = codebook_weights.shape[
                    0
                ]  # Update args to reflect reality
                print(
                    f"Warning: Make sure --codebook-size ({args.codebook_size}) matches the GPT model's vocab_size."
                )

            del temp_vq_model, checkpoint_vq_temp, state_dict_vq  # Free memory
            gc.collect()
        except Exception as e:
            print(f"ERROR: Failed to load VQ weights: {e}")
            traceback.print_exc()
            if needs_hc_calculation:
                print("Cannot proceed without codebook vectors for HC calculation.")
            else:
                print(
                    f"Cannot proceed without codebook vectors for {args.algorithm} algorithm."
                )
            sys.exit(1)
    else:
        # No HC calculation needed and algorithm doesn't need codebook weights (baseline or using pre-loaded HC)
        if args.algorithm == "baseline":
            print(
                "Baseline algorithm selected - no watermarking, no codebook weights needed."
            )
            codebook_weights = np.zeros(
                (args.codebook_size, args.codebook_embed_dim), dtype=np.float32
            )
        else:
            # Pairwise with pre-loaded mapping
            print(
                "Attempting to load HC pairs/mapping from files, skipping VQ weight extraction."
            )
            codebook_weights = np.zeros(
                (args.codebook_size, args.codebook_embed_dim), dtype=np.float32
            )
            print(
                f"Using placeholder codebook weights (shape: {codebook_weights.shape}) for HC initialization."
            )

    # Only initialize HierarchicalCodebook for pairwise algorithm
    if args.algorithm == "pairwise":
        try:
            hc = HierarchicalCodebook(
                codebook_vectors=codebook_weights,
                replacement_ratio=args.replacement_ratio,
                mapping_save_path=args.mapping_save_path,
                pairs_save_path=args.pairs_save_path,
                load_mapping=args.load_mapping,
                load_pairs=args.load_pairs,
            )
            if args.load_mapping and (
                not hc.index_mapping or len(hc.index_mapping) == 0
            ):
                print(
                    "WARNING: Loading mapping was requested, but the index mapping is empty after initialization."
                )
                if not os.path.exists(args.mapping_save_path):
                    print(
                        f"         Reason: Mapping file '{args.mapping_save_path}' not found."
                    )
                else:
                    print(
                        f"         Reason: Mapping file '{args.mapping_save_path}' might be empty, corrupt, or failed to load."
                    )
                if (
                    not needs_hc_calculation and not args.load_pairs
                ):  # No fallback possible
                    print("         Cannot proceed without a valid mapping.")
                    sys.exit(1)
                elif needs_hc_calculation:
                    print(
                        "         Proceeding with calculation based on extracted weights."
                    )
                elif args.load_pairs:
                    print("         Will attempt to derive mapping from loaded pairs.")

        except Exception as e:
            print(f"FATAL: Failed to initialize HierarchicalCodebook: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"Using {args.algorithm} algorithm - HierarchicalCodebook not needed.")

    # --- Load Main VQ Model (for decoding) ---
    print("\nLoading main VQ model for decoding...")
    vq_model_main = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,  # Use potentially updated codebook size
        codebook_embed_dim=args.codebook_embed_dim,
    ).to(device)
    vq_model_main.eval()

    if not os.path.exists(args.vq_ckpt):
        raise FileNotFoundError(
            f"Main C2I VQ checkpoint file not found: {args.vq_ckpt}"
        )
    checkpoint_vq_main = torch.load(args.vq_ckpt, map_location="cpu")

    if "model" in checkpoint_vq_main:
        vq_state_dict_main = checkpoint_vq_main["model"]
    elif "state_dict" in checkpoint_vq_main:
        vq_state_dict_main = checkpoint_vq_main["state_dict"]
    elif "ema" in checkpoint_vq_main:
        vq_state_dict_main = checkpoint_vq_main["ema"]
    else:
        vq_state_dict_main = checkpoint_vq_main

    if all(key.startswith("module.") for key in vq_state_dict_main.keys()):
        vq_state_dict_main = {
            k.replace("module.", ""): v for k, v in vq_state_dict_main.items()
        }

    if any(key.startswith("vq.") for key in vq_state_dict_main.keys()):
        vq_state_dict_main = {
            k.replace("vq.", ""): v
            for k, v in vq_state_dict_main.items()
            if k.startswith("vq.")
        }

    load_result_vq_main = vq_model_main.load_state_dict(
        vq_state_dict_main, strict=False
    )
    print(f"Main VQ load results: {load_result_vq_main}")
    if load_result_vq_main.missing_keys:
        print(f"Warning: Main VQ Missing keys: {load_result_vq_main.missing_keys}")
    if load_result_vq_main.unexpected_keys:
        print(
            f"Warning: Main VQ Unexpected keys: {load_result_vq_main.unexpected_keys}"
        )

    del checkpoint_vq_main, vq_state_dict_main  # Free memory
    gc.collect()
    print("Main VQ model loaded and ready on device.")

    print("\nLoading main GPT model for generation...")
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size**2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    print(f"Loading GPT checkpoint from: {args.gpt_ckpt}")
    if not os.path.exists(args.gpt_ckpt):
        raise FileNotFoundError(f"GPT checkpoint not found: {args.gpt_ckpt}")
    checkpoint_gpt = torch.load(args.gpt_ckpt, map_location="cpu")

    if args.from_fsdp:
        model_weight = checkpoint_gpt
    elif "model" in checkpoint_gpt:
        model_weight = checkpoint_gpt["model"]
    elif "module" in checkpoint_gpt:
        model_weight = checkpoint_gpt["module"]
    elif "state_dict" in checkpoint_gpt:
        model_weight = checkpoint_gpt["state_dict"]
    else:
        model_weight = checkpoint_gpt

    if all(key.startswith("module.") for key in model_weight.keys()):
        model_weight = {k.replace("module.", ""): v for k, v in model_weight.items()}

    try:
        load_result_gpt = gpt_model.load_state_dict(model_weight, strict=False)
        print(f"GPT load results: {load_result_gpt}")
        mismatched_pos_emb = False
        if (
            "pos_embed" in load_result_gpt.missing_keys
            or "pos_embed" in load_result_gpt.unexpected_keys
        ):
            print(
                "Warning: Positional embedding mismatch detected. This might happen if image size/downsampling changed."
            )
            mismatched_pos_emb = True  # Flag it
        if (
            load_result_gpt.missing_keys and not mismatched_pos_emb
        ):  # Report other missing keys
            print(f"Warning: GPT Missing keys: {load_result_gpt.missing_keys}")
        if (
            load_result_gpt.unexpected_keys and not mismatched_pos_emb
        ):  # Report other unexpected keys
            print(f"Warning: GPT Unexpected keys: {load_result_gpt.unexpected_keys}")

    except Exception as e:
        print(f"Error loading GPT state dict: {e}")
        traceback.print_exc()
        raise e

    gpt_model.eval()
    del checkpoint_gpt, model_weight  # Free up memory
    gc.collect()
    print(f"GPT model loaded successfully.")

    if args.compile:
        print(f"Compiling the GPT model...")
        try:
            gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)
            print("GPT model compiled.")
        except Exception as e:
            print(
                f"Warning: GPT Model compilation failed: {e}. Running without compilation."
            )
    else:
        print(f"GPT model compilation disabled.")

    print("\nStarting main generation process...")
    main(
        args,
        hc,
        vq_model_main,
        gpt_model,
        algorithm=args.algorithm,
        codebook_weights=codebook_weights,
        num_assignments=args.num_assignments,
    )

    print("\nScript finished.")
