import argparse
import torch
import numpy as np
from tokenizer.tokenizer_image.vq_model import VQ_models
import gc
import os
from autoregressive.models.gpt import GPT_models
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
        default="/cmlscratch/anirudhs/hub/vq_ds16_c2i.pt",
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
        default="images/Gen_Image",
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
        default=512,
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
        default=512,
        help="Number of families/clusters for spectral clustering (only used when algorithm='spectral-clustering').",
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X = torch.tensor(codebook_weights, dtype=torch.float32, device=device)  # 16384 x 8

    def compute_rbf_affinity(X, sigma, k=100):
        """
        Compute RBF affinity matrix A_ij = exp(-||x_i-x_j||^2 / (2*sigma^2)) on GPU.
        """
        # pairwise squared distances
        with torch.no_grad():
            # Pairwise squared distances
            XX = (X**2).sum(dim=1, keepdim=True)
            dists = XX + XX.T - 2 * X @ X.T  # shape N x N

            # For each row, keep only k nearest neighbors
            knn_mask = torch.zeros_like(dists, dtype=torch.bool)
            topk = torch.topk(
                dists, k=k + 1, largest=False
            )  # +1 because self-distance is zero
            knn_mask.scatter_(1, topk.indices, True)

            # Symmetrize the mask (i connected to j if either i->j or j->i)
            knn_mask = knn_mask | knn_mask.T

            # Apply RBF only to k-NN entries
            A = torch.zeros_like(dists)
            A[knn_mask] = torch.exp(-dists[knn_mask] / (2 * sigma**2))
            del dists, knn_mask
            return A

    def compute_normalized_laplacian(A, epsilon=1e-12):
        """
        Compute symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
        """
        D = torch.diag(A.sum(dim=1))
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D) + epsilon))
        L = torch.eye(A.shape[0], device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
        return L

    def plot_eigengap(eigenvalues, sigma, title="Eigengap Spectrum"):
        gaps = eigenvalues[1:] - eigenvalues[:-1]
        gaps = gaps.cpu().numpy()
        idx = torch.argsort(-torch.tensor(gaps), dim=0)
        top_20 = idx[:20]
        print("Top 20 eigenvalue gaps (descending order):")
        print(top_20)
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(gaps) + 1), gaps, marker="o")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue gap")
        plt.title(f"{title} (sigma={sigma})")
        plt.grid(True)
        plt.savefig(f"eigengap-sigma={sigma:.2f}.png")

    # -----------------------------
    # Parameter sweep
    # -----------------------------
    sigmas = [0.5, 1.0, 2.0]
    n_families_list = [32, 64, 128, 256, 512]

    for sigma in sigmas:
        print(f"\n=== Testing sigma = {sigma} ===")

        # Step 1: RBF affinity
        A = compute_rbf_affinity(X, sigma)

        # Step 2: Normalized Laplacian
        L = compute_normalized_laplacian(A)

        # Step 3: Eigen decomposition (torch.linalg.eigh is GPU accelerated)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        print(f"Smallest 10 eigenvalues: {eigenvalues[:10].cpu().numpy()}")

        # Step 4: Eigengap plot
        plot_eigengap(
            eigenvalues[:1000], sigma, title=f"Eigengap Spectrum (sigma={sigma:.2f})"
        )

    n_clusters = 157

    # Take the first n_clusters eigenvectors (spectral embedding)
    U = eigenvectors[:, :n_clusters]  # 16384 x 157

    # Move to CPU for scikit-learn k-means
    U_np = U.cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(U_np)

    # Compute intra-cluster distances
    intra_cluster_distances = []
    cluster_sizes = []
    for c in range(n_clusters):
        points = U_np[labels == c]  # points in cluster c
        cluster_sizes.append(len(points))
        if len(points) <= 1:
            intra_cluster_distances.append(0.0)
            continue
        # Compute pairwise squared distances
        diff = (
            points[:, None, :] - points[None, :, :]
        )  # shape: num_points x num_points x n_clusters
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))  # Euclidean distances
        # Average distance (excluding self-distance)
        mean_dist = np.sum(dist_matrix) / (len(points) * (len(points) - 1))
        intra_cluster_distances.append(mean_dist)

    intra_cluster_distances = np.array(intra_cluster_distances)
    print(f"Clusters: {n_clusters}")
    print(f"Cluster sizes: {cluster_sizes}")
    print(f"Min intra-cluster distance: {intra_cluster_distances.min():.4f}")
    print(f"Avg intra-cluster distance: {intra_cluster_distances.mean():.4f}")
    print(f"Max intra-cluster distance: {intra_cluster_distances.max():.4f}")
