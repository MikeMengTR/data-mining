#!/usr/bin/env python
"""
Utility script to inspect the intermediate `measure_bos` representations.

It loads a trained PianoLLaMA checkpoint, encodes a configurable number of
dataset items, extracts the encoder BOS vectors for every measure, and runs a
simple statistical + t-SNE analysis.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
except ImportError:  # pragma: no cover
    umap = None
from tqdm import tqdm
from transformers import AutoConfig

from config import ModelConfig
from model import PianoLLaMA
from PianoDataset import PianoDataset


def load_model(checkpoint_dir: Path, device: torch.device) -> PianoLLaMA:
    """Load PianoLLaMA weights from a checkpoint directory."""
    hf_config = AutoConfig.from_pretrained(checkpoint_dir)
    model = PianoLLaMA(hf_config)
    state_path = checkpoint_dir / "model.safetensors"
    state_dict = load_file(str(state_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch. Missing: {missing[:5]}, unexpected: {unexpected[:5]}"
        )
    return model.to(device).eval()


def extract_measure_bos(
    model: PianoLLaMA,
    sample_tokens: torch.Tensor,
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Run the encoder on a single sample and return the BOS vectors for
    every measure row (includes both parts).
    """
    # sample_tokens: (measures, measure_len)
    input_ids = sample_tokens.unsqueeze(0)  # (1, measures, len)
    attention_mask = (input_ids != pad_token_id).long()

    batch_size, num_measures, measure_len = input_ids.shape
    flat_tokens = input_ids.reshape(batch_size * num_measures, measure_len).to(device)
    flat_attention = attention_mask.reshape(batch_size * num_measures, measure_len).to(device)

    with torch.no_grad():
        encoder_outputs = model.encoder(
            input_ids=flat_tokens,
            attention_mask=flat_attention,
            use_cache=False,
            output_hidden_states=False,
        )

    measure_bos = encoder_outputs.last_hidden_state[:, -1, :]  # (measures, hidden)
    return measure_bos.cpu()


def compute_stats(vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Basic descriptive statistics for the BOS embeddings."""
    norms = np.linalg.norm(vectors, axis=1)
    active_ratios = np.array([m["active_ratio"] for m in metadata])

    part_masks = {
        "high": np.array([m["part"] == "high" for m in metadata]),
        "low": np.array([m["part"] == "low" for m in metadata]),
    }

    part_means = {}
    for part, mask in part_masks.items():
        if mask.any():
            part_means[part] = vectors[mask].mean(axis=0)

    cosine = None
    if all(part in part_means for part in ("high", "low")):
        a = part_means["high"]
        b = part_means["low"]
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    dim_var = vectors.var(axis=0)
    top_var_idx = np.argsort(dim_var)[-5:][::-1]

    pca = PCA(n_components=min(5, vectors.shape[1]))
    pca.fit(vectors)

    return {
        "vector_count": int(vectors.shape[0]),
        "hidden_size": int(vectors.shape[1]),
        "unique_files": len({m["file_name"] for m in metadata}),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "norm_range": [float(norms.min()), float(norms.max())],
        "mean_active_ratio": float(active_ratios.mean()),
        "std_active_ratio": float(active_ratios.std()),
        "cosine_between_voice_means": cosine,
        "top_variance_dimensions": top_var_idx.tolist(),
        "top_variances": dim_var[top_var_idx].tolist(),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    }


def _subsample_vectors(
    vectors: np.ndarray,
    metadata: List[Dict[str, Any]],
    seed: int,
    max_points: int,
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """Randomly down-sample vectors+metadata to at most max_points."""
    rng = np.random.default_rng(seed)
    total = vectors.shape[0]
    if total <= max_points:
        idx = np.arange(total)
    else:
        idx = rng.choice(total, size=max_points, replace=False)
    return vectors[idx], [metadata[int(i)] for i in idx]


def _scatter_plot(embedding: np.ndarray, subset_meta: List[Dict[str, Any]], path: Path, title: str):
    """Shared scatter plotting helper."""
    plt.figure(figsize=(8, 6))
    colors = {"high": "#1f77b4", "low": "#ff7f0e"}
    for part in ("high", "low"):
        mask = np.array([meta["part"] == part for meta in subset_meta])
        if mask.any():
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=10,
                alpha=0.7,
                label=f"{part} voice",
                c=colors[part],
            )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_tsne(
    vectors: np.ndarray,
    metadata: List[Dict[str, Any]],
    output_dir: Path,
    seed: int,
    max_points: int,
    perplexity: float,
) -> Dict[str, Any]:
    """Sub-sample (if needed) and run t-SNE."""
    subset_vectors, subset_meta = _subsample_vectors(vectors, metadata, seed, max_points)

    eff_perplexity = min(perplexity, max(5.0, len(subset_vectors) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=eff_perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(subset_vectors)

    np.save(output_dir / "measure_bos_tsne.npy", embedding)
    _scatter_plot(embedding, subset_meta, output_dir / "measure_bos_tsne.png", "t-SNE of measure_bos embeddings")

    return {
        "tsne_points": embedding.tolist(),
        "tsne_metadata": subset_meta,
        "perplexity_used": eff_perplexity,
    }


def run_umap(
    vectors: np.ndarray,
    metadata: List[Dict[str, Any]],
    output_dir: Path,
    seed: int,
    max_points: int,
    n_neighbors: int,
    min_dist: float,
) -> Dict[str, Any]:
    """Run UMAP projection if the dependency is available."""
    if umap is None:
        raise RuntimeError("umap-learn is not installed. Please `pip install umap-learn`.")

    subset_vectors, subset_meta = _subsample_vectors(vectors, metadata, seed, max_points)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        metric="euclidean",
    )
    embedding = reducer.fit_transform(subset_vectors)

    np.save(output_dir / "measure_bos_umap.npy", embedding)
    _scatter_plot(embedding, subset_meta, output_dir / "measure_bos_umap.png", "UMAP of measure_bos embeddings")

    return {
        "umap_points": embedding.tolist(),
        "umap_metadata": subset_meta,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
    }

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Analyze measure_bos embeddings.")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("/home/cby/not_use/Advanced/generative_newtoken_improved_1_4_relative_track_RT_Compress_measure/checkpoints/steps_30000_1107_0001"))
    parser.add_argument("--data_dir", type=Path,
                        default=Path("/DATA2_4T/cby/home/lab-wei.zhenao/boyu/Dataset/allxml_npz_dual_track_optimized"))
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of dataset items to encode.")
    parser.add_argument("--output_dir", type=Path, default=Path("analysis_outputs"))
    parser.add_argument("--dataset_split", choices=["train", "test"], default="train")
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne_max_points", type=int, default=10000)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--umap_max_points", type=int, default=4000)
    parser.add_argument("--umap_neighbors", type=int, default=25)
    parser.add_argument("--umap_min_dist", type=float, default=0.3)
    parser.add_argument("--projection", choices=["tsne", "umap", "both"], default="tsne")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model from {args.checkpoint_dir} on {device}...")
    model = load_model(args.checkpoint_dir, device)

    data_cfg = ModelConfig()
    dataset = PianoDataset(
        data_dir=str(args.data_dir),
        config=data_cfg,
        cache_lengths=False,
        mode=args.dataset_split,
        test_split_ratio=args.test_split,
        random_seed=args.seed,
    )

    max_samples = min(args.num_samples, len(dataset))
    print(f"Collecting measure_bos from {max_samples} items...")

    vectors: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []

    for idx in tqdm(range(max_samples), desc="Encoding samples"):
        sample = dataset[idx]
        bos = extract_measure_bos(model, sample["input_ids"], data_cfg.pad_token_id, device)
        vectors.append(bos.numpy())

        rows = sample["input_ids"].numpy()
        file_name = dataset.data_files[idx]

        for row_idx, row in enumerate(rows):
            active = int((row != data_cfg.pad_token_id).sum())
            metadata.append(
                {
                    "vector_index": len(metadata),
                    "dataset_index": idx,
                    "file_name": file_name,
                    "measure_index": row_idx // 2,
                    "part": "high" if row_idx % 2 == 0 else "low",
                    "active_tokens": active,
                    "active_ratio": active / len(row),
                }
            )

    all_vectors = np.concatenate(vectors, axis=0)
    print(f"Collected {all_vectors.shape[0]} measure_bos vectors of dimension {all_vectors.shape[1]}.")
    np.save(output_dir / "measure_bos_vectors.npy", all_vectors)
    with open(output_dir / "measure_bos_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    stats = compute_stats(all_vectors, metadata)
    with open(output_dir / "measure_bos_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if args.projection in ("tsne", "both"):
        tsne_info = run_tsne(
            all_vectors,
            metadata,
            output_dir,
            seed=args.seed,
            max_points=args.tsne_max_points,
            perplexity=args.tsne_perplexity,
        )
        with open(output_dir / "measure_bos_tsne_metadata.json", "w", encoding="utf-8") as f:
            json.dump(tsne_info, f, ensure_ascii=False)

    if args.projection in ("umap", "both"):
        umap_info = run_umap(
            all_vectors,
            metadata,
            output_dir,
            seed=args.seed,
            max_points=args.umap_max_points,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        with open(output_dir / "measure_bos_umap_metadata.json", "w", encoding="utf-8") as f:
            json.dump(umap_info, f, ensure_ascii=False)

    print(f"Saved embeddings, stats, and plots under {output_dir}")


if __name__ == "__main__":
    main()
