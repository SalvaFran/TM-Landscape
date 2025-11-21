#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_umap_gpu.py
--------------------

Generate a global 2D UMAP manifold for TM-Vec reference embeddings using RAPIDS cuML.

Usage:
    CUDA_VISIBLE_DEVICES=2 python generate_umap_gpu.py --source cath --size large

Outputs:
    data/UMAP/Z_UMAP_2D.npy
    data/UMAP/umap_model_2D.pkl
    data/UMAP/umap_limits_2D.json
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
import pickle

from cuml.manifold import UMAP as UMAP_GPU
import cupy as cp

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.append(str(PROJECT_ROOT))

from scripts.utils.load_tmvec_embeddings import load_tmvec_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Generate global 2D GPU-UMAP for TM-Vec embeddings.")
    parser.add_argument("--source", choices=["cath", "swiss"], required=True)
    parser.add_argument("--size", choices=["small", "large"], required=True)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading TM-Vec reference: {args.source}_{args.size}")
    Z_ref, metadata = load_tmvec_embeddings(source=args.source, size=args.size, frac=1.0)
    print(f"[INFO] Z_ref shape = {Z_ref.shape}")

    umap_dir = PROJECT_ROOT / "data" / "UMAP"
    umap_dir.mkdir(parents=True, exist_ok=True)

    out_embed = umap_dir / "Z_UMAP_2D.npy"
    out_model = umap_dir / "umap_model_2D.pkl"
    out_limits = umap_dir / "umap_limits_2D.json"

    print("[INFO] Moving embeddings to GPU...")
    Z_gpu = cp.asarray(Z_ref)

    print("[INFO] Running GPU UMAP...")
    umap_model = UMAP_GPU(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=42,
        verbose=True
    )

    Z_umap_gpu = umap_model.fit_transform(Z_gpu)
    Z_umap = cp.asnumpy(Z_umap_gpu)

    print(f"[INFO] Saving embedding → {out_embed}")
    np.save(out_embed, Z_umap)

    print(f"[INFO] Saving model → {out_model}")
    pickle.dump(umap_model, open(out_model, "wb"))

    print("[INFO] Saving UMAP limits...")
    limits = {
        "xmin": float(Z_umap[:, 0].min()),
        "xmax": float(Z_umap[:, 0].max()),
        "ymin": float(Z_umap[:, 1].min()),
        "ymax": float(Z_umap[:, 1].max()),
    }
    json.dump(limits, open(out_limits, "w"), indent=4)

    print("\n[FINISHED] Global GPU UMAP created successfully!")


if __name__ == "__main__":
    main()
