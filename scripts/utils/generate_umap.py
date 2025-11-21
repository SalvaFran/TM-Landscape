#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_umap.py
----------------

Generate a global UMAP manifold for TM-Vec reference embeddings (Z_ref).

Usage:
    python generate_umap.py --source cath --size large --n_components 2

Outputs will be stored under:
    data/UMAP/Z_UMAP_{nD}.npy
    data/UMAP/umap_model_{nD}.pkl
    data/UMAP/umap_limits_{nD}.json   (only for nD=2)

The script uses GPU UMAP (cuML) if installed, otherwise CPU UMAP.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json

# Try GPU UMAP
try:
    from cuml.manifold import UMAP as UMAP_GPU
    import cupy as cp
    GPU_AVAILABLE = True
except Exception:
    from umap import UMAP as UMAP_CPU
    GPU_AVAILABLE = False

# TM-Landscape root
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.append(str(PROJECT_ROOT))

from scripts.utils.load_tmvec_embeddings import load_tmvec_embeddings


# ============================================================
# Parse arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate global UMAP for TM-Vec reference manifold.")
    parser.add_argument("--source", choices=["cath", "swiss"], required=True)
    parser.add_argument("--size", choices=["small", "large"], required=True)
    parser.add_argument("--n_components", type=int, default=2)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.05)

    return parser.parse_args()


# ============================================================
# UMAP training
# ============================================================

def run_umap(Z_ref, n_components, n_neighbors, min_dist):
    if GPU_AVAILABLE:
        print("[UMAP] Using GPU UMAP (cuML)")
        Z_gpu = cp.asarray(Z_ref)

        umap_model = UMAP_GPU(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
            verbose=True
        )
        Z_embed_gpu = umap_model.fit_transform(Z_gpu)
        Z_embed = cp.asnumpy(Z_embed_gpu)
    else:
        print("[UMAP] GPU not available → using CPU UMAP")
        umap_model = UMAP_CPU(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=42
        )
        Z_embed = umap_model.fit_transform(Z_ref)

    return umap_model, Z_embed


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    print(f"[INFO] Loading TM-Vec reference manifold: {args.source}_{args.size}")
    Z_ref, metadata = load_tmvec_embeddings(source=args.source, size=args.size, frac=1.0)

    umap_dir = PROJECT_ROOT / "data" / "UMAP"
    umap_dir.mkdir(exist_ok=True, parents=True)

    out_embed = umap_dir / f"Z_UMAP_{args.n_components}D.npy"
    out_model = umap_dir / f"umap_model_{args.n_components}D.pkl"
    out_limits = umap_dir / f"umap_limits_{args.n_components}D.json"

    print("[INFO] Running UMAP...")
    umap_model, Z_embed = run_umap(
        Z_ref,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )

    print(f"[INFO] Saving embedding → {out_embed}")
    np.save(out_embed, Z_embed)

    # Save UMAP model
    import pickle
    pickle.dump(umap_model, open(out_model, "wb"))

    # Save limits only for 2D embeddings
    if args.n_components == 2:
        xmin, xmax = float(Z_embed[:, 0].min()), float(Z_embed[:, 0].max())
        ymin, ymax = float(Z_embed[:, 1].min()), float(Z_embed[:, 1].max())

        limits = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
        json.dump(limits, open(out_limits, "w"), indent=4)
        print(f"[INFO] Saved limits → {out_limits}")

    print("\nDone. Global UMAP generated.")


if __name__ == "__main__":
    main()
