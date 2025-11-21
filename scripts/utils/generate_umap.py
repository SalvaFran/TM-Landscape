#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_umap.py
----------------

Generate a global UMAP manifold for TM-Vec reference embeddings (Z_ref).

Usage:
    python generate_umap.py --source cath --size large --n_components 2

Outputs:
    data/UMAP/Z_UMAP_{nD}.npy
    data/UMAP/umap_model_{nD}.pkl
    data/UMAP/umap_limits_{nD}.json   (only for nD=2)

IMPORTANT:
    umap-learn CANNOT run on GPU (CuPy arrays not supported by scikit-learn).
    Therefore, UMAP always runs on CPU.

    If Z_ref is a CuPy array, it will be converted to NumPy explicitly.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
import pickle

# Optional CuPy detection (only for .get() conversion)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("[INFO] CuPy found (GPU arrays may appear, but UMAP will run on CPU).")
except Exception:
    CUPY_AVAILABLE = False
    print("[INFO] CuPy NOT found → everything will run on CPU.")

# CPU UMAP
from umap import UMAP as UMAP_CPU

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
# CPU-only UMAP training
# ============================================================

def run_umap(Z_ref, n_components, n_neighbors, min_dist):

    print("[UMAP] Running on CPU (umap-learn cannot use GPU).")

    # If Z_ref is a CuPy array → convert explicitly
    if CUPY_AVAILABLE and isinstance(Z_ref, cp.ndarray):
        print("[UMAP] Converting CuPy → NumPy for CPU UMAP...")
        Z_ref = cp.asnumpy(Z_ref)

    model = UMAP_CPU(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )

    Z_embed = model.fit_transform(Z_ref)
    return model, Z_embed


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
    model, Z_embed = run_umap(
        Z_ref,
        args.n_components,
        args.n_neighbors,
        args.min_dist
    )

    print(f"[INFO] Saving embedding → {out_embed}")
    np.save(out_embed, Z_embed)

    print(f"[INFO] Saving UMAP model → {out_model}")
    pickle.dump(model, open(out_model, "wb"))

    # Save 2D limits
    if args.n_components == 2:
        xmin = float(Z_embed[:, 0].min())
        xmax = float(Z_embed[:, 0].max())
        ymin = float(Z_embed[:, 1].min())
        ymax = float(Z_embed[:, 1].max())

        limits = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        json.dump(limits, open(out_limits, "w"), indent=4)

        print(f"[INFO] Saved 2D limits → {out_limits}")

    print("\nDone. Global UMAP generated.")


if __name__ == "__main__":
    main()
