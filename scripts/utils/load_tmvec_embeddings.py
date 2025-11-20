#!/usr/bin/env python3
"""
load_tmvec_embeddings.py
Load the *original* TM-Vec embedding databases (CATH or SWISS, base or large).
"""

import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models" / "TM-vec"

DB_MAP = {
    "cath": {
        "large": {
            "emb": "cath_large.npy",
            "meta": "cath_large_metadata.npy",
        },
        "small": {
            "emb": "cath_small.npy",
            "meta": "cath_small_metadata.npy",
        },
    },
    "swiss": {
        "large": {
            "emb": "swiss_large.npy",
            "meta": "swiss_large_metadata.npy",
        },
        "small": {
            "emb": "swiss_small.npy",
            "meta": "swiss_small_metadata.npy",
        },
    },
}


def load_tmvec_embeddings(source="cath", size="large", frac=1.0, seed=42):
    """
    Load the precomputed TM-Vec embeddings for a chosen database.

    Args:
        source (str): 'cath' or 'swiss'
        size (str): 'small' or 'large'
        frac (float): fraction of total embeddings to randomly sample (0 < frac ≤ 1)
        seed (int): random seed for reproducibility

    Returns:
        tuple (embeddings, metadata)
            embeddings: np.ndarray (N, 512)
            metadata: np.ndarray (N,)
    """
    source, size = source.lower(), size.lower()
    if source not in DB_MAP or size not in DB_MAP[source]:
        raise ValueError(f"Invalid source/size combination: {source}-{size}")

    emb_file = MODELS_DIR / DB_MAP[source][size]["emb"]
    meta_file = MODELS_DIR / DB_MAP[source][size]["meta"]

    if not emb_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            f"Missing embedding files:\n  {emb_file}\n  {meta_file}\n"
            "Download them from the official TM-Vec data archive."
        )

    print(f"[INFO] Loading TM-Vec {source.upper()} ({size}) embeddings...")
    emb = np.load(emb_file, allow_pickle=True)
    meta = np.load(meta_file, allow_pickle=True)

    if not (0 < frac <= 1):
        raise ValueError("frac must be between 0 and 1.")

    if frac < 1.0:
        np.random.seed(seed)
        n_total = emb.shape[0]
        n_sample = int(n_total * frac)
        idx = np.random.choice(n_total, n_sample, replace=False)
        emb = emb[idx]
        meta = meta[idx]
        print(f"[INFO] Subsampled {n_sample}/{n_total} embeddings ({frac*100:.1f}%).")

    print(f"[INFO] Loaded embeddings → shape {emb.shape}\n")
    return emb, meta
