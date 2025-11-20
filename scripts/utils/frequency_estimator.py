#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frequency_estimator.py
----------------------

Estimate local *log-frequency* of variant embeddings within their own ensemble
using k-nearest neighbors.

Conceptually identical to log-density, but restricted to the local ensemble
of variants generated from a single protein sequence.

For a query point z, let r_k(z) be the distance to its k-th nearest neighbor
among the variant embeddings. Then:

    log_frequency(z) = - D_eff * log(r_k(z) + eps)

Higher log_frequency  -> region with many nearby variants (stable conformation)
Lower log_frequency   -> sparse region (rare conformation)

Parameters
----------
Z_var : (M, D) array of variant embeddings
k : int, number of neighbors (default = 10)
metric : str, distance metric (default = "cosine")
D_eff : int or None, effective dimensionality (defaults to D)
eps : float, small positive floor to avoid log(0)

Returns
-------
log_freq : (M,) array of log-frequency values
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional

def estimate_log_frequency(Z_var,
                           k: int = 10,
                           metric: str = "cosine",
                           D_eff: Optional[int] = None,
                           eps: float = 1e-12) -> np.ndarray:
    if Z_var.ndim != 2:
        raise ValueError("Z_var must be a 2D array of shape (M, D).")

    M, D = Z_var.shape
    if D_eff is None:
        D_eff = D

    if k >= M:
        raise ValueError(f"k must be smaller than number of variants (M={M}).")

    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    nn.fit(Z_var)
    distances, _ = nn.kneighbors(Z_var)
    r_k = distances[:, -1]
    r_k_safe = np.clip(r_k, eps, None)

    log_freq = -float(D_eff) * np.log(r_k_safe)
    return log_freq
