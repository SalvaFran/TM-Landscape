#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
density_estimator.py
--------------------

Estimate local *log-density* of query embeddings within a reference latent space
using k-nearest neighbors.

For a query point z, let r_k(z) be the distance to its k-th nearest neighbor
(in the reference set, under the chosen metric). We define:

    log_density(z) = - D_eff * log(r_k(z) + eps)

This is proportional to log(ρ(z)) up to an additive constant, assuming
ρ(z) ~ 1 / r_k(z)^{D_eff}. We work in log-space for numerical stability.

- Higher log_density  -> region more populated -> lower effective energy
- Lower log_density   -> region sparser        -> higher effective energy

You can define:
    energy(z) = -log_density(z)

Parameters
----------
Z_ref : (N, D) reference embeddings
Z_query : (M, D) query embeddings
k : number of neighbors (k >= 2 recommended)
metric : distance metric ("cosine" recommended for TM-Vec)
D_eff : effective dimensionality; if None, uses D
eps : small positive floor to avoid log(0)

Returns
-------
log_densities : (M,) array of log-density values
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional

def estimate_log_density(Z_ref,
                         Z_query,
                         k: int = 100,
                         metric: str = "cosine",
                         D_eff: Optional[int] = None,
                         eps: float = 1e-12) -> np.ndarray:
    """
    Estimate local log-density for each query embedding.

    log_density(z) = - D_eff * log(r_k(z) + eps)

    Notes:
    - Use the same k for all points to compare densities.
    - D_eff can be smaller than the embedding dimension if you believe
      the manifold has lower intrinsic dimensionality.

    """
    if Z_ref.ndim != 2 or Z_query.ndim != 2:
        raise ValueError("Z_ref and Z_query must be 2D arrays of shape (N,D) and (M,D).")

    N, D = Z_ref.shape
    if D_eff is None:
        D_eff = D

    if k < 2:
        raise ValueError("k must be >= 2 to obtain a meaningful local scale.")

    if k > N:
        raise ValueError(f"k={k} is larger than reference size N={N}.")

    # Fit k-NN on reference
    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    nn.fit(Z_ref)

    # Distances to k nearest neighbors for each query
    distances, _ = nn.kneighbors(Z_query)
    # r_k: distance to the k-th neighbor (local scale)
    r_k = distances[:, -1]

    # Safe floor to avoid log(0); we are in a very high-D space
    r_k_safe = np.clip(r_k, eps, None)

    # log-density (relative, up to additive constant)
    log_density = -float(D_eff) * np.log(r_k_safe)

    return log_density
