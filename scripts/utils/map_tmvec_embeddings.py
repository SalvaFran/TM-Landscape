#!/usr/bin/env python3
"""
map_tmvec_embeddings.py

Provides utilities to:
1. Subsample and project TM-Vec embeddings with t-SNE.
2. Plot 2D/3D projections with optional labels and metadata coloring.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# ===============================================================
# 1. RUN T-SNE AND RETURN/SAVE A DATAFRAME
# ===============================================================
def run_tsne(embeddings, metadata=None, dim=2, sample_frac=1.0, seed=42,
             perplexity=30, save_path=None):
    """
    Run t-SNE on TM-Vec embeddings and optionally save results.

    Args:
        embeddings (np.ndarray): TM-Vec embeddings (N, 512)
        metadata (list/np.ndarray | None): Optional labels (e.g., CATH or SWISS names)
        dim (int): Target t-SNE dimensions (2 or 3)
        sample_frac (float): Fraction of points to randomly sample (0 < f ≤ 1)
        seed (int): Random seed
        perplexity (int): t-SNE perplexity
        save_path (str | Path | None): Optional CSV output path

    Returns:
        pd.DataFrame with columns ['Dim1', 'Dim2', ('Dim3'), 'Metadata']
    """
    np.random.seed(seed)
    n = embeddings.shape[0]
    k = int(sample_frac * n)
    idx = np.random.choice(n, k, replace=False)
    X = embeddings[idx]
    meta = np.array(metadata)[idx] if metadata is not None else np.array([None] * k)

    print(f"[INFO] Running t-SNE on {k}/{n} samples (dim={dim})...")
    tsne = TSNE(
        n_components=dim,
        learning_rate="auto",
        init="random",
        perplexity=perplexity,
        random_state=seed,
    )
    mapped = tsne.fit_transform(X)

    df = pd.DataFrame(mapped, columns=[f"Dim{i+1}" for i in range(dim)])
    df["Metadata"] = meta

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[INFO] Saved t-SNE results → {save_path}")

    return df


# ===============================================================
# 2. PLOT 2D/3D T-SNE MAPS
# ===============================================================
def plot_tsne(df, dim=2, color_by="metadata", show_labels=False,
              title=None, save_path=None):
    """
    Visualize a t-SNE DataFrame.

    Args:
        df (pd.DataFrame): Must contain ['Dim1','Dim2',('Dim3'),'Metadata']
        dim (int): 2 or 3 for 2D/3D plot
        color_by (str): 'metadata' | 'none' (no color)
        show_labels (bool): Whether to annotate points with metadata
        title (str | None): Plot title
        save_path (str | Path | None): Optional file path to save plot (PNG or HTML)
    """
    title = title or f"TM-Vec t-SNE ({dim}D)"
    if dim not in [2, 3]:
        raise ValueError("dim must be 2 or 3")

    # -----------------------------
    # 2D PLOT
    # -----------------------------
    if dim == 2:
        plt.figure(figsize=(8, 7))
        if color_by == "metadata" and "Metadata" in df.columns:
            sns.scatterplot(data=df, x="Dim1", y="Dim2", hue="Metadata",
                            palette="tab10", s=25, linewidth=0)
        else:
            plt.scatter(df["Dim1"], df["Dim2"], s=10, color="grey")

        if show_labels and "Metadata" in df.columns:
            for _, row in df.iterrows():
                plt.text(row["Dim1"], row["Dim2"], str(row["Metadata"]),
                         fontsize=6, alpha=0.6)

        plt.title(title)
        plt.xlabel("Dim1")
        plt.ylabel("Dim2")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"[INFO] Saved 2D plot → {save_path}")
        plt.show()

    # -----------------------------
    # 3D PLOT
    # -----------------------------
    elif dim == 3:
        if color_by == "metadata" and "Metadata" in df.columns:
            fig = px.scatter_3d(df, x="Dim1", y="Dim2", z="Dim3",
                                color="Metadata", title=title)
        else:
            fig = px.scatter_3d(df, x="Dim1", y="Dim2", z="Dim3",
                                color_discrete_sequence=["grey"], title=title)

        if show_labels and "Metadata" in df.columns:
            fig.update_traces(text=df["Metadata"], textposition="top center")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            print(f"[INFO] Saved 3D HTML plot → {save_path}")
        fig.show()

def plot_tsne_overlay(df, title="T-SNE landscape", save_path=None):
    """
    Plot 2D or 3D T-SNE results from a DataFrame with columns ['Dim1','Dim2',(optional 'Dim3'),'Type'].

    Features:
    - Automatically detects dimensionality (2D vs 3D)
    - Colors by 'Type' with distinct styling
    - Ensures 'WT embeddings' are plotted first (in the background)
    - Clean white layout and publication-style aesthetics
    """

    # --- Detect dimensionality ---
    dim_cols = [c for c in ["Dim1", "Dim2", "Dim3"] if c in df.columns]
    n_dim = len(dim_cols)

    if n_dim not in (2, 3):
        print("[WARN] No valid TSNE columns found (Dim1, Dim2, Dim3). Nothing plotted.")
        return

    # --- Color and style mapping ---
    color_map = {
        "WT_embeddings": "rgba(0,0,0,0.25)",
        "masked": "rgba(220,0,0,0.85)",
        "ala": "rgba(0,80,255,0.85)",
        "new": "rgba(255,0,0,0.85)"
    }
    size_map = {
        "WT_embeddings": 5,
        "masked": 7,
        "ala": 7,
        "new": 8
    }
    line_map = {
        "WT_embeddings": dict(width=0),
        "masked": dict(color="white", width=0.6),
        "ala": dict(color="white", width=0.6),
        "new": dict(color="black", width=1.0)
    }

    # --- Layer order: WTs first, then everything else ---
    type_order = []
    if "WT_embeddings" in df["Type"].unique():
        type_order.append("WT_embeddings")
    type_order += [t for t in sorted(df["Type"].unique()) if t != "WT_embeddings"]

    fig = go.Figure()

    # --- Add traces in the desired order ---
    for t in type_order:
        df_sub = df[df["Type"] == t]
        if df_sub.empty:
            continue

        if n_dim == 2:
            fig.add_trace(go.Scatter(
                x=df_sub["Dim1"], y=df_sub["Dim2"],
                mode="markers",
                name=t,
                marker=dict(
                    color=color_map.get(t, "gray"),
                    size=size_map.get(t, 6),
                    line=line_map.get(t, dict(width=0))
                ),
                hoverinfo="skip"
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=df_sub["Dim1"], y=df_sub["Dim2"], z=df_sub["Dim3"],
                mode="markers",
                name=t,
                marker=dict(
                    color=color_map.get(t, "gray"),
                    size=size_map.get(t, 5),
                    line=line_map.get(t, dict(width=0))
                ),
                hoverinfo="skip"
            ))

    # --- Layout aesthetics ---
    layout_base = dict(
        title=title,
        template="plotly_white",
        showlegend=True,
        width=850,
        height=700,
        legend=dict(
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=0.6,
            font=dict(size=12)
        )
    )

    if n_dim == 2:
        layout_base.update({
            "xaxis": dict(showgrid=False, zeroline=False),
            "yaxis": dict(showgrid=False, zeroline=False)
        })
    else:
        layout_base.update({
            "scene": dict(
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
                zaxis=dict(showgrid=False, zeroline=False)
            )
        })

    fig.update_layout(**layout_base)

    if save_path:
        fig.write_html(str(save_path))
        print(f"[INFO] Saved interactive plot → {save_path}")

    fig.show()