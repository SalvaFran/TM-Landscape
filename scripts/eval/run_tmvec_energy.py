#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_tmvec_energy.py
-------------------

End-to-end TM-Vec manifold demo with progress bars (tqdm).

Usage example:
    python run_tmvec_energy.py \
        --sequence "MLSDADFKAAVGMTRSAFANLPLWKQQNLKKEKGLF" \
        --frac_min 0.05 --frac_max 0.1 \
        --n_mask 60 --n_ala 40 --n_mix 40 --n_del 20 \
        --source cath --size large \
        --outdir 1VII_test01
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from tqdm.auto import tqdm
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

# Import project modules
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.append(str(PROJECT_ROOT))

from scripts.utils.load_tmvec_embeddings import load_tmvec_embeddings
from scripts.utils.sample_tmvec_embeddings import sample_and_save
from scripts.utils.density_estimator import estimate_log_density
from scripts.utils.frequency_estimator import estimate_log_frequency


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TM-Vec manifold energy landscape demo.")
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--frac_min", type=float, required=True)
    parser.add_argument("--frac_max", type=float, required=True)
    parser.add_argument("--n_mask", type=int, required=True)
    parser.add_argument("--n_ala", type=int, required=True)
    parser.add_argument("--n_mix", type=int, default=0)
    parser.add_argument("--n_del", type=int, default=0)
    parser.add_argument("--source", choices=["cath", "swiss"], required=True)
    parser.add_argument("--size", choices=["small", "large"], required=True)
    parser.add_argument("--outdir", type=str, required=True)
    return parser.parse_args()


# ============================================================
# TM-Vec config helper
# ============================================================

def get_tmvec_config(source: str, size: str):
    source, size = source.lower(), size.lower()
    prot_t5_dir = PROJECT_ROOT / "models" / "Rostlab" / "prot_t5_xl_uniref50"
    tmvec_dir = PROJECT_ROOT / "models" / "TM-vec"

    ckpt_file = f"tm_vec_{source}_model_{size}.ckpt"
    json_file = next(tmvec_dir.glob(f"*{source}_model_{size}*_params.json"), None)

    config = {
        "prot_t5_dir": prot_t5_dir,
        "tm_vec_ckpt": tmvec_dir / ckpt_file,
        "tm_vec_json": json_file or tmvec_dir / f"{source}_{size}_params.json",
    }

    for k, v in config.items():
        if not Path(v).exists():
            raise FileNotFoundError(f"[ERROR] Missing {k}: {v}")
    return config


# ============================================================
# Variant embedding generation
# ============================================================

def generate_variant_embeddings(seq, n_mask, n_ala, n_mix, n_del,
                                frac_min, frac_max,
                                tmvec_cfg, out_dir):
    out_path = out_dir / "embeddings"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating variant embeddings (WT, masked, alanine, mix, del)...")

    results = sample_and_save(
        seq,
        str(tmvec_cfg["prot_t5_dir"]),
        str(tmvec_cfg["tm_vec_ckpt"]),
        str(tmvec_cfg["tm_vec_json"]),
        out_path=str(out_path),
        n_mask=n_mask,
        n_ala=n_ala,
        n_mix=n_mix,
        n_del=n_del,
        frac_mask_range=(frac_min, frac_max),
        frac_ala_range=(frac_min, frac_max),
        frac_mix_range=(frac_min, frac_max),
        frac_del_range=(frac_min, frac_max),
    )

    # ================= FIX: leer bien categorías ===================
    variant_seqs = {
        "WT": [seq],
        "masked": [s for s, _ in results["masked"]],
        "ala":    [s for s, _ in results["ala"]],
        "mix":    [s for s, _ in results["mix"]],
        "del":    [s for s, _ in results["del"]],
    }

    npz_final = out_dir / "embeddings.npz"
    np.savez(
        npz_final,
        WT=results["WT"],
        masked=np.array([e for _, e in results["masked"]]),
        ala=np.array([e for _, e in results["ala"]]),
        mix=np.array([e for _, e in results["mix"]]),
        dele=np.array([e for _, e in results["del"]])   # FIX: así se guardó en sample_and_save
    )

    print(f"[INFO] Saved embeddings → {npz_final}")
    return npz_final, variant_seqs


# ============================================================
# Build variants dataframe
# ============================================================

def build_variants_df(embeddings_npz, Z_ref, metadata, source_label, k_density=200):

    data = np.load(embeddings_npz)

    Z_WT = data["WT"].reshape(1, -1)
    Z_masked = data["masked"]
    Z_ala = data["ala"]
    Z_mix = data["mix"]

    # =============== FIX: leer correctamente deletions ===============
    Z_del = data["dele"]

    variant_sets = {
        "WT": Z_WT,
        "masked": Z_masked,
        "ala": Z_ala,
        "mix": Z_mix,
        "del": Z_del,
    }

    records, query_embeddings = [], []

    print("Collecting variant embeddings...")
    for vtype, Z_var in tqdm(variant_sets.items(), desc="Variant sets"):
        for i in range(Z_var.shape[0]):
            records.append({"Variant_Type": vtype, "Variant_Index": i})
            query_embeddings.append(Z_var[i])

    df = pd.DataFrame(records)
    query_embeddings = np.vstack(query_embeddings)

    # ---- Nearest neighbors ----
    print("Finding nearest neighbors in reference manifold...")
    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(Z_ref)
    distances, indices = nn.kneighbors(query_embeddings)

    df["Nearest_Index"] = indices.ravel()
    df["Cosine_Distance_NN"] = distances.ravel()
    df["Approx_TM_NN"] = 1 - distances.ravel()
    df["Nearest_ID"] = [
        str(metadata[i]) if metadata is not None and i < len(metadata) else f"{source_label}_idx_{i}"
        for i in indices.ravel()
    ]

    # ---- Density ----
    print("Computing global log-density...")
    log_dens = estimate_log_density(Z_ref, query_embeddings, k=k_density, metric="cosine")
    df["Log_Density"] = log_dens

    # ---- Frequency ----
    print("Computing local log-frequency...")
    k_freq = max(5, int(np.sqrt(len(query_embeddings))))
    log_freq = estimate_log_frequency(query_embeddings, k=k_freq, metric="cosine")
    df["Log_Frequency"] = log_freq

    # ---- Energy ----
    df["Log_Density_z"] = (df["Log_Density"] - df["Log_Density"].mean()) / df["Log_Density"].std()
    df["Log_Frequency_z"] = (df["Log_Frequency"] - df["Log_Frequency"].mean()) / df["Log_Frequency"].std()
    df["Energy"] = -(df["Log_Density_z"] + df["Log_Frequency_z"])

    # ---- Distances to WT ----
    orig_vec = Z_WT[0].reshape(1, -1)
    cos_dists = cosine_distances(query_embeddings, orig_vec).ravel()
    df["Cosine_Distance_to_WT"] = cos_dists
    df["Approx_TM_to_WT"] = 1 - cos_dists

    # ---- NN to WT ----
    neighbor_vecs = Z_ref[df["Nearest_Index"].values.ravel()]
    cos_dists_nn = cosine_distances(neighbor_vecs, orig_vec).ravel()
    df["Cosine_Distance_NN_to_WT"] = cos_dists_nn
    df["Approx_TM_NN_to_WT"] = 1 - cos_dists_nn

    return df, query_embeddings


# ============================================================
# Save FASTA
# ============================================================

def save_variant_fasta(variant_seqs, outdir):
    fasta_path = Path(outdir) / "data" / "variants.fasta"
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w") as f:
        for vtype, seqs in variant_seqs.items():
            for i, seq in enumerate(seqs):
                f.write(f">{vtype}_{i}\n{seq}\n")
    print(f"[INFO] Saved FASTA → {fasta_path}")
    return fasta_path


# ============================================================
# Dimensionality reduction
# ============================================================

def compute_tsne_embeddings(Z_ref, query_embeddings, df_variants, random_state=42):

    print("Running 2D t-SNE...")

    n_bg = min(2000, Z_ref.shape[0])
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(Z_ref.shape[0], size=n_bg, replace=False)
    Z_bg = Z_ref[bg_idx]

    nn_idx = df_variants["Nearest_Index"].unique()
    Z_nn = Z_ref[nn_idx]

    Z_all = np.vstack([Z_bg, Z_nn, query_embeddings])
    labels = (
        ["background"] * len(Z_bg)
        + ["neighbor"] * len(Z_nn)
        + df_variants["Variant_Type"].tolist()
    )

    tsne2 = TSNE(n_components=2, perplexity=30, metric="cosine", random_state=random_state)
    coords = tsne2.fit_transform(Z_all)

    df_tsne = pd.DataFrame({"TSNE1": coords[:, 0], "TSNE2": coords[:, 1], "Label": labels})

    # FIX: t-SNE NO produce reaction coordinate → devolvemos None
    return None, df_tsne


# ============================================================
# Plot config
# ============================================================

PALETTE = {
    "WT": "#1f77b4",
    "masked": "#ff7f0e",
    "ala": "#2ca02c",
    "mix": "#9467bd",
    "del": "#e377c2",
    "neighbor": "#111111",
    "background": "#d3d3d3",
}


# ============================================================
# Smooth energy KDE
# ============================================================

def smooth_energy_field(x, y, E, grid_res=80, bw=0.25):
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    xi, yi = np.meshgrid(xi, yi)

    weights = np.exp(-E)
    weights /= np.sum(weights)

    kde = gaussian_kde(np.vstack([x, y]), weights=weights, bw_method=bw)
    p = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

    p_safe = np.clip(p, 1e-12, None)
    E_eff = -np.log(p_safe)
    E_eff -= np.min(E_eff)
    E_eff += np.min(E)

    return xi, yi, E_eff


# ============================================================
# Plotting
# ============================================================

def plot_all(df, df_tsne, figs_dir):

    figs_dir.mkdir(parents=True, exist_ok=True)

    # t-SNE 2D
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    for label, sub in df_tsne.groupby("Label"):
        color = PALETTE.get(label, "#cccccc")
        alpha = 0.25 if label == "background" else 0.7
        size = 10 if label == "background" else 20
        ax.scatter(sub["TSNE1"], sub["TSNE2"], s=size, alpha=alpha, color=color, label=label)
    plt.legend()
    plt.savefig(figs_dir / "tsne_2d_variants.png", dpi=300)
    plt.close()

    # TM-score hist
    plt.figure(figsize=(6, 4))
    sns.histplot(df, x="Approx_TM_NN", hue="Variant_Type",
                 element="step", stat="density", common_norm=False, palette=PALETTE)
    plt.savefig(figs_dir / "tm_score_hist.png", dpi=300)
    plt.close()

    # Boxplot: distance to WT
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        data=df[df["Variant_Type"].isin(["masked", "ala", "mix", "del"])],
        x="Variant_Type", y="Cosine_Distance_to_WT", palette=PALETTE
    )
    plt.savefig(figs_dir / "distance_boxplot.png", dpi=300)
    plt.close()

    # Heatmap neighbor proportions
    df_summary = df.copy()
    variant_types = ["WT", "masked", "ala", "mix", "del"]
    mat = df_summary.pivot_table(
        index="Nearest_ID",
        columns="Variant_Type",
        values="Variant_Index",
        aggfunc="count",
        fill_value=0
    )
    for c in variant_types:
        if c not in mat.columns:
            mat[c] = 0
    mat = mat[variant_types]
    proportions = mat.div(mat.sum(axis=0), axis=1).fillna(0)
    plt.figure(figsize=(10, max(4, 0.4 * proportions.shape[0])))
    sns.heatmap(proportions, annot=True, fmt=".3f", cmap="viridis")
    plt.savefig(figs_dir / "neighbor_variant_proportion_heatmap.png", dpi=300)
    plt.close()

    # TM vs density
    plt.figure(figsize=(6, 4))
    sns.scatterplot(df, x="Approx_TM_NN", y="Log_Density", hue="Variant_Type", palette=PALETTE)
    plt.savefig(figs_dir / "tm_vs_density.png", dpi=300)
    plt.close()

    # 3D energy
    df_var = df_tsne[df_tsne["Label"].isin(["WT", "masked", "ala", "mix", "del"])].copy()
    df_var["Energy"] = df["Energy"].values
    x3, y3, z3 = df_var["TSNE1"], df_var["TSNE2"], df_var["Energy"]
    xi, yi, zi = smooth_energy_field(x3, y3, z3)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=xi, y=yi, z=zi, colorscale="Viridis", opacity=0.45))
    for vtype, sub in df_var.groupby("Label"):
        fig.add_trace(go.Scatter3d(
            x=sub["TSNE1"], y=sub["TSNE2"], z=sub["Energy"],
            mode="markers",
            marker=dict(size=5, color=PALETTE[vtype]),
            name=vtype,
        ))
    fig.write_html(str(figs_dir / "energy_landscape_3d.html"))

    # 2D FEL
    X2, Y2, F2 = smooth_energy_field(x3, y3, z3, grid_res=200, bw=0.20)
    plt.figure(figsize=(8, 7))
    cs = plt.contourf(X2, Y2, F2, levels=14, cmap="turbo")
    plt.scatter(x3, y3, s=10, c="black", alpha=0.3)
    plt.colorbar()
    plt.savefig(figs_dir / "energy_landscape_2d.png", dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    run_root = PROJECT_ROOT / "runs" / args.outdir
    data_dir = run_root / "data"
    figs_dir = run_root / "figs"
    data_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Generating variant embeddings")
    tmvec_cfg = get_tmvec_config(args.source, args.size)

    embeddings_npz, variant_seqs = generate_variant_embeddings(
        args.sequence,
        args.n_mask,
        args.n_ala,
        args.n_mix,
        args.n_del,
        args.frac_min,
        args.frac_max,
        tmvec_cfg,
        data_dir
    )

    print("[2/6] Loading TM-Vec reference manifold")
    Z_ref, metadata = load_tmvec_embeddings(source=args.source, size=args.size, frac=1.0)

    print("[3/6] Computing density and energy metrics")
    df, query_embeddings = build_variants_df(
        embeddings_npz, Z_ref, metadata, f"{args.source}_{args.size}"
    )

    save_variant_fasta(variant_seqs, run_root)

    print("[4/6] Running t-SNE")
    rc, df_tsne = compute_tsne_embeddings(Z_ref, query_embeddings, df)

    # FIX: rc es None → no guardamos RC
    df["Reaction_Coord"] = np.nan

    print("[5/6] Saving data")
    df.to_csv(data_dir / "variants_summary.csv", index=False)
    df_tsne.to_csv(data_dir / "tsne_2d_variants.csv", index=False)

    nn_counts = df["Nearest_ID"].value_counts().reset_index()
    nn_counts.columns = ["Nearest_ID", "Count"]
    nn_counts.to_csv(data_dir / "nearest_neighbors_counts.csv", index=False)

    pd.Series(vars(args)).to_csv(data_dir / "params.tsv", sep="\t", header=False)

    print("[6/6] Generating figures")
    plot_all(df, df_tsne, figs_dir)

    print(f"\nRun completed: {run_root}")


if __name__ == "__main__":
    main()
