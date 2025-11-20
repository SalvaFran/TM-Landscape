#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_tm_alignments.py
------------------------

Validate TM-Vec predictions by:
  * Downloading reference and neighbor PDBs
  * Running TM-align (reference aligned to neighbor)
  * Extracting aligned reference structures (MODEL 1)
  * Building a multi-chain PDB with all aligned references + WT (centered at 0)
  * Writing CSV and scatterplot comparing predicted vs computed TM-scores
"""

import argparse
import subprocess
import re
import sys
from typing import Optional, Tuple, List
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.append(str(PROJECT_ROOT))
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb}.pdb"


# ---------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ref_pdb_id", required=True)
    p.add_argument("--tm_align_path", default="TMalign")
    return p.parse_args()


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def download_pdb(pdb_id: str, out_path: Path) -> bool:
    url = RCSB_PDB_URL.format(pdb=pdb_id.lower())
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200 and len(r.text) > 100:
            out_path.write_text(r.text)
            print(f"[INFO] Downloaded PDB {pdb_id.upper()} → {out_path.name}")
            return True
        print(f"[WARN] Failed to download {pdb_id.upper()} (HTTP {r.status_code})")
        return False
    except Exception as e:
        print(f"[WARN] Error downloading {pdb_id}: {e}")
        return False


def extract_pdb_from_cath_id(neighbor_id: str) -> Optional[str]:
    match = re.search(r"\|([0-9A-Za-z]{4})[A-Za-z0-9]*", str(neighbor_id))
    return match.group(1).upper() if match else None


# ---------------------------------------------------------------
# TM-align execution
# ---------------------------------------------------------------
def run_tm_align(ref_pdb: Path, target_pdb: Path,
                 tm_align_path: str, out_path: Path) -> Tuple[float, Optional[Path]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = out_path.with_suffix("")

    # Run TM-align (reference first)
    cmd = [tm_align_path, str(ref_pdb), str(target_pdb), "-o", str(prefix)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_path = out_path.with_suffix(".log")
    log_path.write_text(result.stdout)

    if result.returncode != 0:
        print(f"[WARN] TM-align failed for {target_pdb.name}")
        return np.nan, None

    # Extract TM-score normalized by Chain_1 (reference)
    m = re.search(r"TM-score\s*=\s*([0-9.]+)\s*\(if normalized by length of Chain_1", result.stdout)
    if not m:
        m = re.search(r"TM-score\s*=\s*([0-9.]+)", result.stdout)
    tm_score = float(m.group(1)) if m else np.nan

    # Find aligned pdb output
    candidates = [
        prefix.with_name(prefix.name + "_sup.pdb"),
        prefix.with_name(prefix.name + "_all_atm"),
        prefix.with_name(prefix.name + "_all_atm.pdb"),
        prefix.with_name(prefix.name + "_atm"),
        prefix.with_name(prefix.name + "_all"),
        prefix.with_name(prefix.name),
    ]
    found = next((c for c in candidates if c.exists()), None)
    if not found:
        print(f"[WARN] No aligned PDB found for {target_pdb.name}")
        return tm_score, None

    found.replace(out_path)
    return tm_score, out_path


# ---------------------------------------------------------------
# Build reference superpositions (only reference atoms, centered)
# ---------------------------------------------------------------
def build_reference_superpositions(ref_pdb: Path, sup_paths: List[Tuple[str, Path]], out_path: Path):
    """
    Create a single multi-chain PDB:
      - Chain A: WT reference, centered
      - Chains B, C...: reference atoms (MODEL 1) from each aligned *_pair.pdb, centered
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def set_chain_id(lines, chain_id):
        new_lines = []
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                line = line[:21] + chain_id + line[22:]
                new_lines.append(line)
        return new_lines

    def extract_model1_atoms(pdb_path: Path):
        """Extract reference atoms (MODEL 1) from a TM-align output."""
        lines = pdb_path.read_text().splitlines()
        atoms = []
        model = None
        for line in lines:
            if line.startswith("MODEL"):
                try:
                    model = int(line.split()[1])
                except Exception:
                    model = None
            elif line.startswith("ENDMDL"):
                model = None
            elif line.startswith(("ATOM", "HETATM")) and model == 1:
                atoms.append(line)
        if not atoms:
            atoms = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
        return atoms

    # --- Load WT reference ---
    ref_atoms = [l for l in ref_pdb.read_text().splitlines() if l.startswith(("ATOM", "HETATM"))]
    ref_coords = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in ref_atoms])
    ref_center = ref_coords.mean(axis=0)

    def center_atoms(atom_lines, center):
        new_lines = []
        for line in atom_lines:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            x = float(line[30:38]) - center[0]
            y = float(line[38:46]) - center[1]
            z = float(line[46:54]) - center[2]
            new_lines.append(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
        return new_lines

    all_lines = []
    # Chain A: WT reference centered
    all_lines.extend(set_chain_id(center_atoms(ref_atoms, ref_center), "A"))

    # --- Add aligned references ---
    current_chain = ord("B")
    for neighbor_id, sup_path in sup_paths:
        if not sup_path.exists():
            continue
        aligned_atoms = extract_model1_atoms(sup_path)
        if not aligned_atoms:
            continue
        aligned_coords = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in aligned_atoms])
        aligned_center = aligned_coords.mean(axis=0)
        centered = center_atoms(aligned_atoms, aligned_center)
        chain = chr(current_chain)
        all_lines.extend(set_chain_id(centered, chain))
        current_chain += 1
        if current_chain > ord("Z"):
            current_chain = ord("a")

    all_lines.append("END")
    out_path.write_text("\n".join(all_lines) + "\n")
    print(f"[INFO] Wrote reference_superpositions.pdb with {len(sup_paths)+1} chains (centered)")


# ---------------------------------------------------------------
# Expected TM selection
# ---------------------------------------------------------------
def select_expected_tm(summary_df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Aprox_TM_NN_to_WT", "Approx_TM_NN_to_WT", "Approx_TM_NN"]:
        if col in summary_df.columns:
            s = summary_df[["Nearest_ID", col]].rename(columns={col: "Expected_TM"})
            s = s.sort_values(["Nearest_ID", "Expected_TM"], ascending=[True, False])
            return s.drop_duplicates("Nearest_ID")
    raise ValueError("Expected TM column not found.")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    args = parse_args()
    run_dir = PROJECT_ROOT / "runs" / args.run_dir
    data_dir = run_dir / "data"
    figs_dir = run_dir / "figs"
    pdb_dir = run_dir / "structures" / "pdb"
    aligned_dir = run_dir / "structures" / "aligned"

    for d in [data_dir, figs_dir, pdb_dir, aligned_dir]:
        d.mkdir(parents=True, exist_ok=True)

    nn_counts = pd.read_csv(data_dir / "nearest_neighbors_counts.csv")
    summary = pd.read_csv(data_dir / "variants_summary.csv")
    expected = select_expected_tm(summary)
    df_neighbors = nn_counts.merge(expected, on="Nearest_ID", how="left")
    print(f"[INFO] Found {len(df_neighbors)} unique neighbors")

    ref_id = args.ref_pdb_id.upper()
    ref_pdb = pdb_dir / f"{ref_id}.pdb"
    if not ref_pdb.exists():
        if not download_pdb(ref_id, ref_pdb):
            raise RuntimeError(f"Failed to get reference PDB {ref_id}")

    tm_rows, ref_models, keep_names = [], [], []
    for _, row in tqdm(df_neighbors.iterrows(), total=len(df_neighbors)):
        neighbor_id = str(row["Nearest_ID"])
        pdb_id = extract_pdb_from_cath_id(neighbor_id)
        if pdb_id:
            pid = pdb_id.upper()
            pdb_path = pdb_dir / f"{pid}.pdb"
            have = pdb_path.exists() or download_pdb(pid, pdb_path)
        else:
            pid = neighbor_id
            pdb_path = pdb_dir / f"{pid}.pdb"
            have = pdb_path.exists()
        if not have:
            continue

        pair_name = f"{ref_id}_vs_{pid}_pair.pdb"
        pair_out = aligned_dir / pair_name
        tm_score, aligned = run_tm_align(ref_pdb, pdb_path, args.tm_align_path, pair_out)
        if aligned is None:
            continue

        tm_rows.append({
            "Nearest_ID": neighbor_id,
            "PDB_ID": pid,
            "Count": row.get("Count", np.nan),
            "Expected_TM": row.get("Expected_TM", np.nan),
            "Computed_TM": tm_score,
            "Aligned_PDB": pair_name
        })
        ref_models.append((pid, aligned))
        keep_names.append(pair_name)

    if tm_rows:
        df_tm = pd.DataFrame(tm_rows)
        csv_out = data_dir / "computed_tm_scores.csv"
        df_tm.to_csv(csv_out, index=False)
        print(f"[INFO] Saved {csv_out}")

        df_plot = df_tm.dropna(subset=["Expected_TM", "Computed_TM"])
        if not df_plot.empty:
            plt.figure(figsize=(6, 5))
            sns.scatterplot(
                data=df_plot,
                x="Expected_TM",
                y="Computed_TM",
                size="Count",
                sizes=(30, 300),
                alpha=0.8,
                edgecolor="black",
                linewidth=0.3
            )
            plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel("Expected TM-score (predicted vs WT)")
            plt.ylabel("Computed TM-score (TM-align vs WT)")
            plt.title("Predicted vs Computed TM-score per neighbor")
            plt.tight_layout()
            fig_path = figs_dir / "tm_scores_comparison.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print(f"[INFO] Saved figure → {fig_path}")
    else:
        print("[WARN] No valid alignments found.")

    if ref_models:
        ref_super = aligned_dir / "reference_superpositions.pdb"
        build_reference_superpositions(ref_pdb, ref_models, ref_super)

    for f in aligned_dir.iterdir():
        if not f.is_file():
            continue
        if f.name not in keep_names and not f.name.endswith(".log") and f.name != "reference_superpositions.pdb":
            try:
                f.unlink()
            except Exception:
                pass

    print("Done.")


if __name__ == "__main__":
    main()
