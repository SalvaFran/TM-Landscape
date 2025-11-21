
# TM-Landscape

TM-Landscape is a framework for exploring protein sequence–structure manifolds using **TM-Vec embeddings**.

It provides tools to:

- Generate local manifolds of sequence variants (masking, alanine scan, mixed perturbations, deletions).
- Map variants onto a global reference manifold (CATH or SWISS embeddings).
- Train and store a **global 2D UMAP manifold** of the TM-Vec reference space.
- Produce full free‑energy–like landscapes and nearest‑neighbor analyses.

This repository is designed for reproducible and scalable research pipelines based on **TM‑Vec**, **ProtT5**, and **UMAP**.

---

## 1. Repository Structure

```
TM-Landscape/
│
├── data/
│   └── UMAP/                  # Global UMAP results (generated locally)
│
├── models/
│   ├── download_protT5.sh     # Download ProtT5 encoder
│   ├── download_tmvec_cath.sh # Download TM-Vec CATH model + embeddings
│   └── download_all.sh        # Download everything
│
├── scripts/
│   ├── eval/
│   │   ├── run_tmvec_energy.py
│   │   ├── generate_umap.py
│   │   └── ...
│   └── utils/
│       ├── load_tmvec_embeddings.py
│       ├── sample_tmvec_embeddings.py
│       ├── density_estimator.py
│       ├── frequency_estimator.py
│       └── ...
│
├── environment.yml
├── umap_environment.yml
└── README.md
```

Large checkpoint files (ProtT5, TM‑Vec, and CATH embeddings) are **not included in the Git repo**.

---

## 2. Installation

### 2.1. Clone the Repository

```bash
git clone https://github.com/SalvaFran/TM-Landscape.git
cd TM-Landscape
```

---

## 3. Main Conda Environment (CPU / Neutral)

```bash
conda env create -f environment.yml
conda activate tmls
```

This environment supports:

- TM‑Vec
- ProtT5
- Variant generation
- Density & frequency estimators
- Plotting & analysis
- **CPU UMAP**

GPU UMAP **is not included** in this environment.

---

## 4. Optional: GPU‑Accelerated UMAP Environment

A dedicated environment is required because RAPIDS/cuML conflicts with PyTorch 2.2 and Python 3.9.

To enable GPU UMAP:

```bash
conda env create -f umap_environment.yml
conda activate umap_gpu
```

Then run UMAP generation using this env:

```bash
python scripts/eval/generate_umap.py --source cath --size large --n_components 2
```

This produces:

```
data/UMAP/Z_UMAP_2D.npy
data/UMAP/umap_model_2D.pkl
data/UMAP/umap_limits_2D.json
```

---

## 5. Install ProtT5 and TM‑Vec Embeddings and Models

### 5.1 Automatic downloads (public)

```bash
bash models/download_all.sh
```

### 5.2 Manual download (private CATH TM‑Vec model)

Open the private Figshare link:

```
https://figshare.com/s/e414d6a52fd471d86d69
```

Download:

- `tm_vec_cath_model_large.ckpt`
- (optional) `tm_vec_cath_model_large_params.json`

Place them in:

```
models/TM-vec/
```

---

## 6. Running the Main TM‑Landscape Pipeline

Example:

```bash
python scripts/eval/run_tmvec_energy.py     --sequence "MLSDADFKAAVGMTRSAFANLPLWKQQNLKKEKGLF"     --frac_min 0.05 --frac_max 0.10     --n_mask 60 --n_ala 40 --n_mix 40 --n_del 20     --source cath --size large     --outdir Example_Run
```

Output folder:

```
runs/Example_Run/
    data/
    figs/
    energy_landscape_3d.html
```

All plots use **global 2D UMAP axes**, so runs are directly comparable.

---

## 7. Generating a Global UMAP Manifold

Before using the main pipeline, generate the UMAP reference:

```bash
python scripts/eval/generate_umap.py     --source cath     --size large     --n_components 2
```

### GPU Mode
Works automatically if CuPy is installed.

### CPU Fallback
Occurs automatically when CuPy is not available.

## License

MIT License © 2025 Franco Salvatore

---

