# TM-Landscape
TM-Landscape is a framework for exploring protein sequence–structure manifolds using TM-Vec embeddings.  
It provides tools to:

- Generate local manifolds of sequence variants (masking, alanine scan, mixed perturbations, deletions).
- Map variants onto a global reference manifold (CATH or SWISS embeddings).
- Generate global UMAP embeddings of the reference space.
- Produce full energy landscapes and nearest-neighbor analyses.

This repository is designed for reproducible research pipelines based on **TM-Vec** and **ProtT5**, with all large models downloaded on demand.

---

## 1. Repository Structure

```
TM-Landscape/
│
├── data/
│   └── UMAP/                  # Stored UMAP projections (generated locally)
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
│       ├── map_tmvec_embeddings.py
│       └── ...
│
├── environment.yml
└── README.md
```

Large model files (ProtT5, TM-Vec checkpoints, and CATH embeddings) are **not** tracked in git and are downloaded locally via scripts.

---

## 2. Installation

### 2.1. Clone the repository

```bash
git clone https://github.com/SalvaFran/TM-Landscape.git
cd TM-Landscape
```

---

## 3. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate tmls
```

---

## 4. Download Required Models

```bash
bash models/download_all.sh
```

This downloads:

### ProtT5 (Rostlab)
```
models/Rostlab/prot_t5_xl_uniref50/
```

### TM-Vec CATH database and model
```
models/TM-vec/tm_vec_cath_model_large.ckpt
models/TM-vec/tm_vec_cath_model_large_params.json
models/TM-vec/cath_large.npy
models/TM-vec/cath_large_metadata.npy
```

All scripts automatically read these paths.

---

## 5. Running the Main Pipeline

Example:

```bash
python scripts/eval/run_tmvec_energy.py     --sequence "MLSDADFKAAVGMTRSAFANLPLWKQQNLKKEKGLF"     --frac_min 0.05     --frac_max 0.10     --n_mask 60     --n_ala 40     --n_mix 40     --n_del 20     --source cath     --size large     --outdir Example_Run
```

Outputs:

```
runs/Example_Run/
    data/
    figs/
```

---

## 6. Global UMAP Generation

```bash
python scripts/eval/generate_umap.py     --source cath     --size large     --n_components 2
```

Outputs saved in:

```
data/UMAP/
```

---

## 7. Troubleshooting

### CUDA/cuML not available
On macOS this is expected. The script falls back to CPU UMAP.

### Missing embedding files
Run:

```bash
bash models/download_all.sh
```

---

## 8. License

MIT License.
