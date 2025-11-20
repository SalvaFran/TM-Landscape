#!/usr/bin/env bash

set -e

echo "[INFO] Creating models/TM-vec directory..."
mkdir -p models/TM-vec
cd models/TM-vec

echo "[INFO] Downloading TM-Vec CATH model checkpoint..."
wget https://zenodo.org/record/11199459/files/tm_vec_cath_model_large.ckpt?download=1 -O tm_vec_cath_model_large.ckpt
wget https://zenodo.org/record/11199459/files/49181521_tm_vec_cath_model_large_params.json?download=1 -O tm_vec_cath_model_large_params.json

echo "[INFO] Downloading TM-Vec CATH embeddings..."
wget https://zenodo.org/records/11199459/files/cath_large.npy?download=1 -O cath_large.npy
wget https://zenodo.org/records/11199459/files/cath_large_metadata.npy?download=1 -O cath_large_metadata.npy

echo "[INFO] Done. CATH embeddings and model saved under models/TM-vec/"

