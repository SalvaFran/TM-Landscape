#!/usr/bin/env bash

set -e

echo "[INFO] Creating models/Rostlab directory..."
mkdir -p models/Rostlab
cd models/Rostlab

echo "[INFO] Downloading ProtT5 from Zenodo..."
wget https://zenodo.org/record/4644188/files/prot_t5_xl_uniref50.zip -O prot_t5_xl_uniref50.zip

echo "[INFO] Unzipping ProtT5..."
unzip -o prot_t5_xl_uniref50.zip

echo "[INFO] Done. ProtT5 saved under models/Rostlab/prot_t5_xl_uniref50"

