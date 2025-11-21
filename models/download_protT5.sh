#!/bin/bash
set -e

DST="models/Rostlab/prot_t5_xl_uniref50"
ZIP="models/Rostlab/prot_t5_xl_uniref50.zip"
URL="https://zenodo.org/records/4644188/files/prot_t5_xl_uniref50.zip"

echo "[INFO] Checking ProtT5 (Rostlab)..."

# Directory already exists? -> skip
if [ -d "$DST" ]; then
    echo "[INFO] ProtT5 directory already exists → skipping download."
    exit 0
fi

mkdir -p models/Rostlab
cd models/Rostlab

# Download only if zip not present
if [ ! -f "prot_t5_xl_uniref50.zip" ]; then
    echo "[INFO] Downloading ProtT5 from Zenodo..."
    wget -O prot_t5_xl_uniref50.zip "$URL"
else
    echo "[INFO] Found existing prot_t5_xl_uniref50.zip → skipping download."
fi

echo "[INFO] Unpacking ProtT5..."
unzip -n prot_t5_xl_uniref50.zip

echo "[INFO] Done. ProtT5 ready at: $DST"
