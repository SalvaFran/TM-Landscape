#!/bin/bash
set -e

echo "======================================"
echo "  Downloading TM-Vec CATH resources   "
echo "======================================"

BASE="models/TM-vec"
mkdir -p "$BASE"
cd "$BASE"

########################################
# TM-Vec CATH embeddings (Zenodo)
########################################
echo "[INFO] Checking TM-Vec CATH embeddings (Zenodo)"

ZENODO_URL="https://zenodo.org/records/11199459/files"

declare -A EMB_FILES=(
    ["cath_large.npy"]="cath_large.npy"
    ["cath_large_metadata.npy"]="cath_large_metadata.npy"
)

for fname in "${!EMB_FILES[@]}"; do
    if [ -f "$fname" ]; then
        echo "[OK] $fname already exists → skipping."
    else
        echo "[INFO] Downloading $fname from Zenodo ..."
        wget -O "$fname" "$ZENODO_URL/$fname?download=1"
    fi
done


########################################
# Done
########################################
echo "======================================"
echo "  CATH embeddings ready!  "
echo "  Location: $BASE"
echo "======================================"
