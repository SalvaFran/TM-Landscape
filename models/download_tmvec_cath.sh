#!/bin/bash
set -e

echo "======================================"
echo "  Downloading TM-Vec CATH resources   "
echo "======================================"

BASE="models/TM-vec"
mkdir -p "$BASE"
cd "$BASE"

########################################
# 1. TM-Vec CATH LARGE model (Figshare)
########################################
echo "[INFO] Checking TM-Vec CATH LARGE MODEL (Figshare private links)"

# CKPT
if [ ! -f "tm_vec_cath_model_large.ckpt" ]; then
    echo "[INFO] Downloading tm_vec_cath_model_large.ckpt ..."
    wget -O tm_vec_cath_model_large.ckpt \
    "https://figshare.com/ndownloader/files/49181521?private_link=e414d6a52fd471d86d69"
else
    echo "[OK] tm_vec_cath_model_large.ckpt already exists → skipping."
fi

# PARAMS
if [ ! -f "tm_vec_cath_model_large_params.json" ]; then
    echo "[INFO] Downloading tm_vec_cath_model_large_params.json ..."
    wget -O tm_vec_cath_model_large_params.json \
    "https://figshare.com/ndownloader/files/49181518?private_link=e414d6a52fd471d86d69"
else
    echo "[OK] tm_vec_cath_model_large_params.json already exists → skipping."
fi


########################################
# 2. TM-Vec CATH embeddings (Zenodo)
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
echo "  All CATH model + embeddings ready!  "
echo "  Location: $BASE"
echo "======================================"
