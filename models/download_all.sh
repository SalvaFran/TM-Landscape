#!/bin/bash
set -e

echo "====================================="
echo " TM-Landscape: Download ALL resources"
echo "====================================="

# Ensure we run from project root
cd "$(dirname "$0")/.."

echo "[STEP] Checking ProtT5 (Zenodo)"
bash models/download_protT5.sh

echo "[STEP] Checking TM-Vec CATH model + embeddings"
bash models/download_tmvec_cath.sh

echo "====================================="
echo " All CATH datasets and models ready!"
echo "====================================="
