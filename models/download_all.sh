#!/usr/bin/env bash

set -e

echo "[INFO] Downloading ProtT5 and TM-Vec CATH embeddings..."
bash models/download_protT5.sh
bash models/download_tmvec_cath.sh

echo "[INFO] All models downloaded successfully."

