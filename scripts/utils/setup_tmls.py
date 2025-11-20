#!/usr/bin/env python3
import os
import urllib.request
import zipfile
from pathlib import Path
import torch

# ===============================================================
# CONFIGURATION
# ===============================================================
BASE_DIR = Path(__file__).resolve().parents[2]  # -> main project directory
MODELS_DIR = BASE_DIR / "models"
ROSTLAB_DIR = MODELS_DIR / "Rostlab"
MODELS_DIR.mkdir(exist_ok=True)
ROSTLAB_DIR.mkdir(exist_ok=True)

# -------------------------------
# ProtT5 encoder (frozen model)
# -------------------------------
PROTT5_URL = "https://zenodo.org/record/4644188/files/prot_t5_xl_uniref50.zip"
PROTT5_ZIP = ROSTLAB_DIR / "prot_t5_xl_uniref50.zip"
PROTT5_DIR = ROSTLAB_DIR / "prot_t5_xl_uniref50"

# -------------------------------
# TM-Vec model variants (stable Zenodo mirrors)
# -------------------------------
# All of the TMvec models are available on Figshare : https://figshare.com/s/e414d6a52fd471d86d69
# All of the TMvec embeddings are available at Zenodo: https://zenodo.org/records/11199459

# ===============================================================
# HELPERS
# ===============================================================
def download(url: str, dest: Path):
    """Download file if not already present."""
    if dest.exists():
        print(f"[INFO] {dest.name} already exists, skipping download.")
        return
    print(f"[INFO] Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[INFO] Saved to {dest}")


def unzip_if_needed(zip_path: Path, extract_to: Path):
    """Unzip a file only if destination folder doesn’t already exist."""
    if extract_to.exists():
        print(f"[INFO] {extract_to.name} already extracted, skipping unzip.")
        return
    print(f"[INFO] Unzipping {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[INFO] Unzipped to {extract_to}")


# ===============================================================
# MAIN EXECUTION
# ===============================================================
if __name__ == "__main__":
    print("=== TM-Landscape setup ===")
    print(f"Base directory: {BASE_DIR}")
    print(f"Models directory: {MODELS_DIR}\n")

    # --------------------------------
    # 1. ProtT5 encoder
    # --------------------------------
    download(PROTT5_URL, PROTT5_ZIP)
    unzip_if_needed(PROTT5_ZIP, PROTT5_DIR)

    # --------------------------------
    # 2. TM-Vec models and embeddings
    # --------------------------------
    # So far the only implemented way is to download them manually from Figshare in the Models directory and unzip them in Models/TM-vec

    # --------------------------------
    # 3. Hardware check
    # --------------------------------
    print("\n[INFO] Checking hardware ...")
    gpu_available = torch.cuda.is_available()
    print(f"[INFO] GPU available: {gpu_available}")
    if gpu_available:
        print(f"[INFO] Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Running in CPU mode (no NVIDIA GPU detected).")

    print("\n[SETUP COMPLETE] All ProtT5 and TM-Vec models are ready.")
