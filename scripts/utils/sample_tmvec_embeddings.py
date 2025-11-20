#!/usr/bin/env python3
"""
sample_tmvec_embeddings.py

Generate perturbed versions of an input sequence:
- masking
- alanine mutations
- mix (Ala + Mask)
- deletions

Encode everything with TM-Vec and return all embeddings.

Dependencies:
    pip install torch transformers tm-vec
"""

import random
import torch
import numpy as np
from transformers.models.t5.modeling_t5 import T5EncoderModel
from transformers import T5Tokenizer
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import encode


# ===============================================================
# 1. HELPER FUNCTIONS
# ===============================================================

def random_mask(seq, frac):
    """Randomly mask a fraction of residues in a protein sequence."""
    seq = list(seq)
    n_mask = max(1, int(len(seq) * frac))
    positions = random.sample(range(len(seq)), n_mask)
    for i in positions:
        seq[i] = "X"  # TM-Vec uses 'X' as the mask token
    return "".join(seq)


def random_ala_scan(seq, frac):
    """Randomly mutate a fraction of residues to Alanine."""
    seq = list(seq)
    n_mut = max(1, int(len(seq) * frac))
    positions = random.sample(range(len(seq)), n_mut)
    for i in positions:
        seq[i] = "A"
    return "".join(seq)


def random_mix(seq, frac):
    """Mutate a fraction of residues such that half become Alanine and half are masked."""
    seq = list(seq)
    n_total = max(1, int(len(seq) * frac))
    positions = random.sample(range(len(seq)), n_total)

    half = n_total // 2
    ala_pos = positions[:half]
    mask_pos = positions[half:]

    for i in ala_pos:
        seq[i] = "A"
    for i in mask_pos:
        seq[i] = "X"

    return "".join(seq)


def random_deletions(seq, frac):
    """Delete a consecutive block of residues."""
    seq = list(seq)
    n_total = max(1, int(len(seq) * frac))

    if n_total >= len(seq):
        # Can't delete more than the length
        n_total = len(seq) - 1

    start = random.randint(0, len(seq) - n_total)
    del seq[start:start + n_total]

    return "".join(seq)


# ===============================================================
# 2. LOAD MODELS
# ===============================================================

def load_models(prot_t5_dir, tm_vec_ckpt, tm_vec_json, device=None):
    """
    Load ProtT5 encoder and TM-Vec model from local paths.
    """
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    # Load ProtT5
    print("[INFO] Loading ProtT5 encoder...")
    tokenizer = T5Tokenizer.from_pretrained(prot_t5_dir, do_lower_case=False)
    prot_model = T5EncoderModel.from_pretrained(prot_t5_dir).to(device).eval()

    # Load TM-Vec checkpoint
    print("[INFO] Loading TM-Vec model...")
    tm_config = trans_basic_block_Config.from_json(tm_vec_json)
    tm_model = trans_basic_block.load_from_checkpoint(tm_vec_ckpt, config=tm_config)
    tm_model = tm_model.to(device).eval()

    return tokenizer, prot_model, tm_model, device


# ===============================================================
# 3. MAIN SAMPLER
# ===============================================================

def generate_tmvec_replicates(
    seq,
    tokenizer,
    prot_model,
    tm_model,
    device,
    n_mask=10,
    n_ala=10,
    n_mix=10,
    n_del=10,
    frac_mask_range=(0.01, 0.05),
    frac_ala_range=(0.01, 0.05),
    frac_mix_range=(0.01, 0.05),
    frac_del_range=(0.01, 0.05)
):
    """Generate masked, alanine, mixed and deletion variants, then encode."""

    sequences = {"WT": seq, "masked": [], "ala": [], "mix": [], "del": []}

    # Masked replicas
    for _ in range(n_mask):
        f = random.uniform(*frac_mask_range)
        sequences["masked"].append(random_mask(seq, f))

    # Ala replicas
    for _ in range(n_ala):
        f = random.uniform(*frac_ala_range)
        sequences["ala"].append(random_ala_scan(seq, f))

    # Mix replicas
    for _ in range(n_mix):
        f = random.uniform(*frac_mix_range)
        sequences["mix"].append(random_mix(seq, f))

    # Deletion replicas
    for _ in range(n_del):
        f = random.uniform(*frac_del_range)
        sequences["del"].append(random_deletions(seq, f))

    # Encode everything
    all_seqs = (
        [seq] +
        sequences["masked"] +
        sequences["ala"] +
        sequences["mix"] +
        sequences["del"]
    )

    print(f"[INFO] Encoding {len(all_seqs)} sequences via TM-Vec...")
    embeddings = encode(all_seqs, tm_model, prot_model, tokenizer, device)

    # Slice embeddings
    idx0 = 1
    idx1 = idx0 + n_mask
    idx2 = idx1 + n_ala
    idx3 = idx2 + n_mix
    idx4 = idx3 + n_del

    out = {
        "WT": embeddings[0],
        "masked": list(zip(sequences["masked"], embeddings[idx0:idx1])),
        "ala": list(zip(sequences["ala"], embeddings[idx1:idx2])),
        "mix": list(zip(sequences["mix"], embeddings[idx2:idx3])),
        "del": list(zip(sequences["del"], embeddings[idx3:idx4])),
    }

    return out


# ===============================================================
# 4. WRAPPER FOR SAVING
# ===============================================================

def sample_and_save(
    seq,
    prot_t5_dir,
    tm_vec_ckpt,
    tm_vec_json,
    out_path=None,
    n_mask=10,
    n_ala=10,
    n_mix=10,
    n_del=10,
    frac_mask_range=(0.01, 0.05),
    frac_ala_range=(0.01, 0.05),
    frac_mix_range=(0.01, 0.05),
    frac_del_range=(0.01, 0.05),
):
    """Load models, generate replicas, save embeddings."""
    
    tokenizer, prot_model, tm_model, device = load_models(
        prot_t5_dir, tm_vec_ckpt, tm_vec_json
    )

    results = generate_tmvec_replicates(
        seq, tokenizer, prot_model, tm_model, device,
        n_mask=n_mask,
        n_ala=n_ala,
        n_mix=n_mix,
        n_del=n_del,
        frac_mask_range=frac_mask_range,
        frac_ala_range=frac_ala_range,
        frac_mix_range=frac_mix_range,
        frac_del_range=frac_del_range
    )

    if out_path:
        np.savez(
            out_path,
            WT=results["WT"],
            masked=np.array([e for _, e in results["masked"]]),
            ala=np.array([e for _, e in results["ala"]]),
            mix=np.array([e for _, e in results["mix"]]),
            del_variants=np.array([e for _, e in results["del"]]),
        )
        print(f"[INFO] Saved embeddings → {out_path}.npz")

    return results
