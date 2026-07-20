"""Shared helpers for T02 P0-c completion experiments (E1-E4).

Mirrors the exact methodology already validated in capacity_probe_bands.py /
capacity_probe_sweep.py (P0-c), extended for full k-grid + full comma-position
curves. Kept as one module so E1-E4 cannot silently drift from each other.
"""
import numpy as np
import torch


def build_canon(pool, tokenizer, target_n):
    canon = []
    for w in pool:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            canon.append((w, ids[0]))
        if len(canon) >= target_n:
            break
    return canon


def build_trial(spec, tokenizer, rng, block_size=20):
    blocks = []
    for fam in spec["block_families"]:
        pool = next(cp["pool"] for cp in spec["candidate_pools"] if cp["name"] == fam)
        n = spec["targets_per_family"][fam]
        canon = build_canon(pool, tokenizer, n)
        block = rng.sample(canon, block_size)
        blocks.append(block)
    rng.shuffle(blocks)
    return [pair for block in blocks for pair in block]


def sliding_windows(full_band, width):
    lo, hi = min(full_band), max(full_band)
    return {f"{s}-{s + width - 1}": list(range(s, s + width)) for s in range(lo, hi - width + 2)}


def comma_positions_for(words, tok, lm, max_length=2048):
    prompt = ", ".join(w for w, _ in words) + ","
    input_ids = lm.encode(prompt, max_length=max_length)
    ids_list = input_ids[0].tolist()
    comma_tok_positions = [i for i, tid in enumerate(ids_list) if "," in tok.decode([tid])]
    words_local = words
    if len(comma_tok_positions) < len(words_local):
        words_local = words_local[: len(comma_tok_positions)]
    return prompt, input_ids, comma_tok_positions[: len(words_local)], words_local


def exact_ranks_per_layer(lens_logits, layers, word_token_ids):
    """lens_logits[layer]: [n_positions, vocab] CPU tensor (as returned by
    JacobianLens.apply). Returns {layer: np.array[n_positions, n_words]} of
    exact 0-indexed rank of each word's token, via double-argsort."""
    out = {}
    for layer in layers:
        logits = lens_logits[layer]
        order = torch.argsort(logits, dim=-1, descending=True)
        rank_all = torch.argsort(order, dim=-1)
        out[layer] = rank_all[:, word_token_ids].numpy().astype(np.int32)
    return out


def band_curve(layer_word_ranks, band, n_positions, k_max):
    """counts_at_k_by_position[pos_idx][k-1] = #{word_idx<=pos_idx :
    band-min-rank(word_idx) <= k}, k=1..k_max. rank is 0-indexed (matches the
    r<=k convention already used in capacity_probe*.py)."""
    band = [l for l in band if l in layer_word_ranks]
    stacked = np.stack([layer_word_ranks[l] for l in band], axis=0)
    band_min = stacked.min(axis=0)
    k_grid = np.arange(1, k_max + 1)
    curve = []
    for pos_idx in range(n_positions):
        row = np.sort(band_min[pos_idx, : pos_idx + 1])
        counts_at_k = np.searchsorted(row, k_grid, side="right")
        curve.append(counts_at_k.tolist())
    return curve, band


def random_orthogonal_matrix(d_model, generator):
    """Haar-uniform random orthogonal matrix via QR + sign correction
    (Mezzadri 2006, "How to generate random matrices from the classical
    compact groups"). Plain torch.linalg.qr() without the sign fix is NOT
    Haar-uniform."""
    a = torch.randn(d_model, d_model, generator=generator)
    q, r = torch.linalg.qr(a)
    d = torch.diagonal(r)
    ph = d.sign()
    ph[ph == 0] = 1.0
    return q * ph.unsqueeze(0)
