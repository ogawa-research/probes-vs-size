"""T02 B5 Job1 diagnostic v2: reproduce exactly r_delta_probe.py's
measure_neutral() path for gemma-3-4b (build_token_pool, 4 sequences,
lengths [1024,2048,4096], same AttnStatsHook logic) but additionally dump
raw per-position stats for ONE layer (layer 0) at length 2048 to see why
n_valid_per_head can be 256/256 (all "finite" per the isfinite check) while
H_mean ends up nan.
"""
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/t02")
sys.path.insert(0, "/workspace/t02/jacobian-lens")
from jlens.examples import load_wikitext_prompts  # noqa: E402

CLIP_FLOOR = 1e-45
HF_NAME = "google/gemma-3-4b-pt"


def build_token_pool(tokenizer, min_tokens):
    n_prompts = 60
    ids = []
    while len(ids) < min_tokens:
        prompts = load_wikitext_prompts(n_prompts, min_chars=600)
        text = "\n\n".join(prompts)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < min_tokens:
            n_prompts *= 2
    return ids


def get_query_positions(seq_len, n_positions=64, min_pos=32):
    hi = seq_len - 1
    raw = np.logspace(np.log10(min_pos), np.log10(hi), n_positions)
    positions = np.unique(np.round(raw).astype(int))
    positions = positions[(positions >= min_pos) & (positions <= hi)]
    return positions.tolist()


captured = {}


def make_hook(layer_idx, query_positions):
    def hook(module, inputs, output):
        if layer_idx != 0:
            return
        aw = output[1][0]  # [n_heads, seq_len, seq_len]
        for pos in query_positions[:3] + query_positions[-3:]:
            if pos >= aw.shape[1]:
                continue
            row = aw[:, pos, : pos + 1]  # [n_heads, pos+1]
            finite = torch.isfinite(row).all(dim=-1)
            row_min = row.min(dim=-1).values
            row_max = row.max(dim=-1).values
            row_sum = row.sum(dim=-1)
            row_has_neg = (row < 0).any(dim=-1)
            clamped = row.clamp(min=CLIP_FLOOR)
            logrow = clamped.log()
            H = -(clamped * logrow).sum(dim=-1)
            key = f"pos{pos}"
            captured.setdefault(key, []).append(
                {
                    "finite_per_head": finite.tolist(),
                    "row_min_per_head": row_min.float().tolist(),
                    "row_max_per_head": row_max.float().tolist(),
                    "row_sum_per_head": row_sum.float().tolist(),
                    "row_has_neg_per_head": row_has_neg.tolist(),
                    "H_per_head": H.float().tolist(),
                }
            )
    return hook


def main():
    model = AutoModelForCausalLM.from_pretrained(
        HF_NAME, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
    model.eval()
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers

    pool = build_token_pool(tokenizer, 1024 * 5)
    n = 2048
    qpos = get_query_positions(n)
    print(f"query_positions sample (first3+last3): {qpos[:3] + qpos[-3:]}", flush=True)

    handles = []
    for li, layer in enumerate(layers):
        h = make_hook(li, qpos)
        handles.append(layer.self_attn.register_forward_hook(h))

    try:
        with torch.no_grad():
            for seq_i in range(4):
                seq_ids = pool[seq_i * n : (seq_i + 1) * n]
                input_ids = torch.tensor([seq_ids], dtype=torch.long, device="cuda")
                model(input_ids=input_ids)
                print(f"seq {seq_i+1}/4 forward done", flush=True)
    finally:
        for h in handles:
            h.remove()

    import json
    with open("/workspace/t02/out/diag2_layer0_raw.json", "w") as f:
        json.dump(captured, f, indent=2)
    print("=== SUMMARY (layer 0) ===", flush=True)
    for key, entries in captured.items():
        for i, e in enumerate(entries):
            print(f"{key} seq{i}: finite={e['finite_per_head']} min={e['row_min_per_head']} "
                  f"max={e['row_max_per_head']} sum={e['row_sum_per_head']} "
                  f"has_neg={e['row_has_neg_per_head']} H={e['H_per_head']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
