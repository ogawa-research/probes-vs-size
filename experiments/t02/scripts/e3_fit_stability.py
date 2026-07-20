"""T02 P0-c completion E3: J-lens fit stability (Qwen3-0.6B).

3 fits: {seed=0,n=500}, {seed=1,n=500}, {seed=0,n=250} (the n=250 draw is a
prefix of the n=500/seed=0 draw -- see select_prompts -- so the 250-vs-500
comparison isolates prompt-count effects from selection effects).

jlens.fitting.fit() takes no seed argument (Jacobian estimation is
deterministic given a fixed prompt list; load_wikitext_prompts() itself is
also unseeded, always returning the same first-N-over-threshold prompts).
"Fit seed" is therefore implemented here as: load a pool of 1000 WikiText
prompts once, then shuffle-and-slice with random.Random(seed) to choose which
prompts feed each fit. This is a mechanical choice to give fit() a seed
concept it doesn't natively have -- recorded here, not a design decision.

For each fit: (i) pairwise J-matrix cosine similarity per layer across all 3
fits, (ii) capacity measurement (band = machine-rule output applied to the
existing cka_band_qwen06b.json, k=1..128, full position curves, word-list
seeds 0..2).

Mechanical execution only -- see t02_p0c_completion_instructions_2026-07-10.md.
"""
import json
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import JacobianLens, fit, from_hf
from jlens.examples import load_wikitext_prompts

sys.path.insert(0, "/workspace/t02")
from band_rule import compute_band  # noqa: E402
from t02_common import band_curve, build_trial, exact_ranks_per_layer  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"
CAPACITY_SPEC = "/workspace/t02/jacobian-lens/data/experiments/capacity.json"
CKA_PATH = "/workspace/t02/cka_band_qwen06b.json"
POOL_SIZE = 1000
FIT_CONFIGS = [
    {"name": "n500_seed0", "seed": 0, "n_prompts": 500},
    {"name": "n500_seed1", "seed": 1, "n_prompts": 500},
    {"name": "n250_seed0", "seed": 0, "n_prompts": 250},
]
CAPACITY_SEEDS = [0, 1, 2]
K_MAX = 128
OUT_PATH = "/workspace/t02/fit_stability_qwen06b.json"

SMOKE = "--smoke" in sys.argv
if SMOKE:
    FIT_CONFIGS = [{"name": "n500_seed0", "seed": 0, "n_prompts": 8}]
    CAPACITY_SEEDS = [0]


def select_prompts(pool, seed, n_prompts):
    perm = list(range(len(pool)))
    random.Random(seed).shuffle(perm)
    return [pool[i] for i in perm[:n_prompts]]


def cosine_sim_per_layer(lens_a, lens_b):
    sims = {}
    for layer in lens_a.source_layers:
        a = lens_a.jacobians[layer].flatten().float()
        b = lens_b.jacobians[layer].flatten().float()
        sims[layer] = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return sims


def run_capacity(lm, lens, tok, spec, seed, band):
    rng = random.Random(seed)
    words = build_trial(spec, tok, rng)
    prompt = ", ".join(w for w, _ in words) + ","
    input_ids = lm.encode(prompt, max_length=2048)
    ids_list = input_ids[0].tolist()
    comma_tok_positions = [i for i, tid in enumerate(ids_list) if "," in tok.decode([tid])]
    if len(comma_tok_positions) < len(words):
        words = words[: len(comma_tok_positions)]
    n_positions = len(comma_tok_positions[: len(words)])
    word_token_ids = torch.tensor([tid for _, tid in words], dtype=torch.long)
    lens_logits, _, _ = lens.apply(lm, prompt, layers=band, positions=comma_tok_positions[: len(words)])
    layer_word_ranks = exact_ranks_per_layer(lens_logits, band, word_token_ids)
    curve, used_band = band_curve(layer_word_ranks, band, n_positions, K_MAX)
    return {"n_positions": n_positions, "band": used_band, "counts_at_k_by_position": curve}


def main():
    t0 = time.time()
    spec = json.load(open(CAPACITY_SPEC))

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    lm = from_hf(model, tok)
    print(f"model loaded in {time.time()-t0:.1f}s, n_layers={lm.n_layers}, d_model={lm.d_model}", flush=True)

    # band via machine rule applied to the existing P0-c CKA matrix (no new CKA run).
    cka_data = json.load(open(CKA_PATH))
    band_result = compute_band(cka_data["cka_matrix"], n_layers_for_floor=lm.n_layers - 1)
    band = band_result["band"]
    print(f"0.6B band-rule result: {band_result}", flush=True)

    pool = load_wikitext_prompts(POOL_SIZE, min_chars=600)
    print(f"loaded pool of {len(pool)} wikitext prompts", flush=True)

    fits = {}
    fit_meta = {}
    for cfg in FIT_CONFIGS:
        ts = time.time()
        prompts = select_prompts(pool, cfg["seed"], cfg["n_prompts"])
        ckpt = f"/workspace/t02/e3_fit_{cfg['name']}_ckpt.pt"
        torch.cuda.reset_peak_memory_stats()
        lens = fit(
            lm, prompts,
            dim_batch=64, max_seq_len=128,
            checkpoint_path=ckpt, checkpoint_every=50,
        )
        elapsed = time.time() - ts
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        fits[cfg["name"]] = lens
        fit_meta[cfg["name"]] = {
            "seed": cfg["seed"],
            "n_prompts_requested": cfg["n_prompts"],
            "n_prompts_fitted": lens.n_prompts,
            "elapsed_s": elapsed,
            "peak_gpu_gb": peak_gb,
        }
        lens.save(f"/workspace/t02/e3_fit_{cfg['name']}_lens.pt")
        print(f"fit[{cfg['name']}] done in {elapsed:.1f}s ({lens.n_prompts} prompts fitted), peak {peak_gb:.2f}GB", flush=True)

    # pairwise cosine similarity, all combinations
    names = list(fits.keys())
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}_vs_{names[j]}"
            pairwise[key] = cosine_sim_per_layer(fits[names[i]], fits[names[j]])
            mean_sim = float(np.mean(list(pairwise[key].values())))
            print(f"cosine[{key}] mean-over-layers={mean_sim:.4f}", flush=True)

    # capacity measurement per fit, per seed
    capacity = {}
    for name, lens in fits.items():
        capacity[name] = {}
        for seed in CAPACITY_SEEDS:
            ts = time.time()
            capacity[name][f"seed{seed}"] = run_capacity(lm, lens, tok, spec, seed, band)
            print(f"capacity[{name}, seed{seed}] done in {time.time()-ts:.1f}s", flush=True)
        with open(OUT_PATH.replace(".json", ".ckpt.json"), "w") as f:
            json.dump({"fit_meta": fit_meta, "pairwise_cosine": pairwise, "capacity_so_far": capacity}, f, separators=(",", ":"))

    output = {
        "model": MODEL_ID,
        "pool_size": POOL_SIZE,
        "fit_configs": FIT_CONFIGS,
        "band_rule_result": band_result,
        "capacity_seeds": CAPACITY_SEEDS,
        "k_max": K_MAX,
        "smoke_mode": SMOKE,
        "fit_meta": fit_meta,
        "pairwise_cosine_similarity_per_layer": pairwise,
        "capacity": capacity,
        "elapsed_s": time.time() - t0,
    }
    out_path = OUT_PATH if not SMOKE else OUT_PATH.replace(".json", ".smoke.json")
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    print(f"total time: {time.time()-t0:.1f}s -> {out_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
