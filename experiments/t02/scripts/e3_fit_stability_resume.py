"""T02 P0-c completion E3 (RESUME): the first full run of e3_fit_stability.py
was silently killed (no Python traceback in the log -- consistent with an
OS-level SIGKILL / cgroup OOM-kill, matching the pattern already seen once on
this shared compute instance for another project's earlier pilot process) after completing fit 1/3
(n500_seed0, saved) and reaching 150/500 prompts into fit 2/3 (n500_seed1,
checkpoint on disk). This script resumes rather than redoing finished work:

  - n500_seed0: loads the already-saved JacobianLens instead of re-fitting.
  - n500_seed1: calls fit() with the existing checkpoint_path; jlens.fitting.fit()
    auto-resumes from its own "next_idx" when resume=True (default) and the
    checkpoint exists.
  - n250_seed0: fresh (no prior checkpoint existed for this one).

Everything else (band rule, capacity measurement, pairwise cosine, output
format) is unchanged from e3_fit_stability.py.
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
ALREADY_DONE = {"n500_seed0"}  # completed + saved in the killed run; load instead of re-fit
CAPACITY_SEEDS = [0, 1, 2]
K_MAX = 128
OUT_PATH = "/workspace/t02/fit_stability_qwen06b.json"


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

    cka_data = json.load(open(CKA_PATH))
    band_result = compute_band(cka_data["cka_matrix"], n_layers_for_floor=lm.n_layers - 1)
    band = band_result["band"]
    print(f"0.6B band-rule result: {band_result}", flush=True)

    pool = load_wikitext_prompts(POOL_SIZE, min_chars=600)
    print(f"loaded pool of {len(pool)} wikitext prompts", flush=True)

    fits = {}
    fit_meta = {}
    for cfg in FIT_CONFIGS:
        name = cfg["name"]
        ckpt = f"/workspace/t02/e3_fit_{name}_ckpt.pt"
        lens_path = f"/workspace/t02/e3_fit_{name}_lens.pt"
        if name in ALREADY_DONE:
            ts = time.time()
            lens = JacobianLens.load(lens_path)
            fits[name] = lens
            fit_meta[name] = {
                "seed": cfg["seed"],
                "n_prompts_requested": cfg["n_prompts"],
                "n_prompts_fitted": lens.n_prompts,
                "elapsed_s": None,
                "peak_gpu_gb": None,
                "resumed_from_prior_killed_run": True,
                "note": "loaded from disk (fit completed before the first run was killed); "
                        "elapsed_s/peak_gpu_gb are from that run's log: 3427.2s, 13.88GB",
            }
            print(f"fit[{name}] loaded from disk in {time.time()-ts:.1f}s ({lens.n_prompts} prompts)", flush=True)
            continue

        ts = time.time()
        prompts = select_prompts(pool, cfg["seed"], cfg["n_prompts"])
        resumed = False
        import os
        if os.path.exists(ckpt):
            resumed = True
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            print(f"fit[{name}]: resuming from checkpoint, n_done={state['n_done']}, "
                  f"next_idx={state['next_idx']}/{len(prompts)}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        lens = fit(
            lm, prompts,
            dim_batch=64, max_seq_len=128,
            checkpoint_path=ckpt, checkpoint_every=50,
        )
        elapsed = time.time() - ts
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        fits[name] = lens
        fit_meta[name] = {
            "seed": cfg["seed"],
            "n_prompts_requested": cfg["n_prompts"],
            "n_prompts_fitted": lens.n_prompts,
            "elapsed_s": elapsed,
            "peak_gpu_gb": peak_gb,
            "resumed_from_prior_killed_run": resumed,
        }
        lens.save(lens_path)
        print(f"fit[{name}] done in {elapsed:.1f}s ({lens.n_prompts} prompts fitted), peak {peak_gb:.2f}GB "
              f"(resumed={resumed})", flush=True)

    names = list(fits.keys())
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}_vs_{names[j]}"
            pairwise[key] = cosine_sim_per_layer(fits[names[i]], fits[names[j]])
            mean_sim = float(np.mean(list(pairwise[key].values())))
            print(f"cosine[{key}] mean-over-layers={mean_sim:.4f}", flush=True)

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
        "smoke_mode": False,
        "incident_note": "first run of e3_fit_stability.py was silently killed (no traceback -- "
                          "consistent with cgroup/OOM SIGKILL under 3-way shared-instance GPU/CPU contention "
                          "from two co-located projects) after completing fit 1/3 and reaching 150/500 into fit 2/3; "
                          "this run resumed via jlens.fitting.fit()'s built-in checkpoint resume plus loading "
                          "the already-saved fit 1/3 lens from disk, rather than redoing completed work.",
        "fit_meta": fit_meta,
        "pairwise_cosine_similarity_per_layer": pairwise,
        "capacity": capacity,
        "elapsed_s": time.time() - t0,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    print(f"total time: {time.time()-t0:.1f}s -> {OUT_PATH}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
