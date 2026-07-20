"""T02 P0-c completion E1: full k-grid x comma-position-curve x seed-variance
capacity measurement (Qwen3-1.7B, machine-rule band 14-22 primary + 19-window
9-layer sweep as post-processing).

Mechanical execution only -- see t02_p0c_completion_instructions_2026-07-10.md.
Reuses the "one lens.apply() over the full layer range, slice into bands
after" pattern already validated in P0-c (capacity_probe_bands.py / _sweep.py).

Usage: python3 e1_capacity_full_grid.py [--smoke]
  --smoke restricts to seed 0 only (no determinism recheck), for a quick
  correctness/timing check before committing to the full 10-seed run.
"""
import json
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import JacobianLens, from_hf

sys.path.insert(0, "/workspace/t02")
from band_rule import compute_band  # noqa: E402
from t02_common import band_curve, build_trial, exact_ranks_per_layer, sliding_windows  # noqa: E402

MODEL_ID = "Qwen/Qwen3-1.7B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"
CAPACITY_SPEC = "/workspace/t02/jacobian-lens/data/experiments/capacity.json"
SEEDS = list(range(10))
K_MAX = 128
WINDOW_WIDTH = 9
MAIN_BAND = list(range(14, 23))  # machine-rule output, design v1.4 S11'''-2
OUT_PATH = "/workspace/t02/capacity_full_grid_qwen17b.json"
CKPT_PATH = "/workspace/t02/capacity_full_grid_qwen17b.ckpt.json"

SMOKE = "--smoke" in sys.argv


def run_one(lm, lens, tok, spec, seed, full_band):
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

    lens_logits, _, _ = lens.apply(lm, prompt, layers=full_band, positions=comma_tok_positions[: len(words)])
    layer_word_ranks = exact_ranks_per_layer(lens_logits, full_band, word_token_ids)

    return {
        "n_positions": n_positions,
        "seq_len": int(input_ids.shape[1]),
        "layer_word_ranks": layer_word_ranks,
    }


def main():
    t0 = time.time()
    spec = json.load(open(CAPACITY_SPEC))

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    lm = from_hf(model, tok)
    print(f"model loaded in {time.time()-t0:.1f}s, n_layers={lm.n_layers}, d_model={lm.d_model}", flush=True)

    lens = JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    full_band = sorted(lens.source_layers)
    print(f"lens loaded, source_layers={full_band}", flush=True)

    # Sanity print only (MAIN_BAND is the design's already-fixed machine-rule
    # output; this re-derives it from the P0-c CKA matrix as a cross-check).
    try:
        cka_data = json.load(open("/workspace/t02/cka_band_qwen17b.json"))
        rule_check = compute_band(cka_data["cka_matrix"], n_layers_for_floor=len(full_band))
        print(f"band_rule sanity check (should match main_band {MAIN_BAND}): {rule_check['band']}", flush=True)
    except Exception as exc:
        print(f"band_rule sanity check skipped: {exc}", flush=True)

    sweep_bands = sliding_windows(full_band, WINDOW_WIDTH)
    all_bands = {"main_14_22": MAIN_BAND, **sweep_bands}
    print(f"n_sweep_bands={len(sweep_bands)}, total_bands={len(all_bands)}", flush=True)

    seeds_to_run = [0] if SMOKE else SEEDS
    per_seed_results = {}
    for seed in seeds_to_run:
        ts = time.time()
        data = run_one(lm, lens, tok, spec, seed, full_band)
        seed_result = {"n_positions": data["n_positions"], "seq_len": data["seq_len"], "bands": {}}
        for band_name, band in all_bands.items():
            curve, used_band = band_curve(data["layer_word_ranks"], band, data["n_positions"], K_MAX)
            seed_result["bands"][band_name] = {"band": used_band, "counts_at_k_by_position": curve}
        per_seed_results[f"seed{seed}"] = seed_result
        print(f"seed {seed} done in {time.time()-ts:.1f}s (n_positions={data['n_positions']})", flush=True)
        with open(CKPT_PATH, "w") as f:
            json.dump({"per_seed": per_seed_results, "seeds_done": list(per_seed_results)}, f, separators=(",", ":"))

        if seed == 0:
            last = seed_result["bands"]["main_14_22"]["counts_at_k_by_position"][-1]
            known = {8: 5, 13: 6, 21: 8, 34: 9, 55: 9}  # from capacity_band_sweep.json, same seed=0 protocol
            mismatches = {k: (last[k - 1], v) for k, v in known.items() if last[k - 1] != v}
            if mismatches:
                print(f"WARNING: seed0 band14-22 final-position counts differ from P0-c sweep result: {mismatches}", flush=True)
            else:
                print("seed0 band14-22 final-position counts match P0-c sweep result (regression check OK)", flush=True)

    determinism_match = None
    curve2_last = None
    if not SMOKE:
        ts = time.time()
        data2 = run_one(lm, lens, tok, spec, 0, full_band)
        curve2, _ = band_curve(data2["layer_word_ranks"], MAIN_BAND, data2["n_positions"], K_MAX)
        curve0 = per_seed_results["seed0"]["bands"]["main_14_22"]["counts_at_k_by_position"]
        determinism_match = curve2 == curve0
        curve2_last = curve2[-1]
        print(f"determinism recheck done in {time.time()-ts:.1f}s, match={determinism_match}", flush=True)
        if not determinism_match:
            diffs = [i for i, (a, b) in enumerate(zip(curve0, curve2)) if a != b]
            print(f"NON-DETERMINISM at positions (showing up to 10): {diffs[:10]}", flush=True)

    final_curves = np.array(
        [per_seed_results[f"seed{s}"]["bands"]["main_14_22"]["counts_at_k_by_position"][-1] for s in seeds_to_run]
    )
    mean_curve = final_curves.mean(axis=0).tolist()
    std_curve = final_curves.std(axis=0).tolist()

    output = {
        "model": MODEL_ID,
        "lens_repo": LENS_REPO,
        "lens_file": LENS_FILE,
        "full_band": full_band,
        "main_band": MAIN_BAND,
        "sweep_bands": sweep_bands,
        "window_width": WINDOW_WIDTH,
        "k_max": K_MAX,
        "seeds": seeds_to_run,
        "smoke_mode": SMOKE,
        "per_seed": per_seed_results,
        "determinism_check": {"match": determinism_match, "seed0_rerun_final_curve": curve2_last},
        "aggregate_main_band": {"mean_final_k_curve": mean_curve, "std_final_k_curve": std_curve},
        "elapsed_s": time.time() - t0,
    }
    out_path = OUT_PATH if not SMOKE else OUT_PATH.replace(".json", ".smoke.json")
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    print(f"total time: {time.time()-t0:.1f}s -> {out_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
