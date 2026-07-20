"""T02 Part A step 2: supplement measurements for models with existing
partial E1/E4 data (design v1.5 / judgement doc S2, instructions Step 2).

mode=band4_only (qwen3-1.7b): main band (10 seed) and both controls already
  exist in full from E1/E2 -- compute ONLY band4 counts, seed 0-9, from a
  fresh lens.apply(layers=band4).

mode=supplement (qwen3-4b / gemma-3-270m / gemma-3-1b): existing
  band_capacity_<model>.json has main-band capacity for seed 0-2 only, no
  controls. This script:
    - reuses (does not recompute) the existing seed0-2 main-band curves
    - computes band4 fresh for all 10 seeds (0-2 via a band4-only apply
      call, 3-9 via the same band_final apply call that also yields the new
      seeds' main-band curves)
    - computes controls A (logit-lens, 10 seed) and B (random-orthogonal,
      3 seed) fresh in full, main band only

Mechanical execution only -- see t02_partA_pod_instructions_2026-07-10.md.

Usage:
  python3 partA_supplement.py band4_only <np_id> <hf_name> <lens_repo> <lens_file> <existing_json> <out_path>
  python3 partA_supplement.py supplement <np_id> <hf_name> <lens_repo> <lens_file> <existing_band_capacity_json> <out_path>
"""
import json
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import JacobianLens, from_hf

sys.path.insert(0, "/workspace/t02")
from t02_common import band_curve, build_trial, exact_ranks_per_layer, random_orthogonal_matrix  # noqa: E402

CAPACITY_SPEC = "/workspace/t02/jacobian-lens/data/experiments/capacity.json"
ALL_SEEDS = list(range(10))
NEW_SEEDS = list(range(3, 10))
EXISTING_SEEDS = [0, 1, 2]
CONTROL_B_SEEDS = [0, 1, 2]
K_MAX = 128


def run_one(lm, lens, tok, spec, seed, layers, use_jacobian=True):
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
    lens_logits, _, _ = lens.apply(
        lm, prompt, layers=layers, positions=comma_tok_positions[: len(words)], use_jacobian=use_jacobian
    )
    layer_word_ranks = exact_ranks_per_layer(lens_logits, layers, word_token_ids)
    return {"n_positions": n_positions, "seq_len": int(input_ids.shape[1]), "layer_word_ranks": layer_word_ranks}


def checkpoint(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))


def main():
    mode, np_id, hf_name, lens_repo, lens_file, existing_json_path, out_path = sys.argv[1:8]
    assert mode in ("band4_only", "supplement"), f"unknown mode {mode}"
    t0 = time.time()
    ckpt_path = out_path.replace(".json", ".ckpt.json")
    existing = json.load(open(existing_json_path))

    spec = json.load(open(CAPACITY_SPEC))
    model = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.bfloat16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(hf_name)
    lm = from_hf(model, tok)
    print(f"[{np_id}] model loaded (bf16) in {time.time()-t0:.1f}s, n_layers={lm.n_layers}, d_model={lm.d_model}", flush=True)

    lens = JacobianLens.from_pretrained(lens_repo, filename=lens_file)
    print(f"[{np_id}] lens loaded, source_layers={lens.source_layers}", flush=True)

    result = {"np_model_id": np_id, "hf_model_name": hf_name, "lens_repo": lens_repo, "lens_file": lens_file, "mode": mode}

    if mode == "band4_only":
        # qwen3-1.7b: main band already MAIN_BAND=[14..22] from E1; band4 is its front 4.
        main_band = existing["main_band"]
        band4 = main_band[: min(4, len(main_band))]
        band4 = [l for l in band4 if l in lens.source_layers]
        result["reused_main_band"] = main_band
        result["band4"] = band4
        result["note"] = "main_band (10 seed) and both controls reused as-is from E1/E2 -- not duplicated here."

        band4_only_data = {}
        for seed in ALL_SEEDS:
            ts = time.time()
            data = run_one(lm, lens, tok, spec, seed, band4, use_jacobian=True)
            curve, used = band_curve(data["layer_word_ranks"], band4, data["n_positions"], K_MAX)
            band4_only_data[f"seed{seed}"] = {
                "n_positions": data["n_positions"],
                "seq_len": data["seq_len"],
                "band": used,
                "counts_at_k_by_position": curve,
            }
            print(f"[{np_id}] band4_only seed{seed} done in {time.time()-ts:.1f}s", flush=True)
            result["band4_data"] = band4_only_data
            checkpoint(ckpt_path, result)

    else:  # supplement
        band_final = existing["capacity_band_used"]
        band4 = band_final[: min(4, len(band_final))]
        band4 = [l for l in band4 if l in lens.source_layers]
        result["band_final"] = band_final
        result["band4"] = band4
        result["k_max"] = K_MAX

        capacity = {}
        # Existing seeds: reuse main-band values verbatim, compute band4 fresh (band4-only apply, cheap).
        t_phase = time.time()
        for seed in EXISTING_SEEDS:
            ts = time.time()
            existing_seed_entry = existing["capacity"][f"seed{seed}"]
            data = run_one(lm, lens, tok, spec, seed, band4, use_jacobian=True)
            curve_b4, used_b4 = band_curve(data["layer_word_ranks"], band4, data["n_positions"], K_MAX)
            capacity[f"seed{seed}"] = {
                "n_positions": existing_seed_entry["n_positions"],
                "main_band": {"band": existing_seed_entry["band"], "counts_at_k_by_position": existing_seed_entry["counts_at_k_by_position"]},
                "main_band_source": "reused_from_E4_band_capacity_json",
                "band4": {"band": used_b4, "counts_at_k_by_position": curve_b4},
            }
            print(f"[{np_id}] existing-seed{seed} band4-only done in {time.time()-ts:.1f}s", flush=True)
            result["capacity"] = capacity
            result["_partial_phase"] = "capacity_existing"
            checkpoint(ckpt_path, result)

        # New seeds 3-9: one apply() over band_final -> both main and band4 sliced.
        for seed in NEW_SEEDS:
            ts = time.time()
            data = run_one(lm, lens, tok, spec, seed, band_final, use_jacobian=True)
            curve_main, used_main = band_curve(data["layer_word_ranks"], band_final, data["n_positions"], K_MAX)
            curve_b4, used_b4 = band_curve(data["layer_word_ranks"], band4, data["n_positions"], K_MAX)
            capacity[f"seed{seed}"] = {
                "n_positions": data["n_positions"],
                "main_band": {"band": used_main, "counts_at_k_by_position": curve_main},
                "main_band_source": "freshly_computed",
                "band4": {"band": used_b4, "counts_at_k_by_position": curve_b4},
            }
            print(f"[{np_id}] new-seed{seed} main+band4 done in {time.time()-ts:.1f}s", flush=True)
            result["capacity"] = capacity
            result["_partial_phase"] = "capacity_new"
            checkpoint(ckpt_path, result)
        result["capacity_elapsed_s"] = time.time() - t_phase

        # Controls, fresh in full, main band only.
        control_a = {}
        t_phase = time.time()
        for seed in ALL_SEEDS:
            ts = time.time()
            data = run_one(lm, lens, tok, spec, seed, band_final, use_jacobian=False)
            curve, used = band_curve(data["layer_word_ranks"], band_final, data["n_positions"], K_MAX)
            control_a[f"seed{seed}"] = {
                "n_positions": data["n_positions"],
                "seq_len": data["seq_len"],
                "bands": {"main_band": {"band": used, "counts_at_k_by_position": curve}},
            }
            print(f"[{np_id}] [A logit-lens] seed{seed} done in {time.time()-ts:.1f}s", flush=True)
            result["control_a_logit_lens"] = control_a
            result["_partial_phase"] = "control_a"
            checkpoint(ckpt_path, result)
        result["control_a_elapsed_s"] = time.time() - t_phase

        control_b = {}
        t_phase = time.time()
        for seed in CONTROL_B_SEEDS:
            ts = time.time()
            gen = torch.Generator().manual_seed(seed)
            fake_jacobians = {layer: random_orthogonal_matrix(lens.d_model, gen) for layer in band_final}
            fake_lens = JacobianLens(jacobians=fake_jacobians, n_prompts=lens.n_prompts, d_model=lens.d_model)
            data = run_one(lm, fake_lens, tok, spec, seed, band_final, use_jacobian=True)
            curve, used = band_curve(data["layer_word_ranks"], band_final, data["n_positions"], K_MAX)
            control_b[f"seed{seed}"] = {
                "n_positions": data["n_positions"],
                "seq_len": data["seq_len"],
                "random_j_seed": seed,
                "bands": {"main_band": {"band": used, "counts_at_k_by_position": curve}},
            }
            print(f"[{np_id}] [B random-orthogonal] seed{seed} done in {time.time()-ts:.1f}s", flush=True)
            result["control_b_random_orthogonal"] = control_b
            result["_partial_phase"] = "control_b"
            checkpoint(ckpt_path, result)
        result["control_b_elapsed_s"] = time.time() - t_phase
        result["control_b_method"] = "Haar-random orthogonal matrix via QR + sign correction (Mezzadri 2006), fresh per seed, over band_final layers only"

    result["_partial_phase"] = "done"
    result["elapsed_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"[{np_id}] total time {result['elapsed_s']:.1f}s -> {out_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
