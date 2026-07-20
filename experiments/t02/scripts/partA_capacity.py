"""T02 Part A capacity measurement: generic per-model driver combining the
validated E1 (full k-grid x position-curve x 10-seed) and E2 (logit-lens +
random-orthogonal controls) patterns, extended with a second (band4) band
computed from the same lens.apply() call. Judgement doc
t02_p0c_judgement_2026-07-10.md and instructions
t02_partA_pod_instructions_2026-07-10.md are the specs this implements.

Mechanical execution only. Reuses run_one/band_curve/build_trial logic
verbatim from e1/e2/e4_band_and_capacity.py -- no new measurement logic.

Usage: python3 partA_capacity.py <np_id> <hf_name> <lens_repo> <lens_file> <cka_json_path> <out_path> [--smoke]
"""
import json
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import JacobianLens, from_hf

sys.path.insert(0, "/workspace/t02")
from band_rule import compute_band  # noqa: E402
from t02_common import band_curve, build_trial, exact_ranks_per_layer, random_orthogonal_matrix  # noqa: E402

CAPACITY_SPEC = "/workspace/t02/jacobian-lens/data/experiments/capacity.json"
CAPACITY_SEEDS = list(range(10))
CONTROL_B_SEEDS = [0, 1, 2]
K_MAX = 128

SMOKE = "--smoke" in sys.argv


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
    np_id, hf_name, lens_repo, lens_file, cka_json_path, out_path = sys.argv[1:7]
    t0 = time.time()
    ckpt_path = out_path.replace(".json", ".ckpt.json")

    cka_data = json.load(open(cka_json_path))
    n_layers = cka_data["n_layers"]
    d_model_cka = cka_data["d_model"]
    band_result = compute_band(cka_data["cka_matrix"], n_layers_for_floor=n_layers - 1)
    band = band_result["band"]
    print(f"[{np_id}] band rule result: {band_result}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.bfloat16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(hf_name)
    lm = from_hf(model, tok)
    print(f"[{np_id}] model loaded (bf16) in {time.time()-t0:.1f}s, n_layers={lm.n_layers}, d_model={lm.d_model}", flush=True)

    lens = JacobianLens.from_pretrained(lens_repo, filename=lens_file)
    band_final = [l for l in band if l in lens.source_layers]
    if len(band_final) != len(band):
        print(
            f"[{np_id}] WARNING: band {band} not fully within lens.source_layers "
            f"{lens.source_layers}; using intersection {band_final}",
            flush=True,
        )
    band4 = band_final[: min(4, len(band_final))]
    print(f"[{np_id}] band_final={band_final} band4={band4}", flush=True)

    spec = json.load(open(CAPACITY_SPEC))

    result = {
        "np_model_id": np_id,
        "hf_model_name": hf_name,
        "lens_repo": lens_repo,
        "lens_file": lens_file,
        "n_layers": n_layers,
        "d_model": d_model_cka,
        "lens_d_model": lens.d_model,
        "lens_source_layers": lens.source_layers,
        "lens_n_prompts": lens.n_prompts,
        "band_rule_result": band_result,
        "band_final": band_final,
        "band4": band4,
        "k_max": K_MAX,
        "smoke_mode": SMOKE,
    }

    seeds_main = [0] if SMOKE else CAPACITY_SEEDS
    seeds_a = [0] if SMOKE else CAPACITY_SEEDS
    seeds_b = [0] if SMOKE else CONTROL_B_SEEDS

    # Phase 1: main capacity measurement, 10 seeds, band_final + band4 sliced
    # from a single lens.apply() call per seed (called with layers=band_final;
    # band4 subset(band_final) so its ranks are already present).
    capacity = {}
    t_phase = time.time()
    for seed in seeds_main:
        ts = time.time()
        data = run_one(lm, lens, tok, spec, seed, band_final, use_jacobian=True)
        curve_main, used_main = band_curve(data["layer_word_ranks"], band_final, data["n_positions"], K_MAX)
        curve_b4, used_b4 = band_curve(data["layer_word_ranks"], band4, data["n_positions"], K_MAX)
        capacity[f"seed{seed}"] = {
            "n_positions": data["n_positions"],
            "seq_len": data["seq_len"],
            "main_band": {"band": used_main, "counts_at_k_by_position": curve_main},
            "band4": {"band": used_b4, "counts_at_k_by_position": curve_b4},
        }
        print(f"[{np_id}] capacity seed{seed} done in {time.time()-ts:.1f}s (n_positions={data['n_positions']})", flush=True)
        result["capacity"] = capacity
        result["_partial_phase"] = "capacity"
        checkpoint(ckpt_path, result)
    result["capacity_elapsed_s"] = time.time() - t_phase

    # Phase 2: control A (logit-lens, use_jacobian=False), 10 seeds, main band only.
    control_a = {}
    t_phase = time.time()
    for seed in seeds_a:
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

    # Phase 3: control B (random-orthogonal J), 3 seeds, main band only.
    # Fake jacobians built only over band_final layers (band4 subset(band_final),
    # not a separate control target per the spec).
    control_b = {}
    t_phase = time.time()
    for seed in seeds_b:
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
    out = out_path if not SMOKE else out_path.replace(".json", ".smoke.json")
    with open(out, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"[{np_id}] total time {result['elapsed_s']:.1f}s -> {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
