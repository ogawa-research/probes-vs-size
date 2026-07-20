"""T02 P0-c completion E4, step 2: apply the CKA band machine-rule and (if
model+lens load succeeds) run the capacity-probe measurement for one model.

Called after cka_band.py (reused as-is, unmodified) has already produced
cka_band_<np_id>.json for this model. Kept as a separate process/script per
model (invoked by e4_driver.sh) so a failure or OOM on one model cannot take
down the whole E4 loop and cannot leave stale CUDA state for the next model.

Mechanical execution only -- see t02_p0c_completion_instructions_2026-07-10.md.

Usage: python3 e4_band_and_capacity.py <np_model_id> <hf_model_name> <lens_repo> <lens_file> <cka_json_path> <out_path>
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
from t02_common import band_curve, build_trial, exact_ranks_per_layer  # noqa: E402

CAPACITY_SPEC = "/workspace/t02/jacobian-lens/data/experiments/capacity.json"
CAPACITY_SEEDS = [0, 1, 2]
K_MAX = 128


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
    np_id, hf_name, lens_repo, lens_file, cka_json_path, out_path = sys.argv[1:7]
    t0 = time.time()
    result = {"np_model_id": np_id, "hf_model_name": hf_name}

    cka_data = json.load(open(cka_json_path))
    n_layers = cka_data["n_layers"]
    band_result = compute_band(cka_data["cka_matrix"], n_layers_for_floor=n_layers - 1)
    result["band_rule_result"] = band_result
    result["n_layers"] = n_layers
    result["d_model"] = cka_data["d_model"]
    print(f"[{np_id}] band rule: {band_result}", flush=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.bfloat16, device_map="cuda")
        tok = AutoTokenizer.from_pretrained(hf_name)
        lm = from_hf(model, tok)
        print(f"[{np_id}] model loaded (bf16) in {time.time()-t0:.1f}s", flush=True)

        lens = JacobianLens.from_pretrained(lens_repo, filename=lens_file)
        band = [l for l in band_result["band"] if l in lens.source_layers]
        if len(band) != len(band_result["band"]):
            print(
                f"[{np_id}] WARNING: band {band_result['band']} not fully within "
                f"lens.source_layers {lens.source_layers}; using intersection {band}",
                flush=True,
            )

        spec = json.load(open(CAPACITY_SPEC))
        capacity = {}
        for seed in CAPACITY_SEEDS:
            ts = time.time()
            capacity[f"seed{seed}"] = run_capacity(lm, lens, tok, spec, seed, band)
            print(f"[{np_id}] capacity seed{seed} done in {time.time()-ts:.1f}s", flush=True)
        result["capacity_status"] = "ok"
        result["capacity_band_used"] = band
        result["capacity"] = capacity
        result["lens_source_layers"] = lens.source_layers
        result["lens_n_prompts"] = lens.n_prompts
    except Exception as exc:
        msg = str(exc)
        is_oom = "out of memory" in msg.lower()
        result["capacity_status"] = "oom" if is_oom else "error"
        result["capacity_error"] = f"{type(exc).__name__}: {msg}"
        print(f"[{np_id}] {'OOM' if is_oom else 'ERROR'} during model/capacity: {type(exc).__name__}: {msg}", flush=True)

    result["elapsed_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"[{np_id}] total time {result['elapsed_s']:.1f}s -> {out_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
