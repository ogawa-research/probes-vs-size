"""T02 P0-c completion E2: two controls for the capacity-probe protocol
(Qwen3-1.7B, same band/k grid as E1: main band 14-22 + 19-window sweep,
k=1..128, full comma-position curves).

Control A: logit-lens (use_jacobian=False, built into JacobianLens.apply) --
  10 seeds, matching E1's seed set for direct per-seed comparison.
Control B: J replaced by a Haar-random orthogonal matrix per layer (QR +
  sign correction) -- 3 seeds. Each seed drives both the word-list RNG and
  the random-orthogonal-matrix RNG (documented, not a design choice: purpose
  is just "a reproducible fresh draw per seed").

Mechanical execution only -- see t02_p0c_completion_instructions_2026-07-10.md.
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
from t02_common import band_curve, build_trial, exact_ranks_per_layer, random_orthogonal_matrix, sliding_windows  # noqa: E402

MODEL_ID = "Qwen/Qwen3-1.7B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"
CAPACITY_SPEC = "/workspace/t02/jacobian-lens/data/experiments/capacity.json"
SEEDS_A = list(range(10))
SEEDS_B = [0, 1, 2]
K_MAX = 128
WINDOW_WIDTH = 9
MAIN_BAND = list(range(14, 23))
OUT_PATH = "/workspace/t02/capacity_controls_qwen17b.json"
CKPT_PATH = "/workspace/t02/capacity_controls_qwen17b.ckpt.json"

SMOKE = "--smoke" in sys.argv


def run_one(lm, lens, tok, spec, seed, full_band, use_jacobian):
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
        lm, prompt, layers=full_band, positions=comma_tok_positions[: len(words)], use_jacobian=use_jacobian
    )
    layer_word_ranks = exact_ranks_per_layer(lens_logits, full_band, word_token_ids)
    return {"n_positions": n_positions, "seq_len": int(input_ids.shape[1]), "layer_word_ranks": layer_word_ranks}


def bands_for(layer_word_ranks, n_positions, all_bands):
    out = {}
    for band_name, band in all_bands.items():
        curve, used_band = band_curve(layer_word_ranks, band, n_positions, K_MAX)
        out[band_name] = {"band": used_band, "counts_at_k_by_position": curve}
    return out


def main():
    t0 = time.time()
    spec = json.load(open(CAPACITY_SPEC))

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    lm = from_hf(model, tok)
    print(f"model loaded in {time.time()-t0:.1f}s, n_layers={lm.n_layers}, d_model={lm.d_model}", flush=True)

    real_lens = JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    full_band = sorted(real_lens.source_layers)
    print(f"real lens loaded, source_layers={full_band}", flush=True)

    sweep_bands = sliding_windows(full_band, WINDOW_WIDTH)
    all_bands = {"main_14_22": MAIN_BAND, **sweep_bands}

    seeds_a = [0] if SMOKE else SEEDS_A
    seeds_b = [0] if SMOKE else SEEDS_B

    results = {"control_a_logit_lens": {}, "control_b_random_orthogonal": {}}

    # Control A: logit-lens (use_jacobian=False). No J needed at all; pass the
    # real lens object purely for its .apply() method and source_layers.
    for seed in seeds_a:
        ts = time.time()
        data = run_one(lm, real_lens, tok, spec, seed, full_band, use_jacobian=False)
        results["control_a_logit_lens"][f"seed{seed}"] = {
            "n_positions": data["n_positions"],
            "seq_len": data["seq_len"],
            "bands": bands_for(data["layer_word_ranks"], data["n_positions"], all_bands),
        }
        print(f"[A logit-lens] seed {seed} done in {time.time()-ts:.1f}s", flush=True)
        with open(CKPT_PATH, "w") as f:
            json.dump(results, f, separators=(",", ":"))

    # Control B: J replaced by a fresh Haar-random orthogonal matrix per layer, per seed.
    for seed in seeds_b:
        ts = time.time()
        gen = torch.Generator().manual_seed(seed)
        fake_jacobians = {layer: random_orthogonal_matrix(real_lens.d_model, gen) for layer in full_band}
        fake_lens = JacobianLens(jacobians=fake_jacobians, n_prompts=real_lens.n_prompts, d_model=real_lens.d_model)
        data = run_one(lm, fake_lens, tok, spec, seed, full_band, use_jacobian=True)
        results["control_b_random_orthogonal"][f"seed{seed}"] = {
            "n_positions": data["n_positions"],
            "seq_len": data["seq_len"],
            "random_j_seed": seed,
            "bands": bands_for(data["layer_word_ranks"], data["n_positions"], all_bands),
        }
        print(f"[B random-orthogonal] seed {seed} done in {time.time()-ts:.1f}s", flush=True)
        with open(CKPT_PATH, "w") as f:
            json.dump(results, f, separators=(",", ":"))

    output = {
        "model": MODEL_ID,
        "lens_repo": LENS_REPO,
        "lens_file": LENS_FILE,
        "full_band": full_band,
        "main_band": MAIN_BAND,
        "sweep_bands": sweep_bands,
        "window_width": WINDOW_WIDTH,
        "k_max": K_MAX,
        "seeds_control_a": seeds_a,
        "seeds_control_b": seeds_b,
        "control_b_method": "Haar-random orthogonal matrix via QR + sign correction (Mezzadri 2006), fresh per seed",
        "smoke_mode": SMOKE,
        **results,
        "elapsed_s": time.time() - t0,
    }
    out_path = OUT_PATH if not SMOKE else OUT_PATH.replace(".json", ".smoke.json")
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    print(f"total time: {time.time()-t0:.1f}s -> {out_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
