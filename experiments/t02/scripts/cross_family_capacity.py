"""T02 Part B cross-family capacity measurement for models without a
neuronpedia/jacobian-lens entry: vanilla logit-lens k* (use_jacobian=False)
+ random-orthogonal control only (no J-lens main measurement, since there is
no fitted lens to use). Reuses run_one/band_curve/build_trial verbatim from
partA_capacity.py; JacobianLens(jacobians={}, ...) is valid here because
apply(..., use_jacobian=False) never touches self.jacobians or
source_layers (confirmed in jlens/lens.py: the "unknown layers" check is
gated on `use_jacobian and unknown`).

Spec: t02_partB_b1b2_instructions_2026-07-11.md sec "W（帯域）", "既製lensの
有無を確認... なければ erank + logit-lens k* のみ".

Usage: python3 cross_family_capacity.py <np_id> <hf_name> <cka_json_path> <out_path> [--smoke]
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


def load_model_and_tokenizer(hf_name, device):
    """Same native-first, custom-code-fallback strategy as r_delta_probe.py /
    a_retrieval_probe.py (approved for this run)."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_name, dtype=torch.bfloat16, device_map=device, trust_remote_code=False
        )
        tok = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=False)
        return model, tok
    except Exception as e:
        print(f"native load failed ({type(e).__name__}: {e}); retrying via glm_compat", flush=True)
        # glm_compat adds config.max_length/config.use_cache defaults and a
        # post_init() patch (validated on chatglm3-6b 2026-07-12: some
        # bundled custom modeling_*.py never call PreTrainedModel.post_init,
        # which is where transformers >=5.13 sets all_tied_weights_keys).
        from glm_compat import load_with_glm_compat
        model, tok, load_mode = load_with_glm_compat(hf_name, device, dtype=torch.bfloat16)
        print(f"load_mode={load_mode}", flush=True)
        return model, tok


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
    np_id, hf_name, cka_json_path, out_path = sys.argv[1:5]
    t0 = time.time()
    ckpt_path = out_path.replace(".json", ".ckpt.json")

    cka_data = json.load(open(cka_json_path))
    n_layers = cka_data["n_layers"]
    d_model_cka = cka_data["d_model"]
    band_result = compute_band(cka_data["cka_matrix"], n_layers_for_floor=n_layers - 1)
    band = band_result["band"]
    run_coverage = band_result["longest_run_length"] / band_result["n_layers_for_floor"]
    print(f"[{np_id}] band rule result: {band_result} run_coverage={run_coverage:.3f}", flush=True)

    model, tok = load_model_and_tokenizer(hf_name, "cuda")
    lm = from_hf(model, tok)
    print(f"[{np_id}] model loaded (bf16) in {time.time()-t0:.1f}s, n_layers={lm.n_layers}, d_model={lm.d_model}", flush=True)

    band4 = band[: min(4, len(band))]
    spec = json.load(open(CAPACITY_SPEC))

    result = {
        "np_model_id": np_id,
        "hf_model_name": hf_name,
        "n_layers": n_layers,
        "d_model": d_model_cka,
        "band_rule_result": band_result,
        "band_run_coverage": run_coverage,
        "band_watchflag_gt90pct": run_coverage > 0.9,
        "band": band,
        "band4": band4,
        "k_max": K_MAX,
        "smoke_mode": SMOKE,
        "lens_available": False,
    }

    seeds_a = [0] if SMOKE else CAPACITY_SEEDS
    seeds_b = [0] if SMOKE else CONTROL_B_SEEDS

    # logit-lens (use_jacobian=False, dummy empty lens -- jacobians dict is
    # never consulted in this path)
    dummy_lens = JacobianLens(jacobians={}, n_prompts=0, d_model=lm.d_model)
    control_a = {}
    t_phase = time.time()
    for seed in seeds_a:
        ts = time.time()
        data = run_one(lm, dummy_lens, tok, spec, seed, band, use_jacobian=False)
        curve_main, used_main = band_curve(data["layer_word_ranks"], band, data["n_positions"], K_MAX)
        curve_b4, used_b4 = band_curve(data["layer_word_ranks"], band4, data["n_positions"], K_MAX)
        control_a[f"seed{seed}"] = {
            "n_positions": data["n_positions"],
            "seq_len": data["seq_len"],
            "main_band": {"band": used_main, "counts_at_k_by_position": curve_main},
            "band4": {"band": used_b4, "counts_at_k_by_position": curve_b4},
        }
        print(f"[{np_id}] [logit-lens] seed{seed} done in {time.time()-ts:.1f}s (n_positions={data['n_positions']})", flush=True)
        result["control_a_logit_lens"] = control_a
        checkpoint(ckpt_path, result)
    result["control_a_elapsed_s"] = time.time() - t_phase

    # random-orthogonal control (fake J, not a fitted lens)
    control_b = {}
    t_phase = time.time()
    for seed in seeds_b:
        ts = time.time()
        gen = torch.Generator().manual_seed(seed)
        fake_jacobians = {layer: random_orthogonal_matrix(lm.d_model, gen) for layer in band}
        fake_lens = JacobianLens(jacobians=fake_jacobians, n_prompts=0, d_model=lm.d_model)
        data = run_one(lm, fake_lens, tok, spec, seed, band, use_jacobian=True)
        curve, used = band_curve(data["layer_word_ranks"], band, data["n_positions"], K_MAX)
        control_b[f"seed{seed}"] = {
            "n_positions": data["n_positions"],
            "seq_len": data["seq_len"],
            "random_j_seed": seed,
            "bands": {"main_band": {"band": used, "counts_at_k_by_position": curve}},
        }
        print(f"[{np_id}] [random-orthogonal] seed{seed} done in {time.time()-ts:.1f}s", flush=True)
        result["control_b_random_orthogonal"] = control_b
        checkpoint(ckpt_path, result)
    result["control_b_elapsed_s"] = time.time() - t_phase
    result["control_b_method"] = "Haar-random orthogonal matrix via QR + sign correction (Mezzadri 2006), fresh per seed"

    result["elapsed_s"] = time.time() - t0
    out = out_path if not SMOKE else out_path.replace(".json", ".smoke.json")
    with open(out, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"[{np_id}] total time {result['elapsed_s']:.1f}s -> {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
