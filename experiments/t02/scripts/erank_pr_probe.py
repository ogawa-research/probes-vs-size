"""T02 erank/PR probe: lens-independent scaling measurement for 副判定 2.

Computes per-layer KV effective rank (erank/NER, KV-CoRE 2602.05929 sec 2.4
definition) and hidden-state participation ratio (PR, standard definition),
over two input sets (neutral WikiText corpus + capacity-list prompts).

Spec: t02_erank_pr_instructions_2026-07-11.md. Mechanical execution only --
definitions, inputs, and models are fixed by that document; do not change them
here.

Usage: python3 erank_pr_probe.py <np_id> <hf_name> <out_path> [--smoke]
"""
import gc
import json
import random
import sys
import time

import torch

# CPU SVD (torch.linalg.svdvals) on this instance's 128-core CPU is ~1000x slower
# at torch's default thread count (128) than at a moderate thread count, for
# both small (KV, ~1e2-1e3 cols) and large (hidden, ~5e3 cols) matrices --
# measured empirically: 640x256 took 6.0s at 128 threads vs 0.007s at 16;
# 12800x5376 took 6.6s at 16 threads (fastest of 4/8/16/32 tested). Thread
# scheduling overhead dominates the actual FLOPs at this core count.
torch.set_num_threads(16)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

sys.path.insert(0, "/workspace/t02")
sys.path.insert(0, "/workspace/t02/jacobian-lens")
from jlens.examples import load_wikitext_prompts  # noqa: E402

from t02_common import build_trial  # noqa: E402

CAPACITY_SPEC = "/workspace/t02/jacobian-lens/data/experiments/capacity.json"
N_NEUTRAL_PROMPTS = 100
NEUTRAL_MAX_SEQ_LEN = 128
CAPACITY_SEEDS = [0, 1, 2]
CAPACITY_MAX_LEN = 2048

SMOKE = "--smoke" in sys.argv


def get_n_layers_d_model(model):
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    return int(text_cfg.num_hidden_layers), int(text_cfg.hidden_size)


def extract_layer_kv(past_key_values, layer_idx):
    """Returns (key, value), each [batch, num_kv_heads, seq_len, head_dim],
    still on-device/bf16. Handles legacy tuple-of-tuples and the new Cache
    object APIs -- transformers' Cache implementation has changed across
    versions; kept multi-path for robustness (confirmed empirically at smoke
    time which branch this environment actually takes)."""
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values
    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    return past_key_values[layer_idx]


def n_cache_layers(past_key_values):
    if hasattr(past_key_values, "layers"):
        return len(past_key_values.layers)
    if hasattr(past_key_values, "key_cache"):
        return len(past_key_values.key_cache)
    return len(past_key_values)


def to_dataset_matrix(per_head_tensor):
    """[num_kv_heads, seq_len, head_dim] (fp32 CPU) -> [seq_len, num_kv_heads*head_dim]."""
    nh, sl, hd = per_head_tensor.shape
    return per_head_tensor.permute(1, 0, 2).reshape(sl, nh * hd)


def erank_and_ner(X):
    """X: [L_tot, D] fp32 CPU tensor. erank(X) = exp(-sum p_i log p_i),
    p_i = sigma_i / sum(sigma), r = min(rows, cols) (all singular values, per
    instructions). Returns (erank, ner, r)."""
    sv = torch.linalg.svdvals(X).clamp(min=0)
    r = sv.shape[0]
    total = sv.sum()
    if total <= 0:
        return 0.0, 0.0, int(r)
    p = sv / total
    nz = p > 0
    entropy = -(p[nz] * torch.log(p[nz])).sum()
    erank = torch.exp(entropy).item()
    return float(erank), float(erank / r), int(r)


def participation_ratio(H):
    """H: [L_tot, D] fp32 CPU tensor (residual-stream hidden states).
    Column-center, PR = (sum lambda)^2 / sum(lambda^2), lambda_i = s_i^2/(n-1)."""
    Hc = H - H.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(Hc)
    n = Hc.shape[0]
    lam = (sv**2) / max(n - 1, 1)
    s1 = lam.sum()
    s2 = (lam**2).sum()
    if s2 <= 0:
        return 0.0
    return float((s1 * s1 / s2).item())


def collect_layerwise(model, tok, prompts, max_length, device, tag):
    """Runs one forward per prompt (batch size 1, use_cache=True,
    output_hidden_states=True), accumulating per-layer key/value/hidden CPU
    fp32 tensors (moved off-GPU immediately per instructions), then computes
    erank/NER/PR per layer. Returns (per_layer dict, l_tot, num_kv_heads,
    head_dim, n_layers_used)."""
    n_layers_cfg, _ = get_n_layers_d_model(model)
    keys = {l: [] for l in range(n_layers_cfg)}
    values = {l: [] for l in range(n_layers_cfg)}
    hiddens = {l: [] for l in range(n_layers_cfg)}
    total_tokens = 0
    num_kv_heads = head_dim = None
    n_layers_cache = None

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            out = model(**enc, use_cache=True, output_hidden_states=True)
            seq_len = enc["input_ids"].shape[1]
            total_tokens += seq_len

            if n_layers_cache is None:
                n_layers_cache = n_cache_layers(out.past_key_values)
                if n_layers_cache != n_layers_cfg:
                    print(
                        f"[{tag}] WARNING: cache layers ({n_layers_cache}) != config "
                        f"n_layers ({n_layers_cfg}); using cache count",
                        flush=True,
                    )

            n_layers = n_layers_cache if n_layers_cache is not None else n_layers_cfg
            hs = out.hidden_states  # tuple len n_layers+1; [0]=embed out, [l+1]=layer l out
            for l in range(n_layers):
                h = hs[l + 1][0].float().cpu()
                hiddens.setdefault(l, []).append(h)
                k, v = extract_layer_kv(out.past_key_values, l)
                k = k[0].float().cpu()
                v = v[0].float().cpu()
                if num_kv_heads is None:
                    num_kv_heads, _, head_dim = k.shape
                keys.setdefault(l, []).append(to_dataset_matrix(k))
                values.setdefault(l, []).append(to_dataset_matrix(v))
            del out
            if device == "cuda":
                torch.cuda.empty_cache()
            if (i + 1) % 20 == 0:
                print(f"[{tag}] processed {i + 1}/{len(prompts)} prompts", flush=True)

    n_layers = n_layers_cache if n_layers_cache is not None else n_layers_cfg
    result = {}
    for l in range(n_layers):
        Xk = torch.cat(keys.pop(l), dim=0)
        Xv = torch.cat(values.pop(l), dim=0)
        Xh = torch.cat(hiddens.pop(l), dim=0)
        k_erank, k_ner, k_r = erank_and_ner(Xk)
        v_erank, v_ner, v_r = erank_and_ner(Xv)
        pr = participation_ratio(Xh)
        result[str(l)] = {
            "key_erank": k_erank, "key_ner": k_ner, "key_r": k_r,
            "value_erank": v_erank, "value_ner": v_ner, "value_r": v_r,
            "hidden_pr": pr,
        }
        del Xk, Xv, Xh
    gc.collect()
    return result, total_tokens, num_kv_heads, head_dim, n_layers


def main():
    np_id, hf_name, out_path = sys.argv[1:4]
    t0 = time.time()
    device = "cuda"

    # Native load path (unchanged, exercised successfully by other models).
    # Falls back to glm_compat.load_with_glm_compat (trust_remote_code=True +
    # config/post_init compat patches, validated on chatglm3-6b 2026-07-12)
    # only on failure -- existing successful models never reach this branch.
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.bfloat16, device_map=device)
        tok = AutoTokenizer.from_pretrained(hf_name)
        load_mode = "native"
    except Exception as e:
        print(f"native load failed ({type(e).__name__}: {e}); retrying via glm_compat", flush=True)
        from glm_compat import load_with_glm_compat
        model, tok, load_mode = load_with_glm_compat(hf_name, device, dtype=torch.bfloat16)
    print(f"[{np_id}] load_mode={load_mode}", flush=True)
    model.eval()
    n_layers_cfg, d_model = get_n_layers_d_model(model)
    print(
        f"[{np_id}] model loaded (bf16) in {time.time() - t0:.1f}s n_layers_cfg={n_layers_cfg} "
        f"d_model={d_model}",
        flush=True,
    )

    result = {
        "np_model_id": np_id,
        "hf_model_name": hf_name,
        "n_layers_config": n_layers_cfg,
        "d_model": d_model,
        "dtype": "bfloat16",
        "smoke_mode": SMOKE,
    }
    ckpt_path = out_path.replace(".json", ".ckpt.json")

    # --- neutral corpus ---
    n_prompts = 5 if SMOKE else N_NEUTRAL_PROMPTS
    ts = time.time()
    prompts = load_wikitext_prompts(n_prompts)
    print(f"[{np_id}] loaded {len(prompts)} neutral prompts", flush=True)
    per_layer, l_tot, num_kv_heads, head_dim, n_layers_cache = collect_layerwise(
        model, tok, prompts, NEUTRAL_MAX_SEQ_LEN, device, f"{np_id}/neutral"
    )
    result["num_key_value_heads"] = num_kv_heads
    result["head_dim"] = head_dim
    result["n_layers_cache"] = n_layers_cache
    result["neutral"] = {
        "n_prompts": len(prompts),
        "max_seq_len": NEUTRAL_MAX_SEQ_LEN,
        "l_tot": l_tot,
        "per_layer": per_layer,
    }
    result["neutral_elapsed_s"] = time.time() - ts
    print(f"[{np_id}] neutral corpus done in {result['neutral_elapsed_s']:.1f}s (L_tot={l_tot})", flush=True)
    with open(ckpt_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    # --- capacity-list prompts, seed 0-2 ---
    spec = json.load(open(CAPACITY_SPEC))
    seeds = [0] if SMOKE else CAPACITY_SEEDS
    capacity_result = {}
    ts = time.time()
    for seed in seeds:
        rng = random.Random(seed)
        words = build_trial(spec, tok, rng)
        prompt = ", ".join(w for w, _ in words) + ","
        per_layer_c, l_tot_c, _, _, _ = collect_layerwise(
            model, tok, [prompt], CAPACITY_MAX_LEN, device, f"{np_id}/cap_seed{seed}"
        )
        capacity_result[f"seed{seed}"] = {"l_tot": l_tot_c, "per_layer": per_layer_c}
        print(f"[{np_id}] capacity seed{seed} done (L_tot={l_tot_c})", flush=True)
        result["capacity_list"] = capacity_result
        with open(ckpt_path, "w") as f:
            json.dump(result, f, separators=(",", ":"))
    result["capacity_elapsed_s"] = time.time() - ts

    result["elapsed_s"] = time.time() - t0
    out = out_path if not SMOKE else out_path.replace(".json", ".smoke.json")
    with open(out, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"[{np_id}] total time {result['elapsed_s']:.1f}s -> {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
