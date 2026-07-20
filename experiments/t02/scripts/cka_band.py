#!/usr/bin/env python3
"""CKA-based layer-block (band) detection for T02 P0-c.

Computes pairwise linear CKA between per-layer residual-stream activations
of a model on a WikiText-103 sample, to empirically locate the "workspace"
block that the capacity-probe rank measurement should use as its band
(replacing the middle-third placeholder). Reuses the official jacobian-lens
corpus loader and activation-recording hook for methodological consistency
with the J-lens fit itself.
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/t02/jacobian-lens")
from jlens.examples import load_wikitext_prompts
from jlens.hooks import ActivationRecorder

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
N_PROMPTS = 24
MAX_SEQ_LEN = 128
MIN_CHARS = 600
CACHE_DIR = "/workspace/t02/hf_cache"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "/workspace/t02/cka_band_result.json"
# fp32-load process was killed with no traceback (no CUDA OOM, no Python
# exception -- consistent with a host/cgroup CPU-RAM limit striking during
# from_pretrained's CPU-side weight materialization, before .to(device) is
# even reached) at ~80% shard progress on 2 separate Yi-34B attempts. The
# try/except around torch.OutOfMemoryError below cannot catch this since the
# process is killed outright. --bf16-first skips the fp32 attempt entirely.
BF16_FIRST = "--bf16-first" in sys.argv


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA (Kornblith et al. 2019). X: [n, p1], Y: [n, p2]."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    XtY = X.T @ Y
    num = (XtY**2).sum()
    XtX = X.T @ X
    YtY = Y.T @ Y
    denom = torch.sqrt((XtX**2).sum()) * torch.sqrt((YtY**2).sum())
    return (num / denom).item()


def main() -> None:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model={MODEL_NAME} device={device}", flush=True)

    # Native load path (unchanged, exercised successfully by 7 other
    # cross-family models). Falls back to glm_compat.load_with_glm_compat
    # (trust_remote_code=True + config/post_init compat patches, validated
    # on chatglm3-6b 2026-07-12) only on failure -- existing successful
    # models never reach this branch.
    load_mode = "native"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        if BF16_FIRST:
            print("--bf16-first set; skipping fp32 attempt", flush=True)
            dtype_used = "bfloat16"
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16
            ).to(device)
        else:
            dtype_used = "float32"
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, cache_dir=CACHE_DIR, torch_dtype=torch.float32
                ).to(device)
            except torch.OutOfMemoryError as e:
                print(f"fp32 load OOM ({e}); falling back to bf16", flush=True)
                torch.cuda.empty_cache()
                dtype_used = "bfloat16"
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16
                ).to(device)
    except Exception as e:
        print(f"native load failed ({type(e).__name__}: {e}); retrying via glm_compat", flush=True)
        from glm_compat import load_with_glm_compat
        dtype_used = "bfloat16"
        model, tokenizer, load_mode = load_with_glm_compat(MODEL_NAME, device, dtype=torch.bfloat16, cache_dir=CACHE_DIR)
    print(f"load_mode={load_mode} dtype_used={dtype_used}", flush=True)
    model.eval()

    cfg = model.config
    if hasattr(cfg, "num_hidden_layers"):
        n_layers = cfg.num_hidden_layers
        d_model = cfg.hidden_size
    elif hasattr(cfg, "text_config"):
        # Larger Gemma-3 checkpoints (4b/12b/27b) expose a nested Gemma3Config
        # with a text_config sub-object rather than flat top-level attributes
        # (270m/1b use a flat Gemma3TextConfig). Same semantic values either way.
        n_layers = cfg.text_config.num_hidden_layers
        d_model = cfg.text_config.hidden_size
    elif hasattr(cfg, "num_layers"):
        # ChatGLM family (ChatGLMConfig) names the layer count num_layers
        # instead of num_hidden_layers; hidden_size is the same name.
        n_layers = cfg.num_layers
        d_model = cfg.hidden_size
    else:
        raise AttributeError(f"cannot find layer-count/hidden-size attributes on config {type(cfg).__name__}")
    print(f"n_layers={n_layers} d_model={d_model}", flush=True)

    prompts = load_wikitext_prompts(N_PROMPTS, min_chars=MIN_CHARS)
    print(f"loaded {len(prompts)} prompts", flush=True)

    layer_acts: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        # Gemma3ForConditionalGeneration (4b/12b/27b, vision-capable checkpoints
        # even for the "-pt" text usage): decoder layers live one level deeper.
        blocks = model.model.language_model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "encoder"):
        # ChatGLM family: no top-level .model attribute at all; decoder
        # layers live under .transformer.encoder.layers instead.
        blocks = model.transformer.encoder.layers
    else:
        raise AttributeError(f"cannot find decoder layers on model {type(model).__name__}")

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
            ).to(device)
            with ActivationRecorder(blocks, at=range(n_layers)) as rec:
                model(**enc)
            for l in range(n_layers):
                # Drop position 0 (BOS/attention-sink token skews similarity).
                act = rec.activations[l][0, 1:, :].float().cpu()
                layer_acts[l].append(act)
            if (i + 1) % 5 == 0:
                print(f"  processed {i + 1}/{len(prompts)}", flush=True)

    X = {l: torch.cat(layer_acts[l], dim=0) for l in range(n_layers)}
    n_tokens = X[0].shape[0]
    print(f"n_tokens_total={n_tokens}", flush=True)

    cka = torch.zeros(n_layers, n_layers)
    for i in range(n_layers):
        for j in range(i, n_layers):
            v = linear_cka(X[i], X[j])
            cka[i, j] = v
            cka[j, i] = v
        print(f"  CKA row {i}/{n_layers} done", flush=True)

    result = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_prompts": len(prompts),
        "n_tokens_total": n_tokens,
        "max_seq_len": MAX_SEQ_LEN,
        "dtype_used": dtype_used,
        "cka_matrix": cka.tolist(),
        "elapsed_s": time.time() - t0,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f)
    print(f"DONE in {result['elapsed_s']:.1f}s -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
