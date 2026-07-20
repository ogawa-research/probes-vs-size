"""T02 Part B r_delta_probe: attention-entropy growth rate (R factor) and
logit-spread (delta, P0-d anchor) measurement.

Spec: t02_partB_b1b2_instructions_2026-07-11.md sec "r_delta_probe.py (R因子+δ)".
Mechanical execution only -- definitions and measurement conditions are fixed
by that document; do not change them here.

Model is loaded with attn_implementation="eager" so eager_attention_forward
(which always computes attn_weights regardless of output_attentions -- verified
against transformers 5.13.0 modeling_qwen3.py) runs; a forward hook on each
decoder layer's self_attn module reads attn_weights from the hook's `output`
argument, computes per-head H/delta at a fixed set of query positions, and
discards the tensor immediately (no output_attentions=True, no full-tensor
retention across layers).

Usage:
  python3 r_delta_probe.py <np_id> <hf_name> <out_path> [--smoke]
  python3 r_delta_probe.py <np_id> <hf_name> <out_path> --code-input  (P0-d delta-only mode)
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/t02")
sys.path.insert(0, "/workspace/t02/jacobian-lens")
from jlens.examples import load_wikitext_prompts  # noqa: E402

LENGTHS = [1024, 2048, 4096]
N_SEQS = 4
N_QUERY_POSITIONS = 64
MIN_QUERY_POS = 32
# CLIP_FLOOR history (B5 Amendment 1, 2026-07-17, researcher):
#   v1 used 1e-45, silently a NO-OP on bf16 tensors -- 1e-45 cast to bf16 is
#   exactly 0.0 (bf16 min subnormal ~9.2e-41), so exact-zero attention rows
#   (softmax over sliding-window-masked columns) passed the clamp, log(0)=-inf,
#   0*(-inf)=nan, and the nan entered H_sum as a "valid" (finite) row. This is
#   the actual root cause of the Gemma-3 sliding-layer NaN pattern in the B2
#   measurements. 1e-38 is representable (nonzero) in bf16/fp16/fp32; for all
#   nonzero attention values in the normal range the clamp result is unchanged,
#   so full-attention measurements are numerically identical to v1 (gated by
#   the qwen3-1.7b consistency check). A lifted exact zero contributes
#   -1e-38*log(1e-38) ~ 8.8e-37 to H, i.e. the intended 0*log0=0 convention.
CLIP_FLOOR = 1e-38
# The instructions cite https://github.com/google-deepmind/gemma/blob/main/gemma/modules.py
# (2410.01104 footnote 3), which now 404s: the repo was refactored and
# gemma/modules.py was renamed to gemma/gm/nn/_modules.py in commit 43b7412
# (2025-04-30), well after the paper's Oct-2024 submission. Using the file at
# its Oct-2024 state instead (commit 91c3739f6ec1, 2024-10-28, the closest
# available commit to the paper's arXiv submission month) so the input is the
# actual historical file rather than the current, structurally different one.
# Judgment call, recorded here for researcher review.
CODE_INPUT_URL = "https://raw.githubusercontent.com/google-deepmind/gemma/91c3739f6ec1/gemma/modules.py"
CODE_INPUT_TARGET_TOKENS = 4000

SMOKE = "--smoke" in sys.argv
CODE_INPUT = "--code-input" in sys.argv


def get_decoder_layers(model):
    if hasattr(model.model, "layers"):
        return model.model.layers
    return model.model.language_model.layers


def get_query_positions(seq_len, n_positions=N_QUERY_POSITIONS, min_pos=MIN_QUERY_POS):
    """n_positions log-spaced integer positions in [min_pos, seq_len-1], deduped."""
    hi = seq_len - 1
    if hi <= min_pos:
        return list(range(min(min_pos, hi), hi + 1))
    raw = np.logspace(np.log10(min_pos), np.log10(hi), n_positions)
    positions = np.unique(np.round(raw).astype(int))
    positions = positions[(positions >= min_pos) & (positions <= hi)]
    return positions.tolist()


class AttnStatsHook:
    """Registered on one decoder layer's self_attn module. Reads
    (attn_output, attn_weights) from the forward output, computes per-head
    entropy H and log-spread delta at `query_positions` (causal range only),
    accumulates running sums in fp64, and drops the attn_weights reference
    immediately after use."""

    def __init__(self, query_positions):
        self.query_positions = query_positions
        self.n_heads = None
        self.H_sum = None
        self.delta_sum = None
        self.delta_sq_sum = None
        self.n_rows = 0
        self.clip_count = 0
        # non-finite guard (2026-07-15; root-cause note corrected 2026-07-17):
        # the 07-15 claim that "the attention weights themselves contain
        # non-finite entries" was WRONG. B5 Job 1 diagnosis: weights outside
        # the sliding window are exact bf16 zeros (finite, so this guard never
        # fires for them); the nan arose downstream from log(0) because the v1
        # CLIP_FLOOR=1e-45 underflowed to 0 in bf16 (see CLIP_FLOOR note).
        # clip_count stayed 0 in B2 precisely because the clamp was a no-op.
        # The guard itself is kept: it still excludes genuinely non-finite
        # raw weights per head-row instead of poisoning the layer mean.
        self.n_valid = None          # per-head count of finite rows
        self.nonfinite_head_rows = 0

    def __call__(self, module, inputs, output):
        attn_weights = output[1]
        if attn_weights is None:
            raise RuntimeError(
                "attn_weights is None from self_attn forward -- eager attention "
                "implementation not returning weights as expected"
            )
        aw = attn_weights[0]  # [n_heads, seq_len, seq_len], MODEL dtype (bf16 when
        # loaded in bf16): transformers eager attention computes softmax in fp32
        # then casts back to query dtype. The v1 comment claiming fp32 was wrong.
        # fail-closed: CLIP_FLOOR must be representable (nonzero) in this dtype,
        # else the clamp silently degenerates to a no-op (the v1 bug).
        assert torch.tensor(CLIP_FLOOR, dtype=aw.dtype).item() > 0, (
            f"CLIP_FLOOR={CLIP_FLOOR} underflows to 0 in {aw.dtype}; clamp would be a no-op"
        )
        n_heads, seq_len, _ = aw.shape
        if self.n_heads is None:
            self.n_heads = n_heads
            self.H_sum = torch.zeros(n_heads, dtype=torch.float64)
            self.delta_sum = torch.zeros(n_heads, dtype=torch.float64)
            self.delta_sq_sum = torch.zeros(n_heads, dtype=torch.float64)
            self.n_valid = torch.zeros(n_heads, dtype=torch.float64)
        for pos in self.query_positions:
            if pos >= seq_len:
                continue
            row = aw[:, pos, : pos + 1]  # [n_heads, pos+1], causal range only
            finite = torch.isfinite(row).all(dim=-1)  # [n_heads] per-head row validity
            self.nonfinite_head_rows += int((~finite).sum().item())
            clipped = (row < CLIP_FLOOR) & torch.isfinite(row)
            self.clip_count += int(clipped.sum().item())
            row = torch.nan_to_num(row, nan=CLIP_FLOOR, posinf=CLIP_FLOOR, neginf=CLIP_FLOOR)
            row = row.clamp(min=CLIP_FLOOR)
            logrow = row.log()
            H = -(row * logrow).sum(dim=-1)  # [n_heads]
            delta = logrow.max(dim=-1).values - logrow.min(dim=-1).values  # log(a_max/a_min)
            fin = finite.double().cpu()
            self.H_sum += H.double().cpu() * fin
            self.delta_sum += delta.double().cpu() * fin
            self.delta_sq_sum += (delta.double() ** 2).cpu() * fin
            self.n_valid += fin
            self.n_rows += 1
        del attn_weights, aw, output

    def summarize(self):
        if self.n_rows == 0:
            return None
        n = self.n_rows
        # per-head means over VALID (finite) rows only; heads with zero valid
        # rows yield nan per-head and are excluded from the head mean below.
        nv = self.n_valid.clamp(min=0)
        safe_nv = nv.clamp(min=1)
        H_ph = self.H_sum / safe_nv
        d_ph = self.delta_sum / safe_nv
        dv_ph = (self.delta_sq_sum / safe_nv - d_ph ** 2).clamp(min=0)
        dead = nv == 0
        H_ph[dead] = float('nan'); d_ph[dead] = float('nan'); dv_ph[dead] = float('nan')
        H_mean_per_head = H_ph.tolist()
        delta_mean_per_head = d_ph.tolist()
        delta_sd_per_head = dv_ph.sqrt().tolist()
        live = [h for h in H_mean_per_head if np.isfinite(h)]
        live_d = [d for d in delta_mean_per_head if np.isfinite(d)]
        return {
            "n_rows": n,
            "n_heads": self.n_heads,
            "H_mean_per_head": H_mean_per_head,
            "H_mean": float(np.mean(live)) if live else float('nan'),
            "delta_mean_per_head": delta_mean_per_head,
            "delta_mean": float(np.mean(live_d)) if live_d else float('nan'),
            "delta_sd_per_head": delta_sd_per_head,
            "delta_sd": float(np.mean([s for s in delta_sd_per_head if np.isfinite(s)]))
                        if live_d else float('nan'),
            "clip_count": self.clip_count,
            "n_valid_per_head": nv.tolist(),
            "nonfinite_head_rows": self.nonfinite_head_rows,
        }


def build_token_pool(tokenizer, min_tokens):
    """Load a generously large WikiText-103 pool and concatenate + tokenize
    until at least min_tokens tokens are available."""
    n_prompts = 60
    ids = []
    while len(ids) < min_tokens:
        prompts = load_wikitext_prompts(n_prompts, min_chars=600)
        text = "\n\n".join(prompts)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < min_tokens:
            n_prompts *= 2
        if n_prompts > 8000:
            raise RuntimeError(f"could not build a token pool of {min_tokens} tokens (got {len(ids)})")
    return ids


def run_layers_forward(model, layers, query_positions, input_ids_list, device, tag):
    """Registers one AttnStatsHook per layer, forwards each sequence in
    input_ids_list once (no_grad), removes hooks, returns per-layer dict."""
    hooks = {}
    handles = []
    for li, layer in enumerate(layers):
        h = AttnStatsHook(query_positions)
        hooks[li] = h
        handles.append(layer.self_attn.register_forward_hook(h))
    try:
        with torch.no_grad():
            for seq_i, seq_ids in enumerate(input_ids_list):
                input_ids = torch.tensor([seq_ids], dtype=torch.long, device=device)
                model(input_ids=input_ids)
                print(f"[{tag}] seq {seq_i + 1}/{len(input_ids_list)} (len={len(seq_ids)}) forward done", flush=True)
    finally:
        for h in handles:
            h.remove()
    return {li: hook.summarize() for li, hook in hooks.items()}


def measure_neutral(model, tokenizer, device, np_id):
    lengths = [128, 256] if SMOKE else LENGTHS
    n_seqs = 2 if SMOKE else N_SEQS
    total_needed = sum(lengths) * n_seqs
    pool = build_token_pool(tokenizer, total_needed)
    print(f"[{np_id}] token pool built: {len(pool)} tokens (need {total_needed})", flush=True)

    layers = get_decoder_layers(model)
    per_length = {}
    ptr = 0
    for n in lengths:
        qpos = get_query_positions(n)
        seqs = []
        for _ in range(n_seqs):
            seq_ids = pool[ptr : ptr + n]
            ptr += n
            if len(seq_ids) < n:
                raise RuntimeError(f"token pool exhausted before reaching length {n}")
            seqs.append(seq_ids)
        per_layer_result = run_layers_forward(model, layers, qpos, seqs, device, f"{np_id}/n{n}")
        H_means = [v["H_mean"] for v in per_layer_result.values() if v]
        delta_means = [v["delta_mean"] for v in per_layer_result.values() if v]
        delta_sds = [v["delta_sd"] for v in per_layer_result.values() if v]
        all_layer_avg = {
            "H_mean": float(np.mean(H_means)) if H_means else None,
            "delta_mean": float(np.mean(delta_means)) if delta_means else None,
            "delta_sd": float(np.mean(delta_sds)) if delta_sds else None,
        }
        per_length[str(n)] = {
            "n_sequences": n_seqs,
            "query_positions": qpos,
            "per_layer": {str(li): v for li, v in per_layer_result.items()},
            "all_layer_avg": all_layer_avg,
        }
        print(f"[{np_id}] length={n} done: all_layer_avg={all_layer_avg}", flush=True)

    ln_ns = np.log(lengths)
    H_bars = [per_length[str(n)]["all_layer_avg"]["H_mean"] for n in lengths]
    r_block = None
    if len(lengths) >= 2 and all(h is not None for h in H_bars):
        slope = float(np.polyfit(ln_ns, H_bars, 1)[0])
        delta_hat = slope
        r_block = {
            "lengths": lengths,
            "ln_lengths": ln_ns.tolist(),
            "H_bar_all_layer_avg": H_bars,
            "ols_slope_nats_per_ln_n": slope,
            "delta_hat": delta_hat,
            "R": 1.0 - delta_hat,
            "note": "R := 1 - delta_hat; delta_hat = OLS slope of H_bar (all-layer-avg, nats) "
            "vs ln(n) over the measured lengths. Uniform dispersion (H~ln n) -> "
            "delta_hat=1 -> R=0; retained concentration (H flat in n) -> delta_hat=0 -> R=1.",
        }
    return {
        "lengths": lengths,
        "n_sequences_per_length": n_seqs,
        "per_length": per_length,
        "R": r_block,
    }


def measure_code_input(model, tokenizer, device, np_id):
    """P0-d delta-only mode: single ~4000-token code sequence, delta pooled
    over all heads x all sampled query positions x all layers (mirrors
    2410.01104's "across all attention heads" pooling; per-row delta
    definition and position sampling reuse the same AttnStatsHook / log-spaced
    positions as the neutral mode for methodological consistency -- the
    original paper does not specify a position-sampling scheme, this is a
    judgment call, recorded here for researcher review).

    Recovery-run addition (2026-07-12, per the internal recovery-run protocol,
    Job 2): POS_SCHEME env var selects an alternate query-position scheme for
    the sensitivity check on position aggregation (unset behaves exactly as
    before -- logspace). TRUNC_TOKENS optionally truncates the code input
    further (applied after the existing CODE_INPUT_TARGET_TOKENS cap). delta
    computation (5 lines), clipping, and fp32-eager loading are all
    unchanged."""
    t0 = time.time()
    with urllib.request.urlopen(CODE_INPUT_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:CODE_INPUT_TARGET_TOKENS]
    trunc_tokens = os.environ.get("TRUNC_TOKENS")
    if trunc_tokens:
        ids = ids[: int(trunc_tokens)]
    n = len(ids)
    print(f"[{np_id}] code input fetched: {n} tokens from {CODE_INPUT_URL}", flush=True)

    pos_scheme = os.environ.get("POS_SCHEME", "logspace")
    if pos_scheme == "uniform-all":
        qpos = list(range(1, n))
    elif pos_scheme == "last-only":
        qpos = [n - 1]
    elif pos_scheme == "logspace":
        qpos = get_query_positions(n)
    else:
        raise ValueError(f"unknown POS_SCHEME={pos_scheme!r}")

    layers = get_decoder_layers(model)
    per_layer_result = run_layers_forward(model, layers, qpos, [ids], device, f"{np_id}/code")

    pooled_delta_sum = 0.0
    pooled_delta_sq_sum = 0.0
    pooled_n = 0
    pooled_clip_count = 0
    for v in per_layer_result.values():
        if v is None:
            continue
        n_rows = v["n_rows"]
        for head_mean in v["delta_mean_per_head"]:
            pooled_delta_sum += head_mean * n_rows
            pooled_n += n_rows
        for head_mean, head_sd in zip(v["delta_mean_per_head"], v["delta_sd_per_head"]):
            # E[X^2] per head = var + mean^2, weight by n_rows for that head
            pooled_delta_sq_sum += (head_sd**2 + head_mean**2) * n_rows
        pooled_clip_count += v["clip_count"]

    pooled_mean = pooled_delta_sum / pooled_n if pooled_n else None
    pooled_var = (pooled_delta_sq_sum / pooled_n - pooled_mean**2) if pooled_n else None
    pooled_sd = (max(pooled_var, 0.0) ** 0.5) if pooled_var is not None else None

    return {
        "source_url": CODE_INPUT_URL,
        "n_tokens": n,
        "pos_scheme": pos_scheme,
        "trunc_tokens_applied": int(trunc_tokens) if trunc_tokens else None,
        "n_query_positions": len(qpos),
        "query_positions": qpos,
        "position_sampling_note": "log-spaced 64 positions from min_pos=32, same scheme as "
        "neutral-corpus R measurement -- not specified in the source paper, a methodological "
        "choice made here for consistency/efficiency; flagged for researcher review.",
        "per_layer": {str(li): v for li, v in per_layer_result.items()},
        "delta_pooled_mean": pooled_mean,
        "delta_pooled_sd": pooled_sd,
        "delta_pooled_n_rows_total": pooled_n,
        "clip_count_total": pooled_clip_count,
        "anchor_2410_01104": {
            "gemma_2b": {"range": [2.28, 14.78], "mean": 5.69, "sd": 2.05},
            "gemma_7b": {"range": [0.09, 32.74], "mean": 5.82, "sd": 2.61},
        },
        "elapsed_s": time.time() - t0,
    }


def load_model_and_tokenizer(hf_name, device):
    """Prefer transformers' native implementation (trust_remote_code=False);
    fall back to the repo's bundled custom code (trust_remote_code=True,
    user-approved for this run) only if native loading fails. An outdated
    modeling_*.py shadowing a newer, compatible native implementation was
    observed with Phi-3-mini: native Phi3ForCausalLM loads fine, but the
    repo's custom code calls a transformers API that no longer exists."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_name, dtype=torch.bfloat16, device_map=device, attn_implementation="eager", trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=False)
        return model, tokenizer
    except Exception as e:
        print(
            f"native load failed ({type(e).__name__}: {e}); retrying with trust_remote_code=True",
            flush=True,
        )
        # Defensive config patch: some bundled custom modeling_*.py (observed
        # with ChatGLM variants) reference config.max_length, which is absent
        # from the shipped configuration_*.py -- config.seq_length carries
        # the equivalent value there. This does not change measurement
        # logic, only makes model loading succeed.
        config = AutoConfig.from_pretrained(hf_name, trust_remote_code=True)
        if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
            print(
                f"config missing max_length; injecting max_length=seq_length={config.seq_length}",
                flush=True,
            )
            config.max_length = config.seq_length
        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_name, config=config, dtype=torch.bfloat16, device_map=device, attn_implementation="eager", trust_remote_code=True
            )
        except AttributeError as e2:
            # Some bundled custom modeling_*.py (observed with ChatGLM
            # variants) never call PreTrainedModel.post_init(), which is
            # where transformers >=5.x sets all_tied_weights_keys (consumed
            # by device_map-based loading's caching_allocator_warmup). That
            # warmup path only runs when device_map is not None, so loading
            # onto CPU first and moving to device after sidesteps it without
            # touching what gets measured.
            print(
                f"device_map load failed ({type(e2).__name__}: {e2}); retrying without device_map",
                flush=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                hf_name, config=config, dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True
            )
            model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        return model, tokenizer


def main():
    np_id, hf_name, out_path = sys.argv[1:4]
    t0 = time.time()
    device = "cuda"

    model, tokenizer = load_model_and_tokenizer(hf_name, device)
    model.eval()
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    n_layers = int(text_cfg.num_hidden_layers)
    d_model = int(text_cfg.hidden_size)
    print(
        f"[{np_id}] model loaded (bf16, eager) in {time.time() - t0:.1f}s n_layers={n_layers} d_model={d_model}",
        flush=True,
    )

    result = {
        "np_model_id": np_id,
        "hf_model_name": hf_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "smoke_mode": SMOKE,
        "mode": "code_input" if CODE_INPUT else "neutral",
    }

    if CODE_INPUT:
        result["code_input"] = measure_code_input(model, tokenizer, device, np_id)
    else:
        result["neutral"] = measure_neutral(model, tokenizer, device, np_id)

    result["elapsed_s"] = time.time() - t0
    out = out_path if not SMOKE else out_path.replace(".json", ".smoke.json")
    with open(out, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"[{np_id}] total time {result['elapsed_s']:.1f}s -> {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
