"""T02 Part B a_retrieval_probe: retrieval-head measurement (A factor).

Spec: t02_partB_b1b2_instructions_2026-07-11.md sec "a_retrieval_probe.py (A因子)".
Mechanical execution only.

Retrieval score s_h per head follows 2404.15574 Equation 1's two copy-paste
criteria: (1) the generated token w is a needle token; (2) the input position
receiving max attention mass from head h is inside the needle and its token
equals w. Measured under teacher forcing (the correct answer is appended and
its generation-position attention/logits inspected in a single forward pass)
rather than autoregressive greedy decoding -- per
t02_partB_design_2026-07-11.md sec "A", this targets the answer-generation
regime directly instead of requiring the model to reach the correct answer on
its own (the lesson from S3's TF-probe position mismatch). Attention-side
condition (2) is read via a forward hook on each decoder layer's self_attn
module (attn_implementation="eager"); the logits-side condition (argmax of
the output logits at that position equals the needle token, an explicit
addition in the instructions beyond the two criteria above) is read from the
model's final output after the forward pass completes.

Usage:
  python3 a_retrieval_probe.py <np_id> <hf_name> <out_path> [--smoke]
  python3 a_retrieval_probe.py <np_id> <hf_name> <out_path> --p0d-extended   (Llama-3.1-8B only)
"""
import json
import os
import random
import sys
import time
import urllib.request
import uuid as uuid_lib

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/t02")
sys.path.insert(0, "/workspace/t02/jacobian-lens")
from jlens.examples import load_wikitext_prompts  # noqa: E402

CONTEXT_LEN_STD = 4096
DEPTHS_STD = [0.1, 0.3, 0.5, 0.7, 0.9]
N_SAMPLES_STD = 10
NEEDLE_KINDS = ["uuid", "text"]

P0D_LENGTHS = [1024, 2048, 4096, 8192]
P0D_DEPTHS = [round(0.1 * i, 1) for i in range(1, 11)]
P0D_N_SEEDS = 3

CITIES = [
    "Paris", "Tokyo", "Berlin", "Cairo", "Lima", "Oslo", "Manila", "Nairobi",
    "Warsaw", "Havana", "Dublin", "Amman", "Quito", "Sofia", "Zagreb", "Tirana",
    "Riga", "Minsk", "Baku", "Dakar", "Bogota", "Hanoi", "Accra", "Muscat",
]
ACTIVITIES = [
    "eating a sandwich in the park", "watching the sunset from the hill",
    "visiting the old cathedral", "walking along the river",
    "trying the street food market", "riding the old tram",
    "browsing the used bookstore", "climbing the clock tower",
    "swimming in the public pool", "sketching the harbor",
    "feeding the pigeons in the square", "touring the botanical garden",
]

SMOKE = "--smoke" in sys.argv
P0D_EXTENDED = "--p0d-extended" in sys.argv


def get_decoder_layers(model):
    if hasattr(model.model, "layers"):
        return model.model.layers
    return model.model.language_model.layers


def build_filler_pool(tokenizer, min_tokens):
    n_prompts = 60
    ids = []
    while len(ids) < min_tokens:
        prompts = load_wikitext_prompts(n_prompts, min_chars=600)
        text = "\n\n".join(prompts)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < min_tokens:
            n_prompts *= 2
        if n_prompts > 8000:
            raise RuntimeError(f"could not build a filler pool of {min_tokens} tokens (got {len(ids)})")
    return ids


def make_needle(kind, rng):
    city = rng.choice(CITIES)
    if kind == "uuid":
        val = str(uuid_lib.UUID(int=rng.getrandbits(128)))
        needle_text = f"The special magic number for {city} is: {val}."
        question = f"What is the special magic number for {city}?"
        answer_text = val
    else:
        activity = rng.choice(ACTIVITIES)
        needle_text = f"The best thing to do in {city} is {activity}."
        question = f"What is the best thing to do in {city}?"
        answer_text = activity
    return needle_text, question, answer_text


def build_instance(tokenizer, filler_pool, ptr, context_len, depth, kind, rng):
    needle_text, question, answer_text = make_needle(kind, rng)
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    insert_pos = int(round(depth * (context_len - len(needle_ids))))
    insert_pos = max(0, min(insert_pos, context_len - len(needle_ids)))

    filler_ids = filler_pool[ptr : ptr + context_len]
    new_ptr = ptr + context_len
    if len(filler_ids) < context_len:
        raise RuntimeError("filler pool exhausted before reaching requested context_len")
    context_ids = filler_ids[:insert_pos] + needle_ids + filler_ids[insert_pos : context_len - len(needle_ids)]
    needle_span = (insert_pos, insert_pos + len(needle_ids))

    question_ids = tokenizer.encode("\n\nQuestion: " + question + "\nAnswer:", add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer_text, add_special_tokens=False)
    if len(answer_ids) == 0:
        raise RuntimeError(f"empty answer tokenization for {answer_text!r}")

    full_ids = context_ids + question_ids + answer_ids
    answer_start = len(context_ids) + len(question_ids)
    return {
        "input_ids": full_ids,
        "needle_span": needle_span,
        "answer_start": answer_start,
        "answer_ids": answer_ids,
        "new_ptr": new_ptr,
        "context_len": context_len,
        "depth": depth,
        "kind": kind,
    }


class RetrievalStatsHook:
    """One instance per decoder layer per forward call. Records, for each
    head and each answer-generation position, whether the attention argmax
    (causal-range) falls inside the needle span AND points at the token equal
    to the correct next answer token (2404.15574 Eq 1 criteria (1)+(2)).
    Criterion (the instructions' added logits check) is combined afterward in
    run_instance using the model's final logits, not inside this hook."""

    def __init__(self, answer_positions, needle_span, answer_token_ids, full_ids):
        self.answer_positions = answer_positions
        self.needle_span = needle_span
        self.answer_token_ids = answer_token_ids
        self.full_ids = full_ids
        self.n_heads = None
        self.attn_condition_hits = None  # [n_heads, n_answer_positions] bool

    def __call__(self, module, inputs, output):
        attn_weights = output[1]
        if attn_weights is None:
            raise RuntimeError("attn_weights is None -- eager attention not returning weights")
        aw = attn_weights[0]  # [n_heads, seq_len, seq_len]
        n_heads = aw.shape[0]
        n_pos = len(self.answer_positions)
        if self.n_heads is None:
            self.n_heads = n_heads
            self.attn_condition_hits = torch.zeros(n_heads, n_pos, dtype=torch.bool)
        lo, hi = self.needle_span
        for i, p in enumerate(self.answer_positions):
            row = aw[:, p, : p + 1]  # [n_heads, p+1] causal range
            j_star = row.argmax(dim=-1).tolist()  # [n_heads]
            correct_tok = self.answer_token_ids[i]
            for h, j in enumerate(j_star):
                if lo <= j < hi and self.full_ids[j] == correct_tok:
                    self.attn_condition_hits[h, i] = True
        del attn_weights, aw, output


def run_instance(model, layers, instance, device):
    full_ids = instance["input_ids"]
    answer_start = instance["answer_start"]
    answer_ids = instance["answer_ids"]
    needle_span = instance["needle_span"]
    n_answer = len(answer_ids)
    answer_positions = [answer_start - 1 + i for i in range(n_answer)]

    hooks = {}
    handles = []
    for li, layer in enumerate(layers):
        h = RetrievalStatsHook(answer_positions, needle_span, answer_ids, full_ids)
        hooks[li] = h
        handles.append(layer.self_attn.register_forward_hook(h))

    try:
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
    finally:
        for h in handles:
            h.remove()

    logits = outputs.logits[0]  # [seq_len, vocab]
    logits_condition = []
    for i, p in enumerate(answer_positions):
        pred = int(logits[p].argmax().item())
        logits_condition.append(pred == answer_ids[i])
    logits_condition_t = torch.tensor(logits_condition, dtype=torch.bool)  # [n_answer]
    del outputs, logits
    if device == "cuda":
        torch.cuda.empty_cache()

    per_layer_s_h = {}
    for li, hook in hooks.items():
        if hook.n_heads is None:
            per_layer_s_h[li] = None
            continue
        final_hits = hook.attn_condition_hits & logits_condition_t.unsqueeze(0)  # [n_heads, n_answer]
        s_h = final_hits.float().mean(dim=-1)  # [n_heads]
        per_layer_s_h[li] = s_h.tolist()
    return per_layer_s_h, n_answer


def gini(x):
    """Standard Gini coefficient (Lorenz-curve form), no closed-form given in
    2601.11020 -- own definition, per t02_partB_design_2026-07-11.md sec A."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * np.sum(x)) - (n + 1) / n)


def aggregate(all_instance_results):
    layer_head_sums = {}
    layer_head_counts = {}
    for inst in all_instance_results:
        for li, s_h_list in inst["per_layer_s_h"].items():
            if s_h_list is None:
                continue
            arr = np.array(s_h_list, dtype=np.float64)
            if li not in layer_head_sums:
                layer_head_sums[li] = np.zeros_like(arr)
                layer_head_counts[li] = 0
            layer_head_sums[li] += arr
            layer_head_counts[li] += 1
    layer_head_mean = {li: (layer_head_sums[li] / layer_head_counts[li]) for li in layer_head_sums}

    all_s_h = []
    for li in sorted(layer_head_mean.keys()):
        all_s_h.extend(layer_head_mean[li].tolist())
    all_s_h = np.array(all_s_h, dtype=np.float64)
    n_heads_total = len(all_s_h)

    top_k = max(1, int(round(0.05 * n_heads_total)))
    top_sum = float(np.sort(all_s_h)[::-1][:top_k].sum())
    gini_coef = gini(all_s_h)
    rate_gt_0_1 = float((all_s_h > 0.1).mean()) if n_heads_total else None
    rate_zero = float((all_s_h == 0.0).mean()) if n_heads_total else None

    return {
        "layer_head_mean_s_h": {str(li): v.tolist() for li, v in layer_head_mean.items()},
        "n_heads_total": n_heads_total,
        "top_5pct_head_count": top_k,
        "top_5pct_s_h_sum": top_sum,
        "gini_coefficient": gini_coef,
        "rate_s_h_gt_0.1": rate_gt_0_1,
        "rate_s_h_zero": rate_zero,
    }


def measure_standard(model, tokenizer, device, np_id):
    # Recovery-run OOM fallback (2026-07-12,
    # per the internal recovery-run protocol, Job 1): only used if
    # the standard CONTEXT_LEN_STD=4096 run OOMs even on H200; unset behaves
    # exactly as before. Actual context_len used is always recorded in the
    # returned dict's "context_len" field, so a fallback run is
    # self-documenting in the output JSON.
    context_len = 128 if SMOKE else int(os.environ.get("CONTEXT_LEN_OVERRIDE", CONTEXT_LEN_STD))
    depths = [0.5] if SMOKE else DEPTHS_STD
    n_samples = 2 if SMOKE else N_SAMPLES_STD
    n_total = len(depths) * len(NEEDLE_KINDS) * n_samples

    layers = get_decoder_layers(model)
    filler_pool = build_filler_pool(tokenizer, context_len * n_total + context_len)
    rng = random.Random(20260711)
    ptr = 0
    results = []
    count = 0
    for depth in depths:
        for kind in NEEDLE_KINDS:
            for sample_i in range(n_samples):
                instance = build_instance(tokenizer, filler_pool, ptr, context_len, depth, kind, rng)
                ptr = instance["new_ptr"]
                per_layer_s_h, n_answer = run_instance(model, layers, instance, device)
                results.append(
                    {
                        "depth": depth,
                        "kind": kind,
                        "sample": sample_i,
                        "n_answer_tokens": n_answer,
                        "per_layer_s_h": per_layer_s_h,
                    }
                )
                count += 1
                print(f"[{np_id}] instance {count}/{n_total} depth={depth} kind={kind} sample={sample_i} done", flush=True)
    agg = aggregate(results)
    return {
        "context_len": context_len,
        "depths": depths,
        "needle_kinds": NEEDLE_KINDS,
        "n_samples_per_depth_per_kind": n_samples,
        "n_instances": len(results),
        "aggregate": agg,
        "instances_raw": results,
    }


def measure_p0d_extended(model, tokenizer, device, np_id):
    """Llama-3.1-8B only: lengths {1024,2048,4096,8192} x 10 depths x 3
    needle-seeds, both needle kinds. Per-layer attn_weights are consumed and
    discarded inside RetrievalStatsHook immediately as each decoder layer's
    self_attn forward completes (torch.no_grad(), no output_attentions=True,
    no cross-layer retention) -- this already keeps only one layer's
    attn_weights resident at a time, which is what the instructions' "process
    one layer at a time" note for the 8192 case calls for; no additional
    layer-by-layer re-forward is implemented."""
    lengths = P0D_LENGTHS
    depths = P0D_DEPTHS
    n_seeds = P0D_N_SEEDS
    layers = get_decoder_layers(model)
    max_len = max(lengths)
    n_total = len(lengths) * len(depths) * n_seeds * len(NEEDLE_KINDS)
    # Bugfix (observed at instance 227/240 on Llama-3.1-8B): the pool must
    # cover the SUM of context_len across all instances, not
    # max_len * (n_total/len(lengths)) -- that formula implicitly assumed
    # every instance was max_len long, undercounting by ~43% versus the
    # actual mix of 1024/2048/4096/8192 lengths. Measurement logic
    # (context_len, depth, kind, seed grid; needle construction) is
    # unchanged -- this only enlarges the filler buffer.
    total_tokens_needed = sum(context_len * len(depths) * n_seeds * len(NEEDLE_KINDS) for context_len in lengths)
    filler_pool = build_filler_pool(tokenizer, total_tokens_needed + max_len * 4)
    rng = random.Random(20260711)
    ptr = 0
    results = []
    count = 0
    for context_len in lengths:
        for depth in depths:
            for kind in NEEDLE_KINDS:
                for seed_i in range(n_seeds):
                    instance = build_instance(tokenizer, filler_pool, ptr, context_len, depth, kind, rng)
                    ptr = instance["new_ptr"]
                    per_layer_s_h, n_answer = run_instance(model, layers, instance, device)
                    results.append(
                        {
                            "context_len": context_len,
                            "depth": depth,
                            "kind": kind,
                            "needle_seed": seed_i,
                            "n_answer_tokens": n_answer,
                            "per_layer_s_h": per_layer_s_h,
                        }
                    )
                    count += 1
                    print(
                        f"[{np_id}] p0d instance {count}/{n_total} len={context_len} depth={depth} "
                        f"kind={kind} seed={seed_i} done",
                        flush=True,
                    )
    agg = aggregate(results)
    agg_by_len = {}
    for context_len in lengths:
        subset = [r for r in results if r["context_len"] == context_len]
        agg_by_len[str(context_len)] = aggregate(subset)
    return {
        "lengths": lengths,
        "depths": depths,
        "needle_kinds": NEEDLE_KINDS,
        "n_seeds": n_seeds,
        "n_instances": len(results),
        "aggregate_pooled": agg,
        "aggregate_by_length": agg_by_len,
        "instances_raw": results,
        "anchor_2404_15574": {
            "rate_gt_0.1_range": [0.03, 0.06],
            "rate_zero_range": [0.45, 0.73],
            "retrieval_head_rate_common": 0.05,
        },
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
        "mode": "p0d_extended" if P0D_EXTENDED else "standard",
    }

    if P0D_EXTENDED:
        result["p0d_extended"] = measure_p0d_extended(model, tokenizer, device, np_id)
    else:
        result["standard"] = measure_standard(model, tokenizer, device, np_id)

    result["elapsed_s"] = time.time() - t0
    out = out_path if not SMOKE else out_path.replace(".json", ".smoke.json")
    with open(out, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"[{np_id}] total time {result['elapsed_s']:.1f}s -> {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
