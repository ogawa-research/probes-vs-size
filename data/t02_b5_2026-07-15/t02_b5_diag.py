"""T02 B5 Job1 diagnostic: for gemma-3-4b at seq_len=2048, one forward pass,
record per-layer non-finite occurrence over ALL causal query positions
(not just the 64 log-spaced positions r_delta_probe.py samples), to answer:
  (a) does non-finite appear only on sliding_attention layers (not
      full_attention / global layers)?
  (b) is it confined to query positions > sliding_window (1024)?
  (c) what is the actual non-finite value (nan vs +-inf)?

Mechanical diagnostic only per t02_b5_expansion_instructions_2026-07-15.md
sec 1.1 -- a few lines of log, interpretation left to the researcher.
"""
import json
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/t02")
sys.path.insert(0, "/workspace/t02/jacobian-lens")
from jlens.examples import load_wikitext_prompts  # noqa: E402

SEQ_LEN = 2048
HF_NAME = "google/gemma-3-4b-pt"


class DiagHook:
    def __init__(self, layer_idx, layer_type):
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.first_nonfinite_pos = None
        self.n_nonfinite_rows = 0
        self.n_total_rows = 0
        self.example_value = None
        self.example_kind = None

    def __call__(self, module, inputs, output):
        aw = output[1][0]  # [n_heads, seq_len, seq_len]
        n_heads, seq_len, _ = aw.shape
        # causal range: for each query position p, only [0, p] is valid/used
        for p in range(seq_len):
            row = aw[:, p, : p + 1]  # [n_heads, p+1]
            finite = torch.isfinite(row)
            self.n_total_rows += n_heads
            n_bad = int((~finite).sum().item())
            if n_bad > 0:
                self.n_nonfinite_rows += n_bad
                if self.first_nonfinite_pos is None:
                    self.first_nonfinite_pos = p
                    bad_idx = (~finite).nonzero()[0]
                    v = row[bad_idx[0], bad_idx[1]].item()
                    self.example_value = v
                    if v != v:
                        self.example_kind = "nan"
                    elif v == float("inf"):
                        self.example_kind = "+inf"
                    elif v == float("-inf"):
                        self.example_kind = "-inf"
                    else:
                        self.example_kind = "other"
        del aw, output

    def summarize(self):
        return {
            "layer_idx": self.layer_idx,
            "layer_type": self.layer_type,
            "first_nonfinite_query_pos": self.first_nonfinite_pos,
            "n_nonfinite_rows": self.n_nonfinite_rows,
            "n_total_rows": self.n_total_rows,
            "example_value_repr": repr(self.example_value),
            "example_kind": self.example_kind,
        }


def main():
    cfg = AutoConfig.from_pretrained(HF_NAME)
    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = list(getattr(text_cfg, "layer_types", []))
    sliding_window = getattr(text_cfg, "sliding_window", None)
    print(f"sliding_window={sliding_window} n_layers={text_cfg.num_hidden_layers}", flush=True)
    print(f"layer_types={layer_types}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        HF_NAME, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
    model.eval()

    prompts = load_wikitext_prompts(30, min_chars=600)
    text = "\n\n".join(prompts)
    ids = tokenizer.encode(text, add_special_tokens=False)[:SEQ_LEN]
    print(f"seq_len actually used: {len(ids)}", flush=True)

    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    hooks = {}
    handles = []
    for li, layer in enumerate(layers):
        lt = layer_types[li] if li < len(layer_types) else "unknown"
        h = DiagHook(li, lt)
        hooks[li] = h
        handles.append(layer.self_attn.register_forward_hook(h))

    try:
        with torch.no_grad():
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()

    results = [hooks[li].summarize() for li in sorted(hooks.keys())]
    out = {
        "hf_name": HF_NAME,
        "seq_len": len(ids),
        "sliding_window": sliding_window,
        "per_layer": results,
    }
    with open("/workspace/t02/out/diag_gemma3_4b_nonfinite.json", "w") as f:
        json.dump(out, f, indent=2)

    print("=== SUMMARY ===", flush=True)
    for r in results:
        if r["n_nonfinite_rows"] > 0:
            print(
                f"layer {r['layer_idx']} ({r['layer_type']}): "
                f"first_nonfinite_pos={r['first_nonfinite_query_pos']} "
                f"n_nonfinite={r['n_nonfinite_rows']}/{r['n_total_rows']} "
                f"example={r['example_kind']}({r['example_value_repr']})",
                flush=True,
            )
        else:
            print(f"layer {r['layer_idx']} ({r['layer_type']}): all finite", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
