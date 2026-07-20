"""Shared model-load compatibility shim for hidden-state/logit-only probes
(cka_band.py, erank_pr_probe.py, cross_family_capacity.py).

Ported and extended from r_delta_probe.py / a_retrieval_probe.py's
load_model_and_tokenizer(). Extension validated directly on chatglm3-6b
(2026-07-12, this session): some bundled custom modeling_*.py (ChatGLM
family) never call PreTrainedModel.post_init(), so transformers >=5.13's
all_tied_weights_keys (set inside post_init, consumed during
_finalize_model_loading regardless of device_map) is never populated,
crashing from_pretrained with an AttributeError. The patch below calls
post_init() once right after __init__ if the attribute is still missing --
semantics-preserving (verified: all_tied_weights_keys={} for chatglm3-6b,
matching its no-tied-embeddings design; forward-pass logits sane, NaN/Inf
free, top-5 for "The capital of France is" includes "Paris").

Does not touch attention output extraction -- callers that need
per-head attention weights (r_delta_probe.py, a_retrieval_probe.py) are
NOT covered by this shim; ChatGLM's CoreAttention has a non-standard
forward-hook output contract (KV-cache-shaped, not
[batch, heads, seq, seq]), and building a correct extractor for it would
be a model-specific measurement implementation, not a load-compat patch.
Per the researcher's 2026-07-12 ruling, R/A measurement for the ChatGLM family is
abandoned; only hidden-state/logit-based probes use this shim.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_with_glm_compat(hf_name, device, dtype=torch.bfloat16, attn_implementation=None, cache_dir=None):
    kwargs = {"dtype": dtype, "device_map": device, "trust_remote_code": False}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=False, cache_dir=cache_dir)
        return model, tokenizer, "native"
    except Exception as e:
        print(f"native load failed ({type(e).__name__}: {e}); retrying with trust_remote_code=True", flush=True)
        config = AutoConfig.from_pretrained(hf_name, trust_remote_code=True, cache_dir=cache_dir)
        if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
            print(f"config missing max_length; injecting max_length=seq_length={config.seq_length}", flush=True)
            config.max_length = config.seq_length
        if not hasattr(config, "use_cache"):
            print("config missing use_cache; injecting use_cache=True", flush=True)
            config.use_cache = True
        if hasattr(config, "auto_map") and "AutoModelForCausalLM" in config.auto_map:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            model_class_path = config.auto_map["AutoModelForCausalLM"]
            model_cls = get_class_from_dynamic_module(model_class_path, hf_name)
            if not getattr(model_cls, "_t02_post_init_patched", False):
                original_init = model_cls.__init__

                def patched_init(self, *a, __orig=original_init, **kw):
                    __orig(self, *a, **kw)
                    if not hasattr(self, "all_tied_weights_keys"):
                        self.post_init()
                        print(
                            f"post_init() called for {type(self).__name__} (custom code never calls it)",
                            flush=True,
                        )

                model_cls.__init__ = patched_init
                model_cls._t02_post_init_patched = True
        kwargs2 = dict(kwargs)
        kwargs2["config"] = config
        kwargs2["trust_remote_code"] = True
        if cache_dir:
            kwargs2["cache_dir"] = cache_dir
        model = AutoModelForCausalLM.from_pretrained(hf_name, **kwargs2)
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True, cache_dir=cache_dir)
        return model, tokenizer, "custom_code_patched"
