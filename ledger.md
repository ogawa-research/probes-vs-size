# Release ledger: R-probe fix acceptance gates (G1–G4, G2a)

This ledger excerpts, verbatim, the acceptance-gate table from the internal
run log of the B5 measurement session (the file referred to as RUN_LOG in the
judgement documents; sha256
`2c31f41778b4e358d16267cb9428a4f6080172b5b37a48c5123591e7894feb7a`, available
on request). These are the gates cited in the paper's measurement-history
appendix. The only redaction is the removal of two compute-resource
identifiers (marked `[compute-resource id redacted]`); every number and
verdict is unchanged. An English translation follows the original.

## Original (Japanese, verbatim except the two marked redactions)

| ゲート | 内容 | 結果 |
|---|---|---|
| G1 | `torch.tensor(1e-38,dtype=bf16).item()>0`・`zeros(4,bf16).clamp(min=1e-38)`全非ゼロ・合成ゼロ混入行でH有限 | **PASS**（1.001e-38・全非ゼロ・H=1.03125）。参考: fp16は1e-38が0にアンダーフローする（今回未使用のため無関係） |
| G2 | qwen3-1.7bをv2(CLIP_FLOOR=1e-38)で再測定しB2記録値とビット同一比較 | 字義不合格（R相対差0.07%・per-layer最大相対差0.48%、B2実行=A100 [compute-resource id redacted]・本セッション=H200 [compute-resource id redacted]とGPU差異あり）→ **G2aへ** |
| G2a | 同一H200上でv1再現版（CLIP_FLOOR=1e-45・assert無効化）とv2を直接比較 | **PASS**（`r_delta_probe_v1_repro.py`使用。R値・per-layer H_mean/delta_mean/delta_sd 全252フィールドが**Python `==` でビット同一**、max abs diff=0.0）。CLIP_FLOOR変更はfull-attentionの数値に無影響と確定・G2の0.48%はGPU差由来と結論 |
| G3 | gemma-3-4bのglobal層（5,11,17,23,29、0-indexed）をB2と相対≤1%で比較（cross-GPUにつき許容緩和） | **PASS**（全15セル、相対差0.010%〜0.325%） |
| G4 | gemma-3-4bのsliding層が全有限・clip_count>0・per-head H≤log(1024)+0.1 | **PASS**（length2048/4096ともnonfinite=0/29層・clip_count=4,909,120/21,362,560・per-head H最大3.52 << log(1024)+0.1=7.03） |

## English translation

| Gate | Check | Result |
|---|---|---|
| G1 | `torch.tensor(1e-38, dtype=bf16).item() > 0`; `zeros(4, bf16).clamp(min=1e-38)` all non-zero; H finite on a synthetic row containing exact zeros | **PASS** (1.001e-38; all non-zero; H = 1.03125). Note: under fp16, 1e-38 underflows to 0 (irrelevant here — fp16 not used) |
| G2 | Re-measure qwen3-1.7b with v2 (CLIP_FLOOR = 1e-38) and compare bit-identically against the recorded B2 values | Literal fail (R relative diff 0.07%; max per-layer relative diff 0.48%; B2 ran on an A100 instance, this session on an H200 — a GPU difference) → **proceed to G2a** |
| G2a | On the same H200, directly compare a v1 reproduction (CLIP_FLOOR = 1e-45, assert disabled) against v2 | **PASS** (using `r_delta_probe_v1_repro.py`; R and all 252 per-layer H_mean / delta_mean / delta_sd fields **bit-identical under Python `==`**, max abs diff = 0.0). Confirms the CLIP_FLOOR change has no effect on full-attention numerics; the 0.48% in G2 is GPU-driven |
| G3 | Compare gemma-3-4b global layers (5, 11, 17, 23, 29; 0-indexed) against B2 within a relative 1% tolerance (relaxed because cross-GPU) | **PASS** (all 15 cells, relative diff 0.010%–0.325%) |
| G4 | gemma-3-4b sliding-window layers: all finite; clip_count > 0; per-head H ≤ log(1024) + 0.1 | **PASS** (lengths 2048/4096: nonfinite = 0/29 layers; clip_count = 4,909,120 / 21,362,560; max per-head H 3.52 ≪ log(1024) + 0.1 = 7.03) |

All gates passed before any of the Gemma R (v2) data used in the paper was
collected under the fixed probe. The diagnostic scripts referenced above
(`r_delta_probe_v1_repro.py`, `t02_b5_diag.py`, `t02_b5_diag2.py`) are
included in `data/t02_b5_2026-07-15/`.
