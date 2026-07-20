# probes-vs-size

Code, data, and frozen protocol/judgement records for the paper:

> **Do Short-Context Mechanistic Probes Predict Long-Context Capability?
> A Pre-Specified Negative Result and the Measurement Science It Required**
> Ryu Ogawa (Keio Research Institute at SFC)
> Paper link: arXiv (to be added upon posting); contact: ryu@sfc.keio.ac.jp

**One-line summary**: across 19 open-weight models in seven families, three
mechanistic probe families proposed in the long-context literature
(attention-entropy resolution, retrieval-head allocation, KV/hidden effective
rank) carry no statistically detectable information about RULER effective
context length beyond model size — and reaching that negative result required
catching two instrument failures that would otherwise have produced
confidently wrong conclusions in either direction.

## Repository structure

```
experiments/t02/scripts/   measurement probes + regression/analysis scripts
data/t02_*/                measurement JSONs, result JSONs, sha manifests
data/t02_*.md              frozen protocol & judgement documents (see below)
data/number_to_source_index.md   extracted number-to-source comment index
ledger.md                  acceptance-gate excerpt (G1–G4) from the run log
LICENSE                    MIT (code)
LICENSE-DATA               CC BY 4.0 (data and documents)
```

The directory layout mirrors the internal research repository, so the
analysis scripts run unmodified from the repository root.

## Reproducing the paper's numbers

The regression and bootstrap scripts are deterministic (fixed seeds) and use
only the Python 3 standard library:

```bash
cd experiments/t02/scripts
python3 b4_regression.py          # B4 primary analysis (N=15)
python3 b5_regression.py          # B5 extension (N=19) — Table 2, same-size contrast
python3 b5_bootstrap_ci.py        # replacement-gap bootstrap CI
python3 b6_crosscheck_analysis.py # correlations, joint models, incremental CI, sensitivity reruns
```

No third-party packages are required for those four paper-number reproduction
commands. The repository-level `requirements.txt` instead records the pinned
environment used by the included GPU measurement probes. That environment used
Python 3.10 or newer and the CUDA 12.8 build of PyTorch 2.8.0; install the
platform-appropriate PyTorch wheel before installing the remaining requirements.
The external `jlens` package is fixed to the exact Git commit used in the runs.

Two narrowly scoped utilities are intentionally outside that recorded
environment: `make_fig_main.py` imports Matplotlib, and a subset of model-specific
tokenizers may require SentencePiece. Their historical package versions were not
frozen, so this release does not invent exact pins for them. Neither is needed by
the four commands above.

Every statistic in the paper's Part-B analysis is reproduced by these four
scripts from the JSONs in `data/`: the pre-specified B4/B5 regressions and
permutation tests, plus the post-hoc supplements (replacement bootstrap CI,
N=19 correlations, joint/incremental models, selection-corrected permutation,
sensitivity reruns), which the paper and the scripts' own headers label as
post hoc relative to the frozen B4/B5 protocol. The remaining paper numbers
(the Part A capacity table and γ regressions, the RULER effective-length
ladder, the acceptance-gate excerpts) are read directly from the frozen JSONs
and judgement documents pointed to below. A quick pointer table:

| Paper statistic | Source |
|---|---|
| LOFO-CV MAE table (Table 2) | `data/t02_b5_2026-07-15/b5_results.json` → `main` |
| Replacement bootstrap CI (±1.2×) | `data/t02_b5_2026-07-15/b5_bootstrap_ci.json` |
| Joint (size+probe) models, selection-corrected permutation (p≈0.50), conditional size+R incremental CI (≈1.12×), N=19 correlations, threshold/imputation reruns | `data/t02_b6_2026-07-20/b6_results.json` |
| Same-size (8B / 7B) contrast | `data/t02_b5_2026-07-15/b5_results.json` → `same_size_groups` |
| RULER effective-length ladder (App. D) | `data/t02_b3_ruler_2026-07-12/ruler_scores_*.json` + `data/t02_b3_judgement_2026-07-15.md` |
| Part A capacity table and γ regressions | `data/t02_partA_2026-07-10/` + `data/t02_partA_final_judgement_2026-07-10.md` |
| R-probe fix acceptance gates (App. G) | `ledger.md` |
| answer_prefix effect (Sec. 3) | `ruler_scores_*_noprefix.json` + `data/t02_ruler_issue_supplement_2026-07-17.md` |

Format note: ten measurement JSONs contain bare `NaN`/`Infinity` tokens
(Python `json` default output) where a probe genuinely produced non-finite
values — notably the pre-fix Gemma-3 `r_delta` files at the center of the
paper's instrument-failure account, and the disclosed `uniform-all` δ
variant. The values are preserved exactly as measured; Python's default
`json` module (which the analysis scripts use) reads them, strict RFC 8259
parsers will not. The release build enforces that exactly these ten files,
and no others, carry non-finite tokens.

The full comment-level index extracted from the manuscript source is in
`data/number_to_source_index.md`.

The measurement probes themselves (`r_delta_probe.py`, `a_retrieval_probe.py`,
`erank_pr_probe.py`, `partA_capacity.py`, ...) are the exact scripts that
produced the measurement JSONs, including the compute-environment paths they
actually ran with (`/workspace/t02/...`); to re-run measurements, edit those
paths for your environment and provide the listed HF checkpoints. GPU types
used per phase are disclosed in the judgement documents (A100 for B2, H200
for B3/B5; the cross-GPU numeric effect was measured at ≤0.5% per layer and
bounded by gate G2a, see `ledger.md`). Four JSONs in the B2 directory came
from a follow-up H200 re-run under transformers 5.13.1 — a patch-version
difference from the pinned 5.13.0, reviewed with no evidence of a numerically
relevant change: `a_retrieval_yi-34b.json` (re-run after an out-of-memory
failure) and the δ position-aggregation sensitivity variants
`r_delta_gemma2b_code_sens_{a,b,c}.json`. All other probe runs used the
`requirements.txt` environment; the exception is also noted there.

## Frozen protocol and judgement documents

The paper's "pre-specified" claims are backed by the frozen internal
documents included under `data/` (design documents fixed before data
collection; judgement records applying the frozen rules). They are in
Japanese (the project's working language); every number cited in the paper is
also machine-readable in the JSONs, and each document carries a release note
stating exactly what was redacted (compute-resource identifiers, cost
estimates, operational sections) and the sha256 of its unredacted frozen
original, which is available on request.

**Role labels.** In these documents, AI-assistant model names are replaced
with role labels: **研究者** (researcher — adjudication, protocol design, and
adversarial review), **エンジニア** (engineer — implementation, execution,
literature collection, and record-keeping), and **監査役** (auditor —
adversarial literature scanning and independent audit). All three roles were
performed by AI assistants working under the direction of the human principal
investigator (**PI**). See the AI-assistance statement in the paper.

**Verification-status note.** The open verification items listed inside the
literature packs (sections marked "未確認/要確認") were closed during
manuscript drafting: every claim the paper relies on was either verified
against primary sources or rewritten to not depend on the unverified item.
The only remaining optional item is a full-PDF cross-check of one cited
OpenReview paper (its forum page — abstract, meta-review, and author
rebuttal — was verified directly and is the only evidence the paper uses).

## Upstream RULER issue

While measuring, we found that the official RULER harness's standalone
prediction path has silently dropped the `answer_prefix` prompt component
since a January 2025 refactor, collapsing base-model scores. Issue and
one-line fix: https://github.com/NVIDIA/RULER/issues/107. Full per-task score
tables in both conditions: `data/t02_ruler_issue_supplement_2026-07-17.md`
plus the `*_noprefix*` score JSONs.

## Raw prediction archives (Zenodo)

The raw generation outputs behind the RULER score matrices (~5.9 GB of
`*_pred.tar.gz`) exceed repository limits and are being deposited to Zenodo;
the DOI will be added here. Their sha256 checksums are already recorded in
`data/t02_b3_ruler_2026-07-12/SHA256SUMS*.txt`, so the deposit can be
verified against this repository once posted. Those manifests are per-stage
captures from the measurement runs: entries for the raw archives are final,
but a few score-JSON entries in earlier-stage manifests are stale because the
score files were later extended in place (64K/128K rows, the `answer_prefix`
patch rerun). For every model the latest-stage manifest (`*_patched` /
`*_128k`, or the sole manifest where only one exists) matches the shipped
score JSON; the frozen manifests themselves are not rewritten for release.

## Notes on redaction

Informational note strings inside a few measurement/result JSONs and script
comments were redacted at release time: an AI-assistant model name and one
internal review-document filename were replaced by role labels / neutral
references, and `pre-registered` was calibrated to `pre-specified` in script
headers/comments and one JSON note string, matching the paper's terminology
(the protocols were frozen internally before the runs; no external registry
was involved). All numeric content is byte-identical to the internal
originals. The release build script enforces this mechanically
(declared-edit-only diffs plus a forbidden-token sweep over every file).

## AI assistance

AI assistance was used extensively in this work, including literature search,
measurement-code implementation, data analysis, manuscript drafting, and
internal adversarial review; study direction and final decisions were the
author's, and the author takes full responsibility for all content.

## License

Code: MIT (see `LICENSE`). Data and documents: CC BY 4.0 (see
`LICENSE-DATA`). If you use the Gemma-3 RULER ladder or the measurement
apparatus, please cite the paper.

## Citation

```bibtex
@article{ogawa2026probes,
  title   = {Do Short-Context Mechanistic Probes Predict Long-Context
             Capability? A Pre-Specified Negative Result and the Measurement
             Science It Required},
  author  = {Ogawa, Ryu},
  year    = {2026},
  note    = {arXiv identifier to be added}
}
```
