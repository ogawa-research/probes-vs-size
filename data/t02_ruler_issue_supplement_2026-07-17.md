# RULER issue supplement: full per-task tables (noprefix = current-main behavior vs answer_prefix restored)

[Release note: internal data-table supplement (English), released as part of the frozen
record behind the paper. Operational metadata (compute-resource identifiers,
internal audit-transcript paths, model/configuration names) has been
omitted; all content, numbers, and verdicts are unmodified from
the frozen original (sha256 2a6af334514174343daa7a3ac604c798a4507dd0ad49b145945cfe80a2655521, available on request). Internal documents
referenced but not included in this release are likewise available on
request.]

Source: `data/t02_b3_ruler_2026-07-12/ruler_scores_{model}[_noprefix].json` (500 samples/task, vLLM, greedy; restored = 1-line re-attachment patch). Prepared for maintainer follow-up on the upstream issue.

## gemma-3-270m (base checkpoint)

### context length 4096

| task | no prefix (current main) | prefix restored |
|---|---|---|
| niah_single_1 | 0.0 | 99.8 |
| niah_single_2 | 0.8 | 83.8 |
| niah_single_3 | 0.2 | 23.8 |
| niah_multikey_1 | 1.6 | 85.6 |
| niah_multikey_2 | 1.0 | 26.0 |
| niah_multikey_3 | 3.6 | 3.2 |
| niah_multivalue | 2.35 | 37.5 |
| niah_multiquery | 0.1 | 44.05 |
| vt | 0.24 | 0.6 |
| cwe | 16.02 | 16.62 |
| fwe | 11.13 | 18.8 |
| qa_1 | 27.2 | 25.8 |
| qa_2 | 20.0 | 22.0 |
| **13-task avg** | **6.48** | **37.5054** |

### context length 8192

| task | no prefix (current main) | prefix restored |
|---|---|---|
| niah_single_1 | 0.0 | 100.0 |
| niah_single_2 | 5.8 | 65.0 |
| niah_single_3 | 0.8 | 7.6 |
| niah_multikey_1 | 17.6 | 60.2 |
| niah_multikey_2 | 0.6 | 14.4 |
| niah_multikey_3 | 2.0 | 1.8 |
| niah_multivalue | 2.05 | 26.35 |
| niah_multiquery | 4.85 | 32.4 |
| vt | 0.04 | 0.04 |
| cwe | 14.72 | 19.2 |
| fwe | 25.4 | 41.4 |
| qa_1 | 17.2 | 17.6 |
| qa_2 | 17.8 | 19.2 |
| **13-task avg** | **8.3738** | **31.1685** |

### context length 16384

| task | no prefix (current main) | prefix restored |
|---|---|---|
| niah_single_1 | 0.0 | 99.4 |
| niah_single_2 | 0.8 | 56.8 |
| niah_single_3 | 1.0 | 17.6 |
| niah_multikey_1 | 0.6 | 71.8 |
| niah_multikey_2 | 1.0 | 12.8 |
| niah_multikey_3 | 0.4 | 1.0 |
| niah_multivalue | 13.3 | 29.35 |
| niah_multiquery | 9.9 | 33.05 |
| vt | 0.28 | 0.24 |
| cwe | 8.28 | 10.18 |
| fwe | 9.6 | 6.13 |
| qa_1 | 14.8 | 17.2 |
| qa_2 | 16.4 | 18.4 |
| **13-task avg** | **5.8738** | **28.7654** |

### context length 32768

| task | no prefix (current main) | prefix restored |
|---|---|---|
| niah_single_1 | 31.0 | 90.2 |
| niah_single_2 | 4.4 | 72.6 |
| niah_single_3 | 0.2 | 14.0 |
| niah_multikey_1 | 0.2 | 60.0 |
| niah_multikey_2 | 0.2 | 8.6 |
| niah_multikey_3 | 0.4 | 0.2 |
| niah_multivalue | 3.75 | 22.95 |
| niah_multiquery | 0.3 | 25.0 |
| vt | 0.16 | 2.08 |
| cwe | 3.76 | 3.62 |
| fwe | 17.2 | 16.33 |
| qa_1 | 14.4 | 15.8 |
| qa_2 | 14.4 | 13.6 |
| **13-task avg** | **6.9515** | **26.5369** |

## qwen3-0.6b (base checkpoint)

### context length 4096

| task | no prefix (current main) | prefix restored |
|---|---|---|
| niah_single_1 | 100.0 | 100.0 |
| niah_single_2 | 99.8 | 99.8 |
| niah_single_3 | 99.4 | 99.4 |
| niah_multikey_1 | 99.2 | 99.2 |
| niah_multikey_2 | 93.0 | 91.8 |
| niah_multikey_3 | 81.0 | 87.6 |
| niah_multivalue | 94.35 | 93.45 |
| niah_multiquery | 98.6 | 98.1 |
| vt | 3.48 | 63.6 |
| cwe | 64.28 | 62.7 |
| fwe | 58.27 | 63.07 |
| qa_1 | 56.2 | 55.0 |
| qa_2 | 31.2 | 32.0 |
| **13-task avg** | **75.2908** | **80.44** |

### context length 8192

| task | no prefix (current main) | prefix restored |
|---|---|---|
| niah_single_1 | 100.0 | 100.0 |
| niah_single_2 | 99.8 | 99.8 |
| niah_single_3 | 99.6 | 99.6 |
| niah_multikey_1 | 93.4 | 92.8 |
| niah_multikey_2 | 80.2 | 67.0 |
| niah_multikey_3 | 34.0 | 51.0 |
| niah_multivalue | 90.95 | 90.7 |
| niah_multiquery | 97.6 | 96.6 |
| vt | 1.36 | 71.56 |
| cwe | 2.66 | 5.06 |
| fwe | 74.47 | 75.13 |
| qa_1 | 44.2 | 42.2 |
| qa_2 | 29.6 | 30.6 |
| **13-task avg** | **65.2185** | **70.9269** |

### context length 16384

| task | no prefix (current main) | prefix restored |
|---|---|---|
| niah_single_1 | 100.0 | 100.0 |
| niah_single_2 | 100.0 | 100.0 |
| niah_single_3 | 99.4 | 99.4 |
| niah_multikey_1 | 86.6 | 86.4 |
| niah_multikey_2 | 69.6 | 54.8 |
| niah_multikey_3 | 23.6 | 22.2 |
| niah_multivalue | 85.8 | 85.35 |
| niah_multiquery | 95.3 | 94.6 |
| vt | 1.4 | 70.44 |
| cwe | 0.12 | 0.22 |
| fwe | 74.4 | 77.2 |
| qa_1 | 38.4 | 36.4 |
| qa_2 | 28.4 | 28.0 |
| **13-task avg** | **61.7708** | **65.77** |

### context length 32768

| task | no prefix (current main) | prefix restored |
|---|---|---|
| niah_single_1 | 100.0 | 100.0 |
| niah_single_2 | 100.0 | 100.0 |
| niah_single_3 | 99.4 | 99.6 |
| niah_multikey_1 | 66.6 | 72.8 |
| niah_multikey_2 | 45.2 | 50.6 |
| niah_multikey_3 | 2.2 | 5.4 |
| niah_multivalue | 65.15 | 65.75 |
| niah_multiquery | 86.75 | 84.0 |
| vt | 0.28 | 35.12 |
| cwe | 0.66 | 0.6 |
| fwe | 55.73 | 53.87 |
| qa_1 | 37.4 | 35.6 |
| qa_2 | 24.2 | 25.2 |
| **13-task avg** | **52.5823** | **56.0415** |

