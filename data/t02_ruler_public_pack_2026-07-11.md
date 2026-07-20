# T02 Part B — RULER 公開値パック（一次ソース収集）

[Release note: internal primary-source transcription pack (Japanese), released as part of the frozen
record behind the paper. Operational metadata (compute-resource identifiers,
internal audit-transcript paths, model/configuration names) has been
omitted, and AI-assistant model names replaced with role labels: 研究者 (researcher --- adjudication, protocol design, and adversarial review), which was an AI assistant working under the direction of the human principal investigator (PI); all content, numbers, and verdicts are unmodified from
the frozen original (sha256 0cae1353c0c9bb62a280dc88d5f201cc96d02fca857f63ad6d46c6de1104fb29, available on request). Internal documents
referenced but not included in this release are likewise available on
request.]

_取得日: 2026-07-11（研究者 配下の調査サブエージェント実行・全数値は raw README から機械パースで転記）_

## 0. 5 行サマリ

1. **表の規模**: NVIDIA/RULER GitHub README のリーダーボードは **Llama2-7B 基準行 + 44 モデル行**（うち wAvg ランク付き 34・wAvg「TBD」10）。評価長は 4K/8K/16K/32K/64K/128K の 6 点のみ（256K 以上の評価列はない）。
2. **open weights**: 44 行中 **42 行が open weights**。API-only は Gemini-1.5-pro と GPT-4-1106-preview の 2 行だけ。
3. **Qwen3 の版**: オリジナル版 6 モデル（235B-A22B / 32B / 30B-A3B / 14B / 8B / 4B、いずれも claimed 128K・Qwen3 Tech Report 2505.09388 からの著者報告値）のみ掲載。**2507 更新版（claimed 256K）は 1 行も載っていない**（README 全文 grep: 文字列 "2507" のヒットは EXAONE 論文番号 2507.11407 の 1 箇所のみ・"Instruct-2507" は 0 件・"256K"/"256k" は Jamba-1.5 の 2 行のみ）。
4. **Gemma-3**: **不掲載**（Gemma 系は 1 行もない。"Gem" ヒットは API モデル Gemini-1.5-pro のみ）。
5. **更新状況と不整合**: README（＝表）の最終更新は **2025-10-09**（commit `ab17b785`、"add rulerv2"）で、以降 9 か月新モデル追加なし。author-reported の 6 行で Effective Length 列が閾値法の機械適用と**一致しない**（§5）。

## 1. effective context length の定義（逐語）

### 1a. 論文本文（arXiv 2404.06654 **v3**〔last revised 6 Aug 2024〕§4 "Effective Context Size" 節、HTML 版から逐語抽出）

（"Effective Context Size" は節見出し。以下は本文。arXiv HTML 版のテキスト抽出ではベンチマーク名が "Ruler" と表示される〔スモールキャップ組版由来とみられる・LaTeX ソースは未確認〕——ここでは慣用表記 RULER に正規化した。それ以外は逐語）

> "We notice large performance degradation in all models as we increase input length in RULER. To determine the maximum context size a model can effectively handle, we grade each model with a fixed threshold, passing which indicates satisfactory performance at the length of evaluation. **We use the performance of Llama2-7b model at the 4K context length as the threshold.** We report in Table 3 the maximum length exceeding the threshold as the "effective length" along with the "claimed length"."

### 1b. 論文 Table 3 キャプション（同 v3、逐語）

> "Long Context Performance (%) of selected models evaluated at length from 4K to 128K. Each score is computed by averaging accuracy of 13 tasks in RULER. **The performance exceeding the Llama2-7B performance at 4K (85.6%) is underlined. The effective context length is the maximum length passing this threshold.** Weighted average score (wAvg.) aggregates performance across all context sizes, with the weights linearly increasing (inc) or decreasing (dec) to simulate length distribution of real-world usage. We put the rank of each model in the subscript. More details about the selected models are in Appendix A."（RULER の表記正規化は 1a と同じ・他は逐語）

### 1c. GitHub README（表の直下の説明文、逐語）

> "While all models claim context size of 32k tokens or greater, only half of them can effectively handle sequence length of 32K by exceeding a qualitative threshold, Llama-2-7b performance at 4K (85.6%). The performance exceeding the threshold is <ins>underlined</ins>."

- 閾値数値 = **85.6**（Llama2-7B **chat** の 4K スコア。13 タスク平均）。
- 補足（論文 Appendix F・逐語）: base モデル用には別閾値がある — "We use the performance of Llama2-7b (base) and Llama2-7b (chat) at context length of 4K as the threshold for determining effective context size."（base 閾値 = **79.4**、論文 Table 12 の Llama2-7B (base) 4K 値）。GitHub 表は aligned/chat 系のみで base 閾値は登場しない。
- 定義文は "maximum length **exceeding** the threshold"（1a）と "maximum length **passing** this threshold"（1b）の 2 表現で、**等号を含むかは明文化されていない**。また「4K からの連続超過」要件も明文化されていない（ただし本表の範囲では「最大超過長」と「4K からの連続超過の最大長」は全行一致する — 途切れて再超過する行はない）。

## 2. 出典・バージョン

- **主出典（表の正本）**: GitHub `NVIDIA/RULER` README.md。
  - raw URL（pin 済み）: `https://raw.githubusercontent.com/NVIDIA/RULER/ab17b7853df4/README.md`
  - README.md の最終コミット: **`ab17b7853df4`（2025-10-09T23:35:01Z、"add rulerv2"）**。
  - 取得時（2026-07-11）の main HEAD = `38da79d79519`（2026-06-25）だが、main の README.md は `ab17b7853df4` 時点と **diff ゼロで同一**（本収集で直接確認）。
- **論文版リーダーボード（旧・小規模）**: arXiv 2404.06654 v3 の Table 3 は **18 行構成**（Llama2 基準 + Gemini-1.5-Pro + GPT-4 + open 15：Llama3.1-70B/8B, Qwen2-72B, Command-R-plus, GLM4, GradientAI/Llama3-70B, Mixtral-8x22B, Yi-34B, Phi3-medium, Mistral-v0.2, LWM, DBRX, Together, LongChat, LongAlpaca）。GitHub 表はその**上位互換の拡張版**（Jamba/Qwen2.5-1M/Qwen3/EXAONE/ProLong/MegaBeam/Mistral-Large/Command-R 追加系/Phi3-mini/Mixtral-8x7B/GradientAI-8B/FILM/InternLM2.5/Mistral-Nemo/GLM3/Qwen1.5 を追加）。**本パックは GitHub 最新を主とする**。
- RULERv2 について: README の Updates 節が `rulerv2-ns` ブランチ（新パイプライン）を案内しているが、**rulerv1-ns / rulerv2-ns 両ブランチの README は実行手順のみでリーダーボード表はない**（本収集で両ブランチ raw を直接確認）。公開リーダーボード表は main README のこの 1 枚だけ。

## 3. 公開表の転記（全 45 行・README as of commit `ab17b7853df4`）

凡例:
- **太字** = 原文で `<ins>`（下線）が付くセル = 「閾値 85.6 超え」の印（原文の表現をそのまま転記。§5 の不整合 2 件を含む）。
- モデル名の `*` = 原文ママ（著者報告値。対応は §3a Notes）。
- `TBD` = 原文ママ（wAvg 未計算）。空欄 = 原文にセルなし。
- Jamba-1.5-large 行の `256k`/`>128k`（小文字 k）・Mistral-Nemo のサイズ表記なし・EXAONE-4.0-1.2B の 128K セル `-` は原文ママ。

| Models | Claimed | Effective | 4K | 8K | 16K | 32K | 64K | 128K | Avg. | wAvg. (inc) | wAvg. (dec) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama2 (7B) | 4K |  | 85.6 |  |  |  |  |  |  |  |  |
| Jamba-1.5-large* (94B/398B) | 256k | >128k | **96.7** | **96.6** | **96.4** | **96.0** | **95.4** | **95.1** | 96.0 | 95.7 **(1st)** | 96.3 **(1st)** |
| Gemini-1.5-pro | 1M | >128K | **96.7** | **95.8** | **96.0** | **95.9** | **95.9** | **94.4** | 95.8 | 95.5 **(2nd)** | 96.1 **(2nd)** |
| Qwen2.5-14B-Instruct-1M* (14B) | 1M | >128K | **97.5** | **97.1** | **94.6** | **94.9** | **94.9** | **92.2** | 95.7 | TBD | TBD |
| Qwen3-235B-A22B* (235B) | 128K | >128K | **97.7** | **97.2** | **96.4** | **95.1** | **93.3** | **90.6** | 95.0 | TBD | TBD |
| Qwen3-14B* (14B) | 128K | >128K | **98.0** | **97.8** | **96.4** | **96.1** | **94.0** | **85.1** | 94.6 | TBD | TBD |
| Jamba-1.5-mini (12B/52B) | 256K | >128K | **95.6** | **95.6** | **94.8** | **94.6** | **92.8** | **90.0** | 93.9 | 93.1 **(3rd)** | 94.8 **(3rd)** |
| Qwen3-32B* (32B) | 128K | >128K | **98.4** | **96.0** | **96.2** | **94.4** | **91.8** | **85.6** | 93.7 | TBD | TBD |
| EXAONE-4.0-32B* (32B) | 128K | >128K | **96.3** | **94.9** | **93.9** | **93.6** | **91.7** | **88.2** | 93.1 | TBD | TBD |
| Qwen2.5-7B-Instruct-1M* (7B) | 1M | >128K | **96.8** | **95.3** | **93.0** | **91.1** | **90.4** | 84.4 | 91.8 | TBD | TBD |
| Qwen3-30B-A3B* (30B) | 128K | 64K | **96.5** | **97.0** | **95.3** | **92.4** | **89.1** | 79.2 | 91.6 | TBD | TBD |
| GPT-4-1106-preview | 128K | 64K | **96.6** | **96.3** | **95.2** | **93.2** | **87.0** | 81.2 | 91.6 | 89.0 **(4th)** | 94.1 **(4th)** |
| Llama3.1 (70B) | 128K | 64K | **96.5** | **95.8** | **95.4** | **94.8** | **88.4** | 66.6 | 89.6 | 85.5 **(10th)** | 93.7 **(5th)** |
| Qwen3-8B* (8B) | 128K | 64K | **96.3** | **96.0** | **91.8** | **91.2** | 82.1 | 77.4 | 89.1 | TBD | TBD |
| Mistral-Large-2411 (123B) | 128K | 64K | **96.4** | **96.3** | **95.3** | **94.0** | **85.9** | 48.1 | 86.0 | 79.5 **(18th)** | 92.5 **(6th)** |
| Command-R-plus-0824 (104B) | 128K | 32K | **96.0** | **95.1** | **94.0** | **92.4** | 85.4 | 64.6 | 87.9 | 83.4 **(13th)** | 92.4 **(7th)** |
| Qwen2 (72B) | 128K | 32K | **96.9** | **96.1** | **94.9** | **94.1** | 79.8 | 53.7 | 85.9 | 79.6 **(17th)** | 92.3 **(8th)** |
| Command-R-plus (104B) | 128K | 32K | **95.6** | **95.2** | **94.2** | **92.0** | 84.3 | 63.1 | 87.4 | 82.7 **(14th)** | 92.1 **(9th)** |
| Command-R-0824 (32B) | 128K | 64K | **94.7** | **93.7** | **93.1** | **90.8** | **86.6** | 74.7 | 88.9 | 86.0 **(8th)** | 91.9 **(10th)** |
| GLM4 (9B) | 1M | 64K | **94.7** | **92.8** | **92.1** | **89.9** | **86.7** | 83.1 | 89.9 | 88.0 **(5th)** | 91.7 **(11th)** |
| Llama3.1 (8B) | 128K | 32K | **95.5** | **93.8** | **91.6** | **87.4** | 84.7 | 77.0 | 88.3 | 85.4 **(11th)** | 91.3 **(12th)** |
| ProLong (8B) | 512K | 32K | **94.5** | **92.5** | **92.3** | **89.3** | 83.2 | 81.6 | 88.9 | 86.6 **(7th)** | 91.2 **(13th)** |
| Command-R (35B) | 128K | 32K | **93.8** | **93.3** | **92.4** | **89.5** | 84.9 | 76.0 | 88.3 | 85.5 **(9th)** | 91.1 **(14th)** |
| MegaBeam-Mistral (7B) | 512K | 32K | **93.8** | **92.5** | **92.0** | **89.2** | 83.7 | 83.7 | 89.1 | 87.3 **(6th)** | 91.0 **(15th)** |
| Mistral-Large-2407 (123B) | 128K | 32K | **96.2** | **96.1** | **95.1** | **93.0** | 78.8 | 23.7 | 80.5 | 70.6 **(24th)** | 90.4 **(16th)** |
| GradientAI/Llama3 (70B) | 1M | 16K | **95.1** | **94.4** | **90.8** | 85.4 | 80.9 | 72.1 | 86.5 | 82.6 **(15th)** | 90.3 **(17th)** |
| Mixtral-8x22B (39B/141B) | 64K | 32K | **95.6** | **94.9** | **93.4** | **90.9** | 84.7 | 31.7 | 81.9 | 73.5 **(22nd)** | 90.3 **(18th)** |
| Yi (34B) | 200K | 32K | **93.3** | **92.2** | **91.3** | **87.5** | 83.2 | 77.3 | 87.5 | 84.8 **(12th)** | 90.1 **(19th)** |
| Qwen3-4B* (4B) | 128K | 64K | **95.1** | **93.6** | **91.0** | **87.8** | 77.8 | 66.0 | 85.2 | TBD | TBD |
| EXAONE-4.0-1.2B* (1.2B) | 64K | 32K | **87.0** | **86.7** | **88.8** | 81.1 | 77.4 | - | 84.2 | TBD | TBD |
| Phi3-mini (3.8B) | 128K | 32K | **92.2** | **91.5** | **90.7** | **87.5** | 80.6 | 66.7 | 84.8 | 80.9 **(16th)** | 88.7 **(20th)** |
| Phi3-medium (14B) | 128K | 32K | **93.3** | **93.2** | **91.1** | **86.8** | 78.6 | 46.1 | 81.5 | 74.8 **(21st)** | 88.3 **(21st)** |
| Mixtral-8x7B (12.9B/46.7B) | 32K | 32K | **94.9** | **92.1** | **92.5** | **85.9** | 72.4 | 44.5 | 80.4 | 72.8 **(23rd)** | 87.9 **(22nd)** |
| GradientAI/Llama3 (8B) | 1M | 16K | **92.8** | **90.3** | **85.7** | 79.9 | 76.3 | 69.5 | 82.4 | 78.5 **(19th)** | 86.3 **(23rd)** |
| FILM-7B* (7B) | 32K | 32K | **92.8** | **88.2** | **88.1** | **86.9** | 70.1 | 27.1 | 75.5 | 66.4 **(26th)** | 84.7 **(24th)** |
| InternLM2.5 (7B) | 1M | 4K | **88.1** | 85.5 | 84.5 | 82.7 | 75.5 | 68.9 | 80.9 | 77.8 **(20th)** | 83.9 **(25th)** |
| Mistral (7B) | 32K | 16K | **93.6** | **91.2** | **87.2** | 75.4 | 49.0 | 13.8 | 68.4 | 55.6 **(28th)** | 81.2 **(26th)** |
| Mistral-Nemo | 128K | 16K | **87.8** | **87.2** | **87.7** | 69.0 | 46.8 | 19.0 | 66.2 | 54.7 **(29th)** | 77.8 **(27th)** |
| GLM3 (6B) | 128K | 4K | **87.8** | 83.4 | 78.6 | 69.9 | 56.0 | 42.0 | 69.6 | 62.0 **(27th)** | 77.2 **(28th)** |
| LWM (7B) | 1M | <4K | 82.3 | 78.4 | 73.7 | 69.1 | 68.1 | 65.0 | 72.8 | 69.9 **(25th)** | 75.7 **(29th)** |
| DBRX (36B/132B) | 32K | 8K | **95.1** | **93.8** | 83.6 | 63.1 | 2.4 | 0.0 | 56.3 | 38.0 **(30th)** | 74.7 **(30th)** |
| Qwen1.5 (72B) | 32K | 8K | **94.9** | **93.8** | 78.0 | 67.8 | 0.0 | 0.0 | 55.7 | 37.5 **(31st)** | 74.0 **(31st)** |
| Together (7B) | 32K | 4K | **88.2** | 81.1 | 69.4 | 63.0 | 0.0 | 0.0 | 50.3 | 33.8 **(32nd)** | 66.7 **(32nd)** |
| LongChat (7B) | 32K | <4K | 84.7 | 79.9 | 70.8 | 59.3 | 0.0 | 0.0 | 49.1 | 33.1 **(33rd)** | 65.2 **(33rd)** |
| LongAlpaca (13B) | 32K | <4K | 60.6 | 57.0 | 56.6 | 43.6 | 0.0 | 0.0 | 36.3 | 24.7 **(34th)** | 47.9 **(34th)** |

### 3a. README Notes（表の脚注・逐語）

> - Jamba-1.5-large results are reported by authors from this [report](https://arxiv.org/pdf/2408.12570).
> - FILM-7B results are reported by authors of this [paper](https://arxiv.org/pdf/2404.16811). They use [YaRN](https://arxiv.org/pdf/2309.00071) without further training for the evaluation length exceeding 32K (64K and 128K). They do not use the one-shot example for the CWE task.
> - Qwen2.5-1M results are reported by authors from this [paper](https://arxiv.org/pdf/2501.15383).
> - Qwen3 results are reported by authors from this [paper](https://arxiv.org/pdf/2505.09388).
> - EXAONE 4.0 results are reported by authors from this [paper](https://arxiv.org/pdf/2507.11407).

（`*` 付き 12 行 = Jamba-1.5-large, Qwen2.5-14B-1M, Qwen3-235B-A22B, Qwen3-14B, Qwen3-32B, EXAONE-4.0-32B, Qwen2.5-7B-1M, Qwen3-30B-A3B, Qwen3-8B, Qwen3-4B, EXAONE-4.0-1.2B, FILM-7B。Jamba-1.5-mini には `*` がない — 原文ママ。）

## 4. open weights のみ抽出表（Part B 横断層候補・42 行）

判定基準 = weights が HuggingFace からダウンロード可能（ライセンス種別は問わない）。除外 = Gemini-1.5-pro・GPT-4-1106-preview（API-only）。基準行 Llama2-7B (chat) も open weights（`meta-llama/Llama-2-7b-chat-hf`、4K=85.6 のみ）。

| Models | HF repo | Claimed | Effective | 4K | 8K | 16K | 32K | 64K | 128K | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|
| Jamba-1.5-large* (94B/398B) | ai21labs/AI21-Jamba-1.5-Large | 256k | >128k | **96.7** | **96.6** | **96.4** | **96.0** | **95.4** | **95.1** | 96.0 |
| Qwen2.5-14B-Instruct-1M* (14B) | Qwen/Qwen2.5-14B-Instruct-1m | 1M | >128K | **97.5** | **97.1** | **94.6** | **94.9** | **94.9** | **92.2** | 95.7 |
| Qwen3-235B-A22B* (235B) | Qwen/Qwen3-235B-A22B | 128K | >128K | **97.7** | **97.2** | **96.4** | **95.1** | **93.3** | **90.6** | 95.0 |
| Qwen3-14B* (14B) | Qwen/Qwen3-14B | 128K | >128K | **98.0** | **97.8** | **96.4** | **96.1** | **94.0** | **85.1** | 94.6 |
| Jamba-1.5-mini (12B/52B) | ai21labs/AI21-Jamba-1.5-Mini | 256K | >128K | **95.6** | **95.6** | **94.8** | **94.6** | **92.8** | **90.0** | 93.9 |
| Qwen3-32B* (32B) | Qwen/Qwen3-32B | 128K | >128K | **98.4** | **96.0** | **96.2** | **94.4** | **91.8** | **85.6** | 93.7 |
| EXAONE-4.0-32B* (32B) | LGAI-EXAONE/EXAONE-4.0-32B | 128K | >128K | **96.3** | **94.9** | **93.9** | **93.6** | **91.7** | **88.2** | 93.1 |
| Qwen2.5-7B-Instruct-1M* (7B) | Qwen/Qwen2.5-7B-Instruct-1M | 1M | >128K | **96.8** | **95.3** | **93.0** | **91.1** | **90.4** | 84.4 | 91.8 |
| Qwen3-30B-A3B* (30B) | Qwen/Qwen3-30B-A3B | 128K | 64K | **96.5** | **97.0** | **95.3** | **92.4** | **89.1** | 79.2 | 91.6 |
| Llama3.1 (70B) | meta-llama/Meta-Llama-3.1-70B-Instruct | 128K | 64K | **96.5** | **95.8** | **95.4** | **94.8** | **88.4** | 66.6 | 89.6 |
| Qwen3-8B* (8B) | Qwen/Qwen3-8B | 128K | 64K | **96.3** | **96.0** | **91.8** | **91.2** | 82.1 | 77.4 | 89.1 |
| Mistral-Large-2411 (123B) | mistralai/Mistral-Large-Instruct-2411 | 128K | 64K | **96.4** | **96.3** | **95.3** | **94.0** | **85.9** | 48.1 | 86.0 |
| Command-R-plus-0824 (104B) | CohereForAI/c4ai-command-r-plus-08-2024 | 128K | 32K | **96.0** | **95.1** | **94.0** | **92.4** | 85.4 | 64.6 | 87.9 |
| Qwen2 (72B) | Qwen/Qwen2-72B-Instruct | 128K | 32K | **96.9** | **96.1** | **94.9** | **94.1** | 79.8 | 53.7 | 85.9 |
| Command-R-plus (104B) | CohereForAI/c4ai-command-r-plus | 128K | 32K | **95.6** | **95.2** | **94.2** | **92.0** | 84.3 | 63.1 | 87.4 |
| Command-R-0824 (32B) | CohereForAI/c4ai-command-r-08-2024 | 128K | 64K | **94.7** | **93.7** | **93.1** | **90.8** | **86.6** | 74.7 | 88.9 |
| GLM4 (9B) | THUDM/glm-4-9b-chat-1m | 1M | 64K | **94.7** | **92.8** | **92.1** | **89.9** | **86.7** | 83.1 | 89.9 |
| Llama3.1 (8B) | meta-llama/Meta-Llama-3.1-8B-Instruct | 128K | 32K | **95.5** | **93.8** | **91.6** | **87.4** | 84.7 | 77.0 | 88.3 |
| ProLong (8B) | princeton-nlp/Llama-3-8B-ProLong-512k-Instruct | 512K | 32K | **94.5** | **92.5** | **92.3** | **89.3** | 83.2 | 81.6 | 88.9 |
| Command-R (35B) | CohereForAI/c4ai-command-r-v01 | 128K | 32K | **93.8** | **93.3** | **92.4** | **89.5** | 84.9 | 76.0 | 88.3 |
| MegaBeam-Mistral (7B) | aws-prototyping/MegaBeam-Mistral-7B-512k | 512K | 32K | **93.8** | **92.5** | **92.0** | **89.2** | 83.7 | 83.7 | 89.1 |
| Mistral-Large-2407 (123B) | mistralai/Mistral-Large-Instruct-2407 | 128K | 32K | **96.2** | **96.1** | **95.1** | **93.0** | 78.8 | 23.7 | 80.5 |
| GradientAI/Llama3 (70B) | gradientai/Llama-3-70B-Instruct-Gradient-1048k | 1M | 16K | **95.1** | **94.4** | **90.8** | 85.4 | 80.9 | 72.1 | 86.5 |
| Mixtral-8x22B (39B/141B) | mistralai/Mixtral-8x22B-instruct-v0.1 | 64K | 32K | **95.6** | **94.9** | **93.4** | **90.9** | 84.7 | 31.7 | 81.9 |
| Yi (34B) | 01-ai/Yi-34B-200K | 200K | 32K | **93.3** | **92.2** | **91.3** | **87.5** | 83.2 | 77.3 | 87.5 |
| Qwen3-4B* (4B) | Qwen/Qwen3-4B | 128K | 64K | **95.1** | **93.6** | **91.0** | **87.8** | 77.8 | 66.0 | 85.2 |
| EXAONE-4.0-1.2B* (1.2B) | LGAI-EXAONE/EXAONE-4.0-1.2B | 64K | 32K | **87.0** | **86.7** | **88.8** | 81.1 | 77.4 | - | 84.2 |
| Phi3-mini (3.8B) | microsoft/Phi-3-mini-128K-instruct | 128K | 32K | **92.2** | **91.5** | **90.7** | **87.5** | 80.6 | 66.7 | 84.8 |
| Phi3-medium (14B) | microsoft/Phi-3-medium-128K-instruct | 128K | 32K | **93.3** | **93.2** | **91.1** | **86.8** | 78.6 | 46.1 | 81.5 |
| Mixtral-8x7B (12.9B/46.7B) | mistralai/Mixtral-8x7B-instruct-v0.1 | 32K | 32K | **94.9** | **92.1** | **92.5** | **85.9** | 72.4 | 44.5 | 80.4 |
| GradientAI/Llama3 (8B) | gradientai/Llama-3-8B-Instruct-Gradient-1048k | 1M | 16K | **92.8** | **90.3** | **85.7** | 79.9 | 76.3 | 69.5 | 82.4 |
| FILM-7B* (7B) | (README リンクは論文。weights は HF `In2Training/FILM-7B` に実在 — HF API 200 で確認) | 32K | 32K | **92.8** | **88.2** | **88.1** | **86.9** | 70.1 | 27.1 | 75.5 |
| InternLM2.5 (7B) | internlm/internlm2_5-7b-chat-1m | 1M | 4K | **88.1** | 85.5 | 84.5 | 82.7 | 75.5 | 68.9 | 80.9 |
| Mistral (7B) | mistralai/Mistral-7B-instruct-v0.2 | 32K | 16K | **93.6** | **91.2** | **87.2** | 75.4 | 49.0 | 13.8 | 68.4 |
| Mistral-Nemo | mistralai/Mistral-Nemo-Instruct-2407 | 128K | 16K | **87.8** | **87.2** | **87.7** | 69.0 | 46.8 | 19.0 | 66.2 |
| GLM3 (6B) | THUDM/chatglm3-6b-128K | 128K | 4K | **87.8** | 83.4 | 78.6 | 69.9 | 56.0 | 42.0 | 69.6 |
| LWM (7B) | LargeWorldModel/LWM-Text-Chat-1M | 1M | <4K | 82.3 | 78.4 | 73.7 | 69.1 | 68.1 | 65.0 | 72.8 |
| DBRX (36B/132B) | databricKs/dbrx-instruct | 32K | 8K | **95.1** | **93.8** | 83.6 | 63.1 | 2.4 | 0.0 | 56.3 |
| Qwen1.5 (72B) | Qwen/Qwen1.5-72B-Chat | 32K | 8K | **94.9** | **93.8** | 78.0 | 67.8 | 0.0 | 0.0 | 55.7 |
| Together (7B) | togethercomputer/Llama-2-7B-32K-instruct | 32K | 4K | **88.2** | 81.1 | 69.4 | 63.0 | 0.0 | 0.0 | 50.3 |
| LongChat (7B) | lmsys/longchat-7b-v1.5-32K | 32K | <4K | 84.7 | 79.9 | 70.8 | 59.3 | 0.0 | 0.0 | 49.1 |
| LongAlpaca (13B) | YuKang/LongAlpaca-13B | 32K | <4K | 60.6 | 57.0 | 56.6 | 43.6 | 0.0 | 0.0 | 36.3 |

## 5. 閾値整合チェック（機械検査の結果・事実）

「effective length = 4K から連続して 85.6 を**超える**最大評価長」を全 44 行に機械適用し、表の Effective Length 列と突合した。**不一致 6 行**（すべて `*` = author-reported 行）:

| モデル | 表の Effective | 機械適用 | 内訳 |
|---|---|---|---|
| Qwen3-14B* | >128K | 64K | 128K=85.1 が下線付きだが 85.6 以下 |
| Qwen3-32B* | >128K | 64K | 128K=85.6 が下線付きだが閾値と同値（"exceeding" でない） |
| Qwen2.5-7B-Instruct-1M* | >128K | 64K | 128K=84.4（下線なし）なのに Effective は >128K |
| Qwen3-8B* | 64K | 32K | 64K=82.1（下線なし）なのに Effective は 64K |
| Qwen3-4B* | 64K | 32K | 64K=77.8（下線なし）なのに Effective は 64K |
| EXAONE-4.0-1.2B* | 32K | 16K | 32K=81.1（下線なし）なのに Effective は 32K |

- 残り 38 行（RULER チーム自走分の全行 + author-reported の Jamba-1.5-large/mini, Qwen2.5-14B-1M, Qwen3-235B-A22B, Qwen3-30B-A3B, EXAONE-4.0-32B, FILM-7B）は機械適用と完全一致。
- 下線マークと「>85.6」の対応も全セル検査済み: 上記 2 セル（Qwen3-14B の 128K=85.1、Qwen3-32B の 128K=85.6）以外に下線の付け誤りはない。
- この 6 行の Effective 値の由来は不明（転載元論文側の自己判定の可能性があるが**未確認・推測**）。

## 6. T02 Part B で使う際の留意点（事実のみ）

1. **両側 censoring がある**: Effective Length は `>128K`（8 行）と `<4K`（3 行）の censored 表記を含む。評価格子は 4K〜128K の 6 点で、`>128K` 行の真の effective 長は本表からは決まらない（claimed 256K/1M のモデルも評価は 128K 止まり）。
2. **格子が粗い**: effective length は {4K, 8K, 16K, 32K, 64K, 128K} + 両側 censored の離散 8 値しか取らない（2 倍刻み）。
3. **数値の混在**: 44 行中 12 行が author-reported（`*`）で、うち 6 行は Effective 列が README 自身の閾値法と不整合（§5）。各長さのスコア列から閾値法を自前で再適用すれば全行を同一規則に揃えられる（その場合 §5 の 6 行は表の値と変わる）。
4. **Qwen3 の版**: 掲載はオリジナル版のみ（表の Claimed 列は「128K」。なお Qwen3 オリジナル版の HF config はネイティブ 32K + YaRN 拡張 131,072 とされるが、この点は**本収集では HF 側を未照合**——表にはこの区別は現れない）。**Qwen3-2507 系・Qwen3-Next 系・Gemma 系・Llama4 等は不掲載**（README 最終更新 2025-10-09 のため）。T02 主戦場の Qwen3 dense 階段（0.6B/1.7B/4B/8B/14B/32B）のうち、**RULER 表にあるのは 4B/8B/14B/32B の 4 点のみ**（0.6B/1.7B はなし）。Gemma-3 系列は全滅。
5. **タスク平均の定義**: 各スコアは「13 タスク平均・recall-based accuracy」（論文 §4）。Qwen3/EXAONE 等の転載値が同一 13 タスク構成かは README Notes からは確認できない（FILM-7B は CWE の one-shot 例を使っていないと明記 = 条件差の実例）。
6. **基準行の非対称**: Llama2-7B 行は 4K=85.6 のみで 8K 以降のセルが存在しない。回帰に含める場合この行だけ構造が異なる。
7. **閾値の等号問題**: 定義は "exceeding"（1a/1c）だが Qwen3-32B の 128K=85.6（同値）に下線が付いており、等号の扱いが原文内で一貫しない（§5）。自前再適用時は等号規約を明示する必要がある。
8. **wAvg ランクは 34 モデル母集団**: TBD 10 行はランク外。inc/dec の重み定義は 1b の逐語どおり（線形増加/減少）。
9. 細部の原文ママ事項: DBRX の HF リンク URL に大文字タイポ（`databricKs`）・Jamba-1.5-mini 行は行末パイプ欠落・Jamba-1.5-large 行のみ小文字 k 表記。転記には影響なし。

---
_収集手順の記録: (1) `raw.githubusercontent.com/NVIDIA/RULER/main/README.md` 取得 → (2) GitHub API で README.md の最終コミット `ab17b7853df4`（2025-10-09）を特定し、pin した raw と main の diff ゼロを確認 → (3) rulerv1-ns / rulerv2-ns ブランチ README も取得（リーダーボードなしを確認）→ (4) arXiv abs ページで論文が v3（2024-08-06）と確認、`arxiv.org/html/2404.06654v3` から定義節・Table 3 キャプション・Appendix F を逐語抽出 → (5) README 表 45 行を Python で機械パース（手写しなし）し、閾値整合を全セル機械検査 → (6) FILM-7B の weights 実在を HF API（HTTP 200）で確認。_
