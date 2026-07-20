# T02 B3 実行指示書 — RULER 自前測定（2026-07-12 起草・B3 測定前に凍結）

[Release note: excerpt of the frozen internal execution protocol (Japanese).
Operational metadata --- compute specifications and cost-gate thresholds
(secs. 0 and 4, stubbed to preserve section numbering) --- has been omitted,
and AI-assistant model names replaced with role labels: 研究者 (researcher ---
adjudication, protocol design, and adversarial review) and エンジニア
(engineer --- implementation, execution, and record-keeping), both of which
were AI assistants working under the direction of the human principal
investigator (PI); all protocol content, numbers, and decision
rules are unmodified from the frozen original (sha256
6a3cf1558c9d3c2894855e2ce8ab99d9cfaec301499a1ee30b71e024593b2f97, available
on request).]

目的 = Part B の目的変数（RULER effective context length）を、公開値のないモデル 7 点で自前測定する。設計正本 = `t02_partB_design_2026-07-11.md` §1/§5 B3（本書の逸脱裁定 2 件は末尾 §7——設計 v1.2 相当の修正として研究者が承認済み）。エンジニアは**機械実行と記録のみ**——スコア行列の閾値法適用・effective length の算出・判定はすべて研究者が後段で行う。

## 0. 計算資源仕様・想定規模

[omitted --- operational compute details]

## 1. 対象モデルと評価格子

因子測定（B2）と**同一の HF チェックポイント**を使う（B2 出力 JSON の `hf_model_name` と一致させること——ここが本測定の存在理由。instruct 版への差し替えは禁止）：

| モデル | HF checkpoint | 格子（context length） | 備考 |
|---|---|---|---|
| qwen3-0.6b | Qwen/Qwen3-0.6B | 4K, 8K, 16K, 32K | claimed 32K・**最初に実行（コスト実測）** |
| qwen3-1.7b | Qwen/Qwen3-1.7B | 4K, 8K, 16K, 32K | claimed 32K |
| gemma-3-270m | google/gemma-3-270m | 4K, 8K, 16K, 32K | claimed 32K |
| gemma-3-1b | google/gemma-3-1b-pt | 4K, 8K, 16K, 32K | claimed 32K |
| gemma-3-4b | google/gemma-3-4b-pt | 4K, 8K, 16K, 32K（+64K 条件付き） | claimed 128K |
| gemma-3-12b | google/gemma-3-12b-pt | 同上 | claimed 128K |
| gemma-3-27b | google/gemma-3-27b-pt | 同上 | claimed 128K |

- **64K の条件付き実行**：gemma-3-4b/12b/27b のうち、32K の 13 タスク平均が **79.4 を超えた**モデルのみ 64K を追加（さらに 64K も超えたら 128K——ただし 128K は 27b の VRAM/時間を見て skip 可・skip したら記録）。閾値超えが出なければ 64K 以降は不要（effective length の定義上、下で切れた時点で先は使わない）。
- 順序 = 表の上から（小 → 大）。**qwen3-0.6b 完了時点で 1 モデル×4 長の実測時間を RUN_LOG に記録**（§4 コストゲートの入力）。

## 2. ハーネス（RULER 公式・改変禁止）

- `git clone https://github.com/NVIDIA/RULER.git` → **clone 時の HEAD commit hash を RUN_LOG に記録し、以後 run 途中で pull しない**（参考: README の転記正本は commit `ab17b785`・2025-10-09 が最終更新）。
- **標準 13 タスク構成・タスクあたり 500 サンプル（公式既定）を変えない**——公開値（Qwen3 4b/8b/14b の NVIDIA 測定行）との比較可能性が Part B の生命線。サンプル数・タスク構成・プロンプトテンプレートの縮約は一切しない。困った場合は改変でなく停止・報告。
- 合成データ生成はモデルごと・長さごとに公式手順どおり（tokenizer 依存のため使い回し不可の箇所は公式の指示に従う）。
- **推論エンジン = vLLM を第一候補**（公式サポート経路・生成が支配的コストのため）。特定モデルが vLLM で動かない場合のみ HF transformers 経路へフォールバックし、**モデル単位でエンジンを混ぜたことを RUN_LOG に記録**（タスク採点は string-match なのでエンジン差の影響は限定的だが開示対象）。
- **生成設定 = greedy**。**Qwen3 は non-thinking モード**（chat template の `enable_thinking=False` 相当）で実行し、設定方法を RUN_LOG に記録。**Gemma-3 の `-pt`/`270m` は base モデルとして chat template なし**（RULER の base モデル扱い・§7 裁定 1 参照）。

## 3. 出力・退避

- 保存する正本 = **タスク×長さのスコア行列（13×長さ点・モデルごと）+ 13 タスク平均**。可能なら公式出力（per-task summary）の生ファイルもまとめて退避。effective length の算出は**しない**（閾値法の機械適用は研究者の判定工程）。
- 出力先 = `data/t02_b3_ruler_2026-07-12/`（新ディレクトリ）：`ruler_scores_<model>.json`（スコア行列・エンジン・commit hash・生成設定込み）+ 生出力 tar + `RUN_LOG.md` + sha256 マニフェスト。ローカル退避後 `shasum -a 256 -c` 照合。
- モデルごとに完了 → HF キャッシュ `rm -rf` → 解放確認（B2 教訓・27b は 54GB 級）。

## 4. コストゲート（自律判断の規約）

[omitted --- operational cost-gate thresholds; the measurement-integrity rule (no sample-count or task reduction by the engineer) is restated in §2]

## 5. sanity（測定の破綻検知・判定ではない）

- qwen3-0.6b の 4K・13 タスク平均が 20 未満（ほぼ全滅）or 99 超（ほぼ満点）の場合、ハーネス設定ミスの疑い——1 タスク分の生成例 2〜3 件を目視できる形で RUN_LOG に貼り、停止して報告。
- Gemma-3 base 系で「指示に全く従わない」形の全滅が 4K で出た場合も同様に生成例を貼って停止・報告（base モデルの RULER 適用は論文 Appendix F にある正規の使い方だが、フォーマット崩壊の水準は目視確認が要る）。

## 6. 禁止事項（B2 と同じ規律）

- 測定ロジック（RULER 公式コード）の変更は禁止。モデル固有の構造対応（ロード・テンプレート指定・エンジン選択）のみ可・すべて RUN_LOG に記録。
- 数値の解釈・合否判定・effective length の宣言はしない（研究者差し戻し）。

## 7. 設計からの逸脱裁定（研究者承認済み・設計 v1.2 相当）

1. **Gemma-3 は `-pt`（base）チェックポイントで測る**。設計 §1 の「chat/instruct 系のみ使うので 85.6 で統一」は、因子測定（B2）が Gemma-3 を全て `-pt` で行っていた事実と衝突する。因子と目的変数のチェックポイント同一性が Part B の主張（機構因子 → 行動の予測）の前提なので、**同一チェックポイント優先**。閾値は RULER 論文 Appendix F の base モデル用 **79.4** を Gemma-3 系に適用し（Qwen3 は 85.6 のまま）、閾値の混在は B4 で開示 + 感度分析（Gemma-3 を 85.6 で切った場合も併記）。この適用は研究者の判定工程で行う——エンジニアはスコア行列を出すだけ。
2. **Gemma-3 階段 5 点は「コスト実測後に確定」から「実施を基本線」へ**（P0-d 判定書 §5.3——横断 8 → 実効 5 の N 減により、Gemma-3 なしの主回帰 11 点/5 families は薄い）。コストゲート §4 は残すが、資源制約は緩い（PI 確認 2026-07-12）。
