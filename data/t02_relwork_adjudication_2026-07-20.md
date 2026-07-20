# T02 関連研究照合の裁定書（研究者・2026-07-20）

[Release note: internal related-work adjudication record (Japanese), released as part of the frozen
record behind the paper. Operational metadata (compute-resource identifiers,
internal audit-transcript paths, model/configuration names) has been
omitted, and AI-assistant model names replaced with role labels: 研究者 (researcher --- adjudication, protocol design, and adversarial review), エンジニア (engineer --- implementation, execution, literature collection, and record-keeping), and 監査役 (auditor --- adversarial literature scanning and independent audit), all of which were AI assistants working under the direction of the human principal investigator (PI); all content, numbers, and verdicts are unmodified from
the frozen original (sha256 91f0045b5115beafaca9543a2b858f258a58c6810a1a364278ed9c5cb1b133ba, available on request). Internal documents
referenced but not included in this release are likewise available on
request.]

- 席 = 3: 席 1（エンジニア・頭対頭不在確認 22 式）= `t02_relwork_headtohead_pack_2026-07-20.md` / 席 2（エンジニア・R/A/W 原典逐語精査）= `t02_relwork_raw_claims_pack_2026-07-20.md` / 席 3（監査役・敵対スキャン 75 式）= 内部監査記録（本リリースからは省略・on request）。
- 席の出力は未検証入力として扱い、load-bearing 主張は研究者が一次再照合した（§2）。

## §1. 総合裁定

**T02 の中心命題（長文脈能力の RULER effective length に対する機構 probe vs サイズの頭対頭・事前登録否定的結果）を閉じる先行は 3 席いずれも発見せず。ただし 監査役 が挙げた closes_core 候補 1 本の存在により、intro の「不在」主張は一般形では書けない——長文脈限定形へ較正する（§3）。**

## §2. 監査役 closes_core 候補の研究者裁定 = overlaps（must-cite・格下げ）

対象 = **"Learning is Forgetting; LLM Training As Lossy Compression"（Conklin, Hosking, Yi-Chern, Cohen, Leslie, Griffiths, Bartolo, Goldfarb-Tarrant・OpenReview tvDlQj0GZB）**。

研究者の一次照合（2026-07-20・OpenReview forum ページを browser で実読・API/curl は challenge 403）:
- **ICLR 2026 Poster 採択**（"Published: 26 Jan 2026 ... ICLR 2026 Poster"・Decision: Accept (Poster) を実読）。監査役 の URL・存在は真正。
- abstract 逐語: "even across different families of LLMs the optimality of a model's compression, and the information present in it, can predict downstream performance on across a wide array of benchmarks"——**肯定的結果**（大域的情報理論量 = distance-to-bound が下流性能を予測）。
- meta-review 逐語: "neither expressivity nor complexity alone reliably predicts downstream performance, while the expressivity-to-compression ratio (or distance-to-bound) is the meaningful signal"。ベンチは MMLU-Pro が名指し。
- **決定打 = 著者自身の rebuttal 逐語**: "increasing the context window will likely have minimal effect on the core results here. **In future work we hope to systematically study how long-context models differ** in terms of the compressions they converge to"——**長文脈は明示的に future work・forum 全文（50,150 字）に RULER の出現ゼロ**（js 全文検索で確認）。

裁定理由:
1. T02 の中心命題は**長文脈能力**への増分情報。本論文は長文脈を扱わず（著者自認）、対象量も機構 probe（R/A/W = 文献が長文脈に紐づけた局所量）でなく大域圧縮量。closes_core 不成立。
2. ただし「内部量がサイズを超えて下流性能を予測できるか」という**一般形の問い**では、本論文が params 共変量つきで肯定側の主張をする（監査役 が PDF から引いた "a partial spearman that treats a model's number of parameters as a covariate" の句・**偏相関の結果の向き・数値は研究者未照合 = 起草時必須照合 §5-1**）。→ T02 intro の不在主張を一般形で書くと即反例を食らう。
3. むしろ T02 に有利な対比が立つ: **大域圧縮量は一般ベンチで params 超えの信号を持ちうる（彼ら）のに、長文脈文献が提案した機構 probe は長文脈能力でサイズを超えない（T02）**——「どの内部量が・どの能力に」という解像度の必要性の証拠として related work の中心に置く。

## §3. intro/R3 文言の較正（設計書 §4 への反映・確定）

1. 不在主張は長文脈限定形のみ: "for **long-context capability**, the probe families proposed in the long-context literature had not been tested head-to-head against a size baseline"。一般形（"internals beyond size was never tested"）は**禁止**（Learning is Forgetting が反例）。
2. 席 2 の発見を反映——**3 原典とも定量予測を自らは主張していない**（R = 推論時配分の手法・A = 存在主張 + 介入因果・W = 圧縮可能性ベンチ）。「予測に使う実践」の証拠は A 系列のみ（QRHead・DuoAttention・2601.11020）。→ intro の前提文は「provide/predict と主張してきた」でなく「**長文脈能力に機構的に紐づけられ、診断・改善への転用が始まっている（A 系列で実例）。その予測的含意はサイズ統制つきで検定されたことがない**」の形。原典への過大帰属は R1 逆風（strawman 批判）になるので厳守。
3. 席 1 の closes_core ゼロ（22 式）+ 監査役 の could_not_kill（RULER-specific core・R/A/W metric-specific・W probe の 3 軸で先行なし・計 75 式）を不在確認の証跡として §2 Related work に脚注化。

## §4. 公共財セクションの新規性 = 両方生存（監査役 could_not_kill で裏づけ）

- **answer_prefix 回帰の公開報告 = 不在**（監査役: 実装分離 commit 48cbc8b は確認・回帰としての公開 issue/論文/ブログは未発見。近縁 issue #100/#80 は別問題）。**先取権は生きている——issue 投稿（PI GitHub 待ち）と arXiv の時間敏感性は従来評価どおり**。
- **Gemma-3 RULER effective length ladder の先行公開 = 不在**（監査役: 公式 model card の 128K は claimed window であり effective 測定でない、を web_verified で確認）。

## §5. 起草時の必須一次照合（load-bearing・未照合のまま本文に書かない）

1. Learning is Forgetting **PDF 本文**の partial Spearman の結果（向き・係数・どのベンチか・"47 open weights models from 6 different families" の逐語）——related work の対比段落の根拠。PDF は openreview がダウンロード形式のため本セッション未取得。
2. 席 1 の Kramár 2601.11516 の該当文言（WebFetch 要約経由・逐語未確認と席が自己申告）。
3. 席 2 の 2601.11020（QRHead 系）の verbatim・head count 数値の出典（席が研究者再照合要と自己申告）。
4. 席 3 overlaps 群のうち related work に採る各論文（2604.02650・2602.13699・2210.07352・2604.01025）の abstract 逐語は席の web_verified を信頼しつつ、**本文引用する文だけ**起草時に abs 再照合。

## §6. related work の採用リスト（節構成 §2 用・overlaps/adjacent の統合）

- 対比の中心: Learning is Forgetting（ICLR 2026・大域圧縮量 vs 機構 probe・一般ベンチ vs 長文脈）。
- probe→性能予測の隣接: 2604.01025（訓練中 checkpoint の内部表現→下流性能・単一モデル系）・2210.07352（probing→fine-tuning 性能）・2602.13699（attention entropy→回答正解性・per-instance であってモデル横断でない）。
- A 系列の実践: 2404.15574（原典）・2604.02650（retrieval head = 訓練モニタ・単一軌跡）・2601.11020（最適化転用）・QRHead/DuoAttention（席 2 パック）。
- probe 妥当性批判: 2602.11162（retrieval head 検出の動的性）・2607.01033（Model Organism Lottery）・2603.18353（interpretability without actionability）・2605.11448（Deep Minds and Shallow Probes）。
- 逆方向: 2604.24827（内部量→パラメータ数推定 = probe がサイズ代理であることの独立傍証・T02 の診断と響き合う）。
- ベンチ系: RULER 2404.06654・one-ruler 2503.01996・2605.10544（effective-context の行き先）。

## status: done（設計書 v1.1 へ反映済み・残 = §5 の起草時照合 4 件）
