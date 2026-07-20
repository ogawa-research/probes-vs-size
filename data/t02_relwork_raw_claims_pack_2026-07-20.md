# T02 related work 一次照合パック — R/A/W 三因子原典の主張マップ + 実践の広がり

[Release note: internal literature-verification pack (Japanese), released as part of the frozen
record behind the paper. Operational metadata (compute-resource identifiers,
internal audit-transcript paths, model/configuration names) has been
omitted, and AI-assistant model names replaced with role labels: エンジニア (engineer --- implementation, execution, literature collection, and record-keeping) and 研究者 (researcher --- adjudication, protocol design, and adversarial review), both of which were AI assistants working under the direction of the human principal investigator (PI); all content, numbers, and verdicts are unmodified from
the frozen original (sha256 0533f4b129e55a5c494956f45447d9027c049a54e62099dd58d50bdbf198e444, available on request). Internal documents
referenced but not included in this release are likewise available on
request.]

作成 2026-07-20（エンジニア・文脈非継承想定の調査 subagent・一次ソース直接取得）。
目的 = T02 intro の「機構 probe を長文脈能力の予測に使う実践が広がっている」という問題設定の証拠材料。全て未検証入力として研究者が再検証すること。

## §0 スコープ確認（既存 T02 資産との対応）

T02 の R/A/W 因子定義は `data/t02_partB_design_2026-07-11.md` §2 で既に一次照合済みと記録されている：
- R = attention entropy（2606.09508 §2 の式 `H(a_{h,i:}) = -Σ a log a`）
- A = retrieval score（2404.15574 Eq.1 `s_h = |g_h∩k|/|k|`）
- W (副) = KV effective rank / NER（2602.05929 §2.4）

本パックはこれらの**定義の再確認**ではなく（design doc の照合を信頼）、**原典が「長文脈能力の予測」についてどこまで主張したか**を新たに精査した。3本とも直接 PDF 取得・全文確認済み（断定）。

---

## §1 三原典の主張マップ

| 指標 | 原典 | 主張逐語 | 根拠実験規模 | サイズ統制 |
|---|---|---|---|---|
| R (attention entropy) | 2606.09508「From Rigid to Dynamic: Entropy-Guided Adaptive Inference for Long-Context LLMs」 | 断定（PDF全文確認）：この論文は entropy を**推論時のコンピュート配分**（どのheadにどれだけ予算を割くか）に使う `EntropyInfer` を提案する手法論文。長文脈**能力の予測**という主張は本文に**存在しない**。§2 逐語："The row-wise entropy H(a_i:) = -Σ_j a_ij log a_ij ... measures how concentrated the i-th query's attention is over the keys and is the central quantity our method exploits."（能力予測でなく圧縮/配分の中心量、という位置づけ） | Llama-3.1-8B-Instruct・Qwen-2.5-7B-Instruct の2モデルで entropy パターンのcase study（Fig.1-2）。効果測定はLongBench/InfiniteBenchでの**推論高速化率**（up to 2.39×）であり、能力の予測実験ではない。 | 不明（2モデルとも7-8B級・スケール比較は本文になし） |
| A (retrieval score) | 2404.15574「Retrieval Head Mechanistically Explains Long-Context Factuality」(Wu et al., NeurIPS-style, arXiv 2024-04-24 v1) | 断定（PDF全文確認）：abstract 逐語 "a special type of attention heads are largely responsible for retrieving relevant information from long context ... (1) universal: all the explored models with long-context capability have a set of retrieval heads"。これは**「長文脈能力を持つモデルには retrieval head が存在する」という存在主張**であり、T02が検定する「retrieval score から実効長を回帰予測できるか」という定量予測とは異なる主張形式。因果性の主張は masking 実験（Fig.7-8）："masking out retrieval heads severely damages the model's Needle-in-a-Haystack performance"——これは head を壊すと能力が落ちるという**介入的因果**の主張であり、T02が使う「retrieval head の分布量からモデル間の能力を予測する」という**相関的予測**の主張ではない。 | 4 model families (LLaMA, Yi, Qwen, Mistral)・6 model scales・3 finetuning types（Table 1: 7B/13B/6B/34B/14B, MoE込み）。NIAH 約600 instances/model。 | **部分的にあり**：Fig.3で retrieval head比率（~5%）が32×32headから60×56headまでのモデル規模を跨いで一定と観測（記述的な一致観察であり、回帰・統計的統制ではない）。Fig.6でbase/変種間の相関(r>0.8)・別family間の相関(r<0.1)を報告——これも記述的相関であり、サイズを共変量として統制した回帰は本文に**不在**。 |
| W (KV effective rank / NER) | 2602.05929「KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs」 | 断定（PDF全文確認）：この論文は**KVキャッシュの圧縮可能性のベンチマーク**論文であり、長文脈**能力**の予測は主張していない。関連する記述（§4.2.1）："KV Capacity governs compressibility. We observe that the original KV dimension strongly affects model compressibility ... Earlier models such as LLaMA-2-7B exhibit substantially lower NER ... likely due to larger key/value dimensions and weaker utilization of rank capacity during training."——これは「KV次元が大きいほど圧縮しやすい（低NER）」という**圧縮可能性についての観察**であり、「NERが高文脈能力を予測する」という主張ではない。相関の主張は NER と PPL 劣化（圧縮下）："NER correlates strongly with performance degradation under compression"（abstract）——これも**圧縮ロバスト性**の相関であり、長文脈の実効長予測とは別軸。 | 5 open-source LLM series（Qwen3 4B/8B・Mistral-7B・Gemma-1.1 2B/7B・Phi-3-mini-128k）× 5 datasets（英語）+ 15言語（多言語QA）。 | 不明（各シリーズ内の複数サイズ観察はあるが、サイズを明示的に統制した回帰・相関検定は本文中に確認できず） |

**T02側の自己認識との整合**：`t02_capacity_design_2026-07-08.md` 行95 に「KV圧縮系（KV-CoRE・MosaicKV・CompressKV等）：圧縮のための測定であり、実効長予測への回帰・容量スケーリング検定はない。erank は測定器として借りるだけ」と既に明記されている——本パックの独立確認は**この既存の自己認識と一致**（断定）。

---

## §2 実践の広がりの証拠（引用網）

### A（retrieval head）系列 — 診断→性能最適化への転用が明確に存在

1. **arXiv:2601.11020「From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models」**
   - 断定（WebFetch経由、要再確認: 直接PDF未取得）：abstract 逐語（fetch結果）"Advances in mechanistic interpretability have identified special attention heads, known as retrieval heads, that are responsible for retrieving information from the context." / "These results validate the functional role of retrieval heads and show that mechanistic insights can be transformed into performance enhancements."
   - "gains correlate with the sparsity of the retrieval score distribution: models with sparser distributions...respond more strongly"——**retrieval score の分布形状がモデル間の性能利得を予測する**という、T02の問題設定に最も近い先行主張。ただし「性能」が長文脈NIAH系のgainであり、RULERのような実効長閾値予測とは尺度が異なる。
   - **未確認**：モデル数・サイズ統制の有無。この論文はT02自身の design doc（`t02_capacity_design_2026-07-08.md` 行95文脈）で既に既知として扱われている形跡があるが、本パックでは verbatim 未検証（WebFetch要約のみ）——**load-bearing に使うなら研究者が直接PDF照合すべき**。

2. **arXiv:2506.09944「Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking」（QRHead）**
   - 断定（WebFetch確認）：retrieval headを**性能改善の運用部品**として使う（QRRetriever）——診断でなく能動的コンポーネント化。"We further introduce QRRetriever ... We use QRRetriever for long-context reasoning by selecting the most relevant parts with the highest retrieval scores." 10%超の性能向上を報告。
   - 実践の広がりの型：診断（誰が retrieval head か）→ 運用（それを retrieval に使う）という転用の一例。T02の「予測」文脈とは別方向（予測でなく道具化）。

3. **arXiv:2410.10819「DuoAttention」**
   - 断定（WebFetch確認）：abstract に Wu et al. への引用言及なし——独立に "Retrieval Heads" / "Streaming Heads" の分類を再発見・自己完結的に用いている（"we identify that only a fraction of attention heads, a.k.a, Retrieval Heads, are critical..."）。KVキャッシュ圧縮への応用（メモリ削減、最大2.55x）。
   - これも運用的転用（圧縮のためのhead分類）であり、能力予測の実践ではない。

4. **arXiv:2606.21249「Does RoPE Prevent or Degrade Retrieval Heads? A Mechanistic Analysis Across Model Families」**（未確認・検索結果タイトルのみ）— retrieval head をモデルfamily横断で分析する系譜として存在確認のみ、内容未照合。

**§2-A の要約**：retrieval head 診断を「性能向上の運用部品」に転用する実践（QRHead・DuoAttention）は複数確認できたが、「retrieval score の量からモデル間の長文脈**能力を予測**する」という、T02の回帰設計に直接対応する実践は arXiv:2601.11020 のみ候補（要 verbatim 再確認）。

### R（attention entropy）系列 — 能力予測でなく効率化・分析の実践が主流

検索で見つかった関連論文（2412.16545「Attention Entropy is a Key Factor」・2506.16640「Long-Context Generalization with α-entmax」・2602.22175「DySCO」）はいずれも**推論効率化・アーキ改良**の文脈で entropy を扱っており、「entropy からモデルの長文脈能力を予測する」という実践は検索範囲内で**確認できず**（不明、確信度低）。

### W（effective rank）系列 — 長文脈能力予測の実践は検索範囲内で未発見

「effective rank / KV cache / hidden representation」が長文脈能力を予測するという直接主張の論文は検索で**発見できなかった**（不明）。関連して見つかった "Diff-eRank"（OpenReview, model evaluation指標としてのrank-based metric）は評価指標としての一般提案だが、長文脈に特化した主張かは未確認。

---

## §3 妥当性検証・批判の先行

1. **retrieval head 検出の再現性への疑義 — arXiv:2602.11162「Retrieval Heads are Dynamic」**
   - 断定（WebFetch確認、abstract逐語取得）："prior works largely rely on static statistics aggregated across datasets, identifying heads that perform retrieval on average. This perspective overlooks the fine-grained temporal dynamics of autoregressive generation." 3つの主張：(1) Dynamism（timestepごとに変動）(2) Irreplaceability（動的headは静的headで代替不能）(3) Correlation（隠れ状態が将来のretrieval headパターンを予測する内的計画信号を持つ）。
   - これは Wu et al. 2404.15574 の「retrieval head は固定的な集合」という前提そのものへの反証に近い——**A因子（retrieval score）を単一の静的スカラーとして測定するT02の設計への間接的な妥当性懸念**として引用可能。ただし本論文はNIAH+multi-hop QAでの分析であり、T02のRULER実効長予測への直接反証ではない。
   - **数値的な再現性の弱さ**：WebSearchの二次情報（未検証）で「head count is not a fixed quantity: it depends on the detection context length and the threshold, so it varies modestly across runs (for example OLMo 81–95, Owen 58–64, Mistral 96–98 across different context sets)」という記述を見つけたが、**出典論文名が特定できておらず不明**（検索結果の要約からの引用で、どの一次論文の数値か研究者が再確認する必要あり）。

2. **effective rank の予測妥当性への疑義 — arXiv:2207.02099「An empirical study of implicit regularization in deep offline RL」**
   - 断定（WebSearch要約経由、要研究者確認）：**ドメインが異なる**（LLMでなくoffline RL・Atari/bsuite/DeepMind Lab）。"A direct association exists only in restricted settings and disappears in the more extensive hyperparameter sweeps... the rank alone might be a misleading indicator of performance."
   - T02への適用は**類推**（analogy）としてのみ有効——「effective rankを単独の性能予測指標として使うことへの一般的な注意」という文脈で引用できるが、長文脈LLM能力への直接的な反証ではない。開示：ドメイン越境の弱い証拠。

3. **KV-CoRE自体の限界の自己申告**：2602.05929 は§2.4でNERが `[1/r, 1]` の範囲を持つことを明記するのみで、予測妥当性への批判的検討は本文になし（未確認 = 著者自身の批判的検討は不在という消極的事実）。

---

## §4 未確認事項（研究者が裏取り必要な項目）

1. **arXiv:2601.11020 の verbatim 内容**（サイズ統制の有無・retrieval scoreの分布指標の定義）は WebFetch要約のみで、直接PDF未取得。T02のintroでload-bearingに使うなら研究者が一次照合すること。
2. **「head count varies OLMo 81-95, Qwen 58-64, Mistral 96-98」という数値**の出典論文が特定できていない（WebSearch二次要約からの孤立した引用・原典未特定）。
3. R（attention entropy）系列で「entropy が長文脈能力を予測する」と主張する論文の**不在確認**は、検索クエリ数本（3クエリ×上位10件程度）の範囲に限られており、網羅的な文献レビューではない——低確度。
4. W（effective rank）系列も同様に不在確認は限定的探索の結果であり、見落としの可能性あり。
5. 2606.09508（entropy paper）の著者・所属・実際の出版状態（PDFヘッダで PolyU/HKUST と確認・2026-06-08提出）は確認済みだが、査読状態（peer-reviewedか否か）は未確認。
6. 2602.05929（KV-CoRE）も同様、2026-02-07投稿・v2、査読状態未確認。

---

## 確信度サマリ

- **断定**：3原典の定義一致・3原典いずれも「長文脈能力の定量予測」を主張していないこと・DuoAttention/QRHeadの運用的転用の存在・Retrieval Heads are Dynamicのabstract内容。
- **おそらく**：T02の問題設定（「probe→能力予測」の実践が広がっている）は、A因子（retrieval head）の系譜では部分的に支持される（QRHead・2601.11020候補）が、R・W因子の系譜では検索範囲内で直接支持する論文が見つからず、T02自身の新規性（3因子合成による予測という設計）を裏付ける形になっている。
- **未確認/不明**：2601.11020のサイズ統制詳細・head count数値の出典・査読状態・網羅性（限定的探索）。
