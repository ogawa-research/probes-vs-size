# T02 related work 照合：head-to-head 先行研究の不在確認（席1）

[Release note: internal literature-verification pack (Japanese), released as part of the frozen
record behind the paper. Operational metadata (compute-resource identifiers,
internal audit-transcript paths, model/configuration names) has been
omitted, and AI-assistant model names replaced with role labels: エンジニア (engineer --- implementation, execution, literature collection, and record-keeping), which was an AI assistant working under the direction of the human principal investigator (PI); all content, numbers, and verdicts are unmodified from
the frozen original (sha256 b4da817eb59fcbb66cdbb0c6a459e763a603f7d995f0eacd9c3ae18b09be1fb1, available on request). Internal documents
referenced but not included in this release are likewise available on
request.]

_2026-07-20・席 1・エンジニア（文献調査）_

## §1 結論

**確信度＝高い（未確認は残るが closes_core 該当なし）**：「短文脈の内部機構量（mechanistic/interpretability probe 由来の量）を、モデルサイズという単純ベースラインと頭対頭で対決させ、長文脈行動（RULER effective length 等）への予測力を系統比較した」という T02 の中心設計に **closes_core 判定のヒットはゼロ**。22の検索式・9本の個別論文精読（WebFetch で abstract 逐語取得）を行ったが、(a) 対象量（内部機構メトリクス vs パラメータ数）(b) 目的変数（長文脈行動・RULER 系）(c) 頭対頭の統計比較（回帰・LOFO-CV 等） の3条件を同時に満たす論文は見つからなかった。

ただし、**個々の要素（内部量がサイズと相関する／確率的にサイズが支配的／probe が汎化に失敗する）を単独で示す隣接論文は複数存在**し、T02 の related work・防御材料として引用候補になる（§2 overlaps/adjacent 参照）。「不在確認」は web 検索・arXiv API という限られた経路での確認であり、査読前 preprint の網羅性には限界がある（§4）。

## §2 分類別ヒット一覧

### closes_core（該当なし）
なし。

### overlaps（部分重複・引用要）

**1. arXiv:2604.02650 — "Revealing the Learning Dynamics of Long-Context Continual Pre-training"**
- 逐語（要約回答からの再構成、abstract 全文は取得できず一部のみ）: "Existing studies on Long-Context Continual Pre-training (LCCP) mainly focus on small-scale models and limited data regimes... This work provides a comprehensive monitoring framework, evaluation system, and mechanistic interpretation for the LCCP of industrial-grade LLM."
- 関係：retrieval heads の attention score 変化と perplexity を、単一モデル（Hunyuan-A13B・80B）の**訓練軌跡上**（200Bトークン）でモニタリング指標として比較。T02 との違い＝(i) 単一モデルの時間発展であり、**モデルサイズ間の頭対頭比較ではない** (ii) 目的変数は SFT 性能・収束状態であり RULER ではない (iii) 明示的な統計的比較（回帰等）の記載なし。
- 判定：mechanistic interpretation for long-context という同じ問題領域に位置するため related work で引用すべきだが、T02 の中心主張（cross-family, size-controlled）を先取りしない。

**2. arXiv:2509.13333 — "Evaluation Awareness Scales Predictably in Open-Weights Large Language Models"**
- 逐語: "We investigate evaluation awareness across 15 models scaling from 0.27B to 70B parameters from four families using linear probing on steering vector activations. Our results reveal a clear power-law scaling: evaluation awareness increases predictably with model size."
- 関係：**probe 精度が model size に対して予測可能にスケールする**という構造は T02 の "internal metric ≈ サイズの代理" という結論と方法論的に同型。ただし目的変数は evaluation awareness（deception）であり長文脈と無関係、head-to-head の「probe が size を超える付加価値を持つか」という明示的対決構造は無い。
- 判定：related work で「probe-scales-with-size は他ドメインでも観測されている一般的パターン」として引用する価値あり。closes_core ではない。

**3. arXiv:2604.13386 — "Linear Probe Accuracy Scales with Model Size and Benefits from Multi-Layer Ensembling"**
- 逐語: "Across 12 models (0.5B--176B parameters), we find probe accuracy improves with scale: ~5% AUROC per 10x parameters (R=0.81)."
- 関係：deception probe（Insider Trading・Harm-Pressure Knowledge）の精度が model size とスケール則で相関するという発見。長文脈の言及ゼロ。頭対頭の「probe vs baseline」構造ではなく「probe accuracy that scales with size」という別の主張形。
- 判定：2509.13333 と同型の隣接論文（同一パターンの別ドメイン再現）。adjacent 寄りの overlaps。

### adjacent（引用候補・上記以外）

**4. arXiv:2607.01033 — "The Model Organism Lottery: Model Organism Interpretability Strongly Depends on Training Methodology"**
- 逐語: "Our findings show that (i) MO interpretability depends strongly on training objective, target behaviour, model architecture, and training data generation pipeline; (ii) substantial variance remains even after controlling for differences in the strength of target behaviour expression; and (iii) our more realistic integrated training often yields less interpretable MOs than standard post-hoc methods."
- 関係：interpretability の有効性が訓練方法論に強く依存するという知見——T02 の「lens の有効域は文脈・モデル依存」という measurement-science 的教訓と精神的に近い。ただし対象は model organisms（意図的に仕込んだ振る舞い）の可視性であり、長文脈・容量測定とは無関係。
- 判定：measurement science / probe validity の一般論として related work 導入部の引用候補。

**5. arXiv:2601.11516 — "Building Production-Ready Probes For Gemini"（Kramár et al.）**
- 逐語: "activation probes may be a promising misuse mitigation technique, but we identify a key remaining challenge: probes fail to generalize under important production distribution shifts"。長文脈シフトへの言及あり："the shift from short-context to long-context inputs is difficult for existing probe architectures"（要約経由・逐語未確認、下記§4参照）。
- 関係：**probe の頑健性**（短文脈で訓練した probe が長文脈入力に汎化するか）という**別の問い**——T02（短文脈で測った内部量が長文脈"行動"を予測するか）とは方向が異なる。対象ドメインも safety/misuse detection（cyber-offensive）であり容量測定ではない。
- 判定：「probe と長文脈の相互作用」という語のレベルでは近いが、問いの構造が別物——誤って closes_core と早合点しないよう明示的に区別して記録。adjacent（誤爆防止のための記録）。

**6. arXiv:2605.08853 — "Architecture, Not Scale: Circuit Localization in Large Language Models"**
- 逐語: "We challenge this assumption by showing that the attention architecture matters more than parameter count... These findings suggest that some architectural choices make large models more tractable to study and that interpretability difficulty is not a fixed consequence of model size."
- 関係：内部機構的性質（circuit の局在・安定性）が「scale か architecture か」という対決構造を持つ点は T02 と精神的に類似するが、対象は回路の解析可能性（tractability）であり、目的変数が長文脈行動でも RULER でもない。統計手法も定性的比較（qualitative）で LOFO-CV 等の形式的検定はない。
- 判定：「機構的性質の説明因子としてサイズ vs 他要因を競わせる」という研究デザインの先例として related work・方法論比較で言及候補。

**7. arXiv:2607.07670 — "Does Bielik Know What It Doesn't Know? Activation Dispersion Separates Entity Familiarity from Factual Reliability Across Model Scale"**
- 逐語: "This representational signal is already at ceiling at 1.5B, whereas behavioral factual reliability scales sharply... Entity familiarity and factual reliability are distinct phenomena on different scaling curves."
- 関係：**内部表象シグナルとふるまい（behavioral）が異なるスケーリング曲線を持つ**という発見——「内部量と外部行動は別物」という T02 の教訓（測定器の有効域は行動と一致しない）と類似の精神。ただし対象は事実性・幻覚（4モデル・1.5B-11B・Bielik系）であり、長文脈能力ではない。model size との頭対頭の形式的比較（回帰・AIC/BIC 等）は無い。
- 判定：measurement science の教訓として related work で引用候補（「内部量と外部行動の乖離」の別事例）。

**8. arXiv:2604.24827 — "Incompressible Knowledge Probes: Estimating Black-Box LLM Parameter Counts via Factual Capacity"**
- 逐語: "Storing F facts requires at least F/(bits per parameter) weights, so factual recall lower-bounds parameter count--an intrinsic, serving-independent signal, though (as we show) a coarse one."
- 関係：**逆方向の問い**（内部容量シグナル→パラメータ数を推定）。T02 が「内部量がサイズを超える情報を持つか」を問うのに対し、この論文は「内部量からサイズを回収できるか」という相補的な問い。目的変数も予測方向も違うため overlaps ではないが、方法論的な近縁（capacity ≈ 容量の下界としてのパラメータ数）として引用候補。
- 判定：adjacent（方向が逆の相補研究）。

### unrelated（検索で見つかったが無関係と確認）

- arXiv:2406.04391（"Why Has Predicting Downstream Capabilities... Remained Elusive?"）：scaling law + loss-based のみ、mechanistic probe 言及ゼロ。
- arXiv:2601.11020, 2510.02219, 2506.09944, 2410.10819 等（retrieval heads 最適化系）：性能改善技術（RetMask 等）が主題で、予測・回帰の頭対頭比較ではない。
- arXiv:2605.17887, 2603.05498（attention sink / activation outlier 系）：B4-01（別テーマ）に近縁だが T02 の測定量ではない。RULER 予測の頭対頭構造なし。

## §3 検索式全数とヒット数

| # | 検索式 | ヒット数（提示） | 新規発見（要精読） |
|---|---|---|---|
| 1 | mechanistic probe internal metric predict long context capability RULER | 10 | 0 |
| 2 | "retrieval heads" predict long context performance model size baseline comparison | 9 | 0 |
| 3 | probing metrics fail to generalize long context capability interpretability arxiv 2026 | 8 | 2604.13386 |
| 4 | Kramár probes fail long-context relevant information distributed many tokens | 9 | 2601.11516 |
| 5 | effective rank attention entropy activation magnitude predict effective context length RULER | 9 | (2607.07670系フレーズ) |
| 6 | "effective rank is unstable" "no scale trend" "common depth band" attention entropy | 9 | 2607.07670 |
| 7 | does parameter count explain interpretability probe predictive power downstream capability confound | 6 | 2604.24827, 2406.04391 |
| 8 | activation probe outperforms parameter count baseline predicting benchmark score regression cross-family | 9 | (既出のみ) |
| 9 | interpretability metric behavioral prediction leave-one-family-out cross-validation model size baseline | 6 | 0（統計手法論のみ） |
| 10 | arXiv API: abs:"long-context" AND abs:"interpretability" AND abs:"parameter" | 12 | 2604.02650 |
| 11 | "Jacobian lens" OR "detached jacobian" capacity probe long context prediction language model | 9 | 0（J-lens自体の解説記事のみ） |
| 12 | arXiv API: abs:"effective context length" AND abs:"model size" AND abs:probe | 0 | 0 |
| 13 | negative result mechanistic interpretability does not predict downstream behavior scale confound 2026 | 10 | 2605.08853 |
| 14 | what predicts RULER effective length across model families empirical study 2026 | 10 | 0 |
| 15 | measurement science interpretability probes lens validity domain application 2026 | 9 | 0 |
| 16 | superposition capacity measurement predicts benchmark performance regression scaling | 8 | 0 |
| 17 | papers citing "jacobian lens" Anthropic capacity capability prediction downstream | 7 | 0 |
| 18 | interpretability probe generalizes across model families evaluation multiple architectures benchmark 2026 | 10 | 2607.01033, 2509.13333, 2606.29196 |
| 19 | attention sink activation outlier magnitude predicts long context retrieval failure across model families | 10 | 0（B4-01隣接のみ） |
| 20 | does mechanistic feature add predictive value beyond scale AIC BIC model comparison language model | 8+9 | 0 |
| 21 | arxiv 2607 2608 mechanistic probe long context capability model size comparison | 8 | 0 |
| 22 | RULER benchmark predicted by internal representation geometry vs number of parameters ablation study | 9 | 0 |

計22検索式、ヒットゼロは #12（arXiv API 厳密フレーズ、AND条件が厳しすぎ0件）のみ。それ以外は全て何らかの結果を返したが、closes_core に該当するものはなかった。

## §4 未確認事項

- **未精読**：arXiv:2606.29196（"Representational Depth of Evaluation Awareness Shifts With Scale"）は検索スニペットのみで abstract 未取得——2509.13333 と同系統（evaluation awareness × scale）の可能性が高く closes_core とは考えにくいが未確認のまま。
- **逐語未確認**：arXiv:2601.11516 の「long-context input への shift が既存 probe architecture に困難」という文言は WebFetch 要約からの二次引用で、abstract 全文の直接逐語確認はできていない（要約ツールの見落としリスク、`60_subagent_discipline.md §B` 相当の懸念——load-bearing に使うなら研究者が原文照合すべき）。
- **網羅性の限界**：本照合は WebSearch（要約付き Google 相当検索）と arXiv API の2経路のみ。Semantic Scholar の被引用グラフ探索（RULER論文・jacobian-lens論文からの forward citation）は未実施——これを行えばより体系的な closes_core 探索ができた可能性がある。特に 2026年7月前半（直近1週間）の投稿は検索エンジンのインデックス遅延で漏れている懸念が残る。
- **「T02 の正確な central claim」の再確認は本席では未実施**：本タスクは current.md の記述（機構probe vs baseline のsize-controlled比較）をそのまま前提にしており、論文本体（未執筆・T02の実データ・判定書）への直接照合はしていない。load-bearing な最終判断は研究者が T02 の判定書群（`data/t02_partA_final_judgement_*`・`data/t02_b3/b4_judgement_*`）と本パックを突き合わせて行うべき。
