# T02 Part B 詳細設計 — 三因子合成 → 実効文脈長の頭対頭回帰

[Release note: excerpt of the frozen internal protocol document (Japanese).
Operational metadata --- approval flow, cost estimates, compute-resource and
phase-status columns, drafting personnel --- has been omitted, and AI-assistant
model names replaced with role labels: 研究者 (researcher --- adjudication,
protocol design, and adversarial review) and エンジニア (engineer ---
implementation, execution, and record-keeping), both of which were AI
assistants working under the direction of the human principal investigator
(PI); all protocol content, numbers, and decision rules are
unmodified from the frozen original (sha256
8de571c2a2253de3143daadbb0ce497d9d494f9478c2f1bea4bec2460a6363e3, available on
request). English summaries of the decision rules appear in the paper and the
release README.]

**Revisions affecting the frozen protocol**: v1.1 (2026-07-11) — R/A の長さ格子を {1K, 2K, 4K} へ修正（大型モデルの per-layer attention メモリ安全のため・§2 R 測定の注記どおり）。v1.2 (2026-07-12・B3 測定前) — Gemma-3 の RULER は因子測定と同一の `-pt` チェックポイントで実施し base 閾値 79.4 を適用（§1 の「85.6 で統一」はこの裁定で上書き・混在は B4 で開示 + 85.6 感度併記）・Gemma-3 階段 5 点を「コスト実測後判断」から「実施を基本線」へ格上げ。

## 0. Part B の主張と Part A からの継承

**主張（親設計 §0 のまま）**: 短文脈（≤8K）で測った三機構因子（解像度 R・配分 A・帯域 W）の合成は、単一因子および無機構ベースラインより、長文脈（≥64K 帯）の実効文脈長をよく予測する。

**Part A からの継承と修正**:
- W は新規測定しない——Part A の **k\*₆₄(J-lens)・k\*₆₄(logit-lens)・value erank（band 平均・中立コーパス）の 3 変種**を予測子候補にする。Part A の結論（W 測定器は系列依存）を受け、「どの W 測定器が行動（実効長）を最も予測するか」自体を報告対象に格上げする（測定器の外的妥当性という Part A の open への回帰側からの答え）。
- 横断層モデルにも band が要る（W・R の band 限定版のため）→ CKA 機械規則を適用。**run 被覆率 >90% の監視条項**（Part A の教訓）を最初から組み込む。

## 1. 目的変数

- **主 = RULER effective context length・ただし「スコア行列への閾値法の機械再適用」を正とする（新規約・重要）**。定義の逐語 = 「The performance exceeding the Llama2-7B performance at 4K (**85.6**) is underlined. The effective context length is the maximum length passing this threshold.」（論文 v3 Table 3 キャプション・base モデル用の別閾値 **79.4** が Appendix F——本設計は chat/instruct 系のみ使うので 85.6 で統一）。**表の Effective 列は使わない**——author-reported 6 行（Qwen3-14B/32B/8B/4B・Qwen2.5-7B-1M・EXAONE-1.2B）で表の Effective 列が閾値法の機械適用と不一致であることを機械検査で確認済み（例: Qwen3-8B は 64K=82.1 < 85.6 なのに「64K」表記——RULER パック §5 に全件）。**再適用規約 = 「4K から連続して 85.6 を超える最大評価長」**（4K で既に ≤85.6 なら "<4K"）。
- **表の実態（B0 完了・RULER パック正本）**: 正本 = GitHub NVIDIA/RULER README（commit `ab17b785`・2025-10-09 が最終更新・main HEAD と diff ゼロ確認）。44 行中 open weights 42。評価格子は 4K–128K の 6 点。**Qwen3 はオリジナル版 4B/8B/14B/32B の 4 行のみ（0.6B/1.7B なし・2507 版なし——版問題は「混在」でなく「未掲載」でクリーン）。Gemma-3 は 0 行**。author-reported（`*`）行はスコア行列自体が転載値であり RULER 標準 13 タスク構成との条件同一性は未確認（開示事項）。
- **自前測定**: (a) qwen3-0.6b / 1.7b（必須・公開値なし・4K-32K 帯・RULER 公式ハーネス）(b) **Gemma-3 階段 5 点（RULER 0 行のため・4K-32K 帯を主とし 64K は資源次第）**——(b) はコスト実測（B3 前半で 0.6b/1.7b を回した後）を見て規模を確定。Gemma-3 が間に合わない場合、主回帰は「Qwen3 4 点（再計算値）+ 自前 2 点 + 横断 8 点 = 14 点」で成立し、Gemma-3 は追補とする。
- **censored 処理（規約・事前固定）**: 「>128K」（= 128K でも閾値超え）は (a) 主解析 = 128K で代入 (b) 感度分析 = 該当点除外、の両方を報告。"<4K" は 4K の半分（2K）で代入 + 除外感度。N が小さく Tobit 型は不採用。
- **副 = Oolong-synth**（階段層のみ・16K-131K）。RULER 回帰の結論が立った後のフェーズ B3 後半に置く（優先度は RULER より下・単一スカラー批判への応答として維持）。

## 2. 因子測定の仕様

### R（解像度）— attention エントロピーの長さ成長率

- **定義（一次照合済み・design pack）**: 行単位エントロピー H(a_i:) = −Σ_{j≤i} a_ij log a_ij（a = softmax(QK^T/√d_k)・causal）。2606.09508 §2 の逐語式。
- **測定**: 中立コーパス（wikitext 連結）で長さ n ∈ **{1K, 2K, 4K}** の 3 点・各長さ 4 シーケンス（v1.1 修正: 当初の {1K,4K,8K} から変更——per-layer attention 行列が 8K で bf16 8.5GB/層〔27b・64head〕となり大型モデルで危険。4K なら 2.1GB/層で全 18 モデル安全。2 oct の勾配推定には十分で、2410.01104 の Gemma 測定も ~4K 帯・「因子 ≤8K」制約内）。**実装 = eager attention + per-layer forward hook で attn_weights を層ごとに取得 → query 64 位置（対数間隔）の行の H・δ を即時計算 → 破棄**（全層同時実体化しない・ピークは 1 層分）。RoPE 適用後の実効 logit で測るため hook 方式を主とする（Q/K 自前再計算方式はモデル依存の RoPE 再現リスクがあるため不採用）。
- **集約**: head → 層平均 → (a) 全層平均（主）(b) band 層平均（感度）。**R := 1 − Δ̂**、Δ̂ = H̄(n) の log n に対する OLS 勾配（3 点）を log(n) 正規化（一様分散なら H ~ log n で Δ̂→1・R→0／集中維持で R→1）。
- **副指標 δ（logit spread）**: 同じ forward で attention logit の行内 max−min（定義 = δ = max_i e_i − min_j e_j・2410.01104 Lemma 2.1・逐語照合済み〔`t02_p0d_anchor_pack_2026-07-11.md`・design pack と二重確認一致〕）。head 別分布の平均±SD を保存。
- **P0-d 照合（δ 側）**: 公開実測は**初代 Gemma 2B/7B**（δ∈[2.28,14.78] 平均 5.69±2.05 ／ δ∈[0.09,32.74] 平均 5.82±2.61・gemma/modules.py 単一コードファイル ~4,000 tokens・全 head プール・図表番号なし〔本文 §2 箇条書き〕）。**T02 階段の Gemma-3 とはモデル世代が異なるため、直接照合には初代 Gemma 2B（bf16 ~5GB・軽量）を B1 パイロットに追加**し、同条件（コード入力 ~4K・全 head プール）で δ 分布を再現測定する——「平均が公開値 ±1SD 帯に入る」を照合成立の目安（判定条件でなく sanity 水準・事前固定）。

### A（配分）— retrieval head 分布

- **定義（逐語照合済み）**: retrieval score s_h = |g_h ∩ k| / |k|（2404.15574 Equation 1・copy-paste 判定 2 条件込み・閾値 0.1 = retrieval head の判定閾値〔§2 末尾・逐語 = アンカーパック〕）。**S3 の簡略代理（needle への attention 質量）ではなく正式定義を使う**——P0-d で公開値と照合するため + S3 の教訓（TF probe の場面ずれ = 測る場面は回答生成時）。
- **測定**: NIAH（文脈 **4K** を主〔v1.1: R と同じメモリ理由〕・needle = UUID 型 1 種 + 文章型 1 種・深度 {0.1,0.3,0.5,0.7,0.9} × 各 10 サンプル）。回答生成時（teacher-forced で needle 再生成中）の attention argmax が needle トークンを指しコピーが成立する頻度 = s_h（式 1 の判定 2 条件どおり）。P0-d 照合対象（Llama-3.1-8B）のみ lengths {1K,2K,4K,8K} × 10 depths に拡張（原設定に寄せる・8K は 1 層ずつ処理で 80GB 内）。
- **集約**: (i) retrieval head 総質量 = s_h 上位 head（全 head 数の 5%）の s_h 和 (ii) head 分布の Gini 係数（自前定義と明示——2601.11020 に数式なし・親設計どおり）。符号は事前固定しない（集中/分散のどちらが効くかは回帰に決めさせる・親設計どおり）。
- **P0-d 照合（A 側）**: 原論文の公開統計は**レンジ形式のみ**（score>0.1 の head = 全体の 3–6%・zero score = 45–73%・retrieval head 率 ~5%——モデル別の個別数値表は本文に不在〔Figure 3 は画像ヒストグラムのみ〕・アンカーパックで不在確認済み）。照合は **B1 で Llama-3.1-8B を測り「3 統計がレンジ帯に入るか」の帯照合**とする（点照合は原理的に不可能と開示）。照合用の測定は原設定に寄せる（lengths を 1K–8K で複数点 × 10 depths・greedy——原論文は 1K–50K × 10 depths ≈ 600 instances）。

### W（帯域）— Part A から流用 + 横断層のみ新規

- 階段層 9 点は Part A の値をそのまま使用。**横断層は k\* 測定（10 seed・軽量 2 分/モデル）+ erank（中立コーパス）を新規測定**（lens が存在するモデルのみ J-lens k\*——**既製 lens の有無が横断層の選定制約になる**〔neuronpedia/jacobian-lens は 34 モデル分・カバレッジは選定時に確認〕。lens がないモデルは erank + logit-lens k\* のみ = 欠測を明示して回帰は変種別に）。

## 3. 横断層の選定（確定・RULER パックの 42 行に基準を機械適用）

**基準（優先順）**: (a) open weights (b) RULER 公開値あり（NVIDIA 測定〔`*` なし〕を優先・author-reported は開示つき）(c) **dense かつ bf16 ≤ 80GB**（MoE は親設計 §5 で scope out——Jamba/Mixtral/DBRX/Qwen3-MoE は除外）(d) family 多様性 (e) claimed/effective の乖離レンジを広く張る。

**確定リスト（8 点・family +5）**:

| モデル | params | claimed | effective（再計算） | 選定理由 |
|---|---|---|---|---|
| Llama3.1-8B-Instruct | 8B | 128K | 32K | family +1・**P0-d 照合対象**・NVIDIA 測定 |
| Mistral-7B-Instruct-v0.2 | 7B | 32K | 16K | family +1 |
| Mistral-Nemo-Instruct | 12B | 128K | 16K | Mistral 系 2 点目（系列内サイズ差） |
| GLM4-9B-chat-1M | 9B | 1M | 64K | family +1・claimed 1M の極端点 |
| GLM3-6B-128K | 6B | 128K | 4K | GLM 系 2 点目・**低 effective の極端点** |
| Yi-34B-200K | 34B | 200K | 32K | family +1・サイズ上端（bf16 68GB・80GB 可） |
| Phi3-mini-128K | 3.8B | 128K | 32K | family +1・小型点 |
| Phi3-medium-128K | 14B | 128K | 32K | Phi 系 2 点目 |

補欠（8 点で回帰が不安定な場合の追加候補・優先順）: InternLM2.5-7B-1M（claimed 1M・effective 4K の乖離最大点）・Command-R-0824（32B）・MegaBeam-Mistral-7B-512K。fine-tune 派生系（ProLong/GradientAI/LWM/LongChat/LongAlpaca 等）は「base モデルの機構因子」の解釈が濁るため優先度最下位。**W 因子**: 横断層は既製 lens の有無を B1 冒頭に HF で確認し、lens があるモデルのみ k\*(J-lens) を測定（なければ erank + logit-lens k\* のみ・欠測明示）。**band**: 各モデル CKA 機械規則 + run 被覆率 >90% 監視条項（Part A の教訓）。

## 4. 回帰プロトコル（事前固定）

- **N** ≈ 15-19（階段 9〔14b は W=erank のみで参加・k\* は欠測扱い〕+ 横断 6-10）。
- **予測子セット**: {R}・{A_mass}・{A_gini}・{W}・{R+A+W 加法}・{R×A×W 乗法 = log 空間加法}・ベースライン {claimed length}・{log params}・{d_model}・{訓練長}。**同時投入は 3 個まで**（親設計どおり）。W は 3 変種（k\*J / k\*logit / erank）× 上記を回す。
- **評価**: log(effective length) の **leave-one-family-out CV**（family = Qwen3/Gemma-3/Llama/Mistral/…）・指標 = held-out MAE。
- **有意性**: permutation test（目的変数をシャッフルして CV MAE の帰無分布・1000 回・p < 0.05）。
- **判定（親設計 §6 の具体化）**:
  - 「合成 > 単一」= 乗法（または加法）合成の CV MAE が最良単一因子より小さく、permutation p < 0.05。不成立 → **kill 基準発動**（Part A は独立に生存・親設計どおり）。
  - 「機構 > 無機構」= 三因子系の最良が全ベースラインに勝つ。不成立 → 「機構測定の付加価値なし」と報告。
  - 「乗法 vs 加法」= MAE 差を報告（関連理論の傍証としての解釈は判定でなく考察）。
- **開示**: N が小さいことによる検出力限界・family 数・censored の扱い・W 変種依存性を Methods に明記。

## 5. フェーズと概算

| フェーズ | 内容 |
|---|---|
| B0 | RULER 公開値整備 + P0-d アンカー収集 + 横断層確定（完了 2026-07-11） |
| B1 | R/A 測定の実装 + パイロット（qwen3-1.7b + llama-3.1-8b + **初代 gemma-2b〔P0-d 直接照合用・世代差の解消〕**）+ **P0-d 照合**（δ@初代 Gemma 2B = 帯 5.69±2.05・retrieval head@Llama-3.1-8B = 3 統計のレンジ帯照合〔>0.1 率 3-6%・zero 率 45-73%・head 率 ~5%〕） |
| B2 | 全モデル R/A 測定（階段 9 + 横断 8・≤8K）+ 横断層の W（k\*/erank/CKA band・lens 有無は冒頭確認） |
| B3 | RULER 自前測定: 前半 = qwen3-0.6b/1.7b（必須）→ コスト実測 → **Gemma-3 階段 5 点（4K-32K 主・規模はコスト実測後に確定）** →（後半・任意）Oolong 階段 |
| B4 | 回帰・判定・報告書 |

## 6. 差別化・循環回避の再確認（親設計から不変）

長さ帯の分離（因子 ≤8K・目的変数 ≥64K 帯を含む effective length）・目的変数二本立て・無機構ベースライン。母体 workspace 論文は行動回帰なし（Part B が固有）・HELMET は単一スカラー批判（二本立てで内包）・KV 圧縮系は回帰なし。
