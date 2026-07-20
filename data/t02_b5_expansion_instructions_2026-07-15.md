# T02 B5 — 論文化前の補強測定：Gemma R 再測定 + 横断層増員（実行指示書・事前登録証跡を兼ねる）

[Release note: internal execution-protocol record (Japanese), released as the frozen
source of the numbers reported in the paper. Operational metadata ---
compute-resource identifiers, cost estimates, internal commit hashes
--- has been omitted, and AI-assistant model names replaced with role labels:
研究者 (researcher --- adjudication, protocol design, and adversarial review)
and エンジニア (engineer --- implementation, execution, and record-keeping),
both of which were AI assistants working under the direction of the human
principal investigator (PI); all numbers, verdicts, and disclosures are
unmodified from the frozen original (sha256 73e5c5a0a7cb3f1daa4149ab8ce6722d2cbd58c31894c3d2b8af91a58b00d4fb, available on request).
The internal run logs (referred to as RUN_LOG in the text) are omitted as
operational records; their acceptance-gate sections are excerpted verbatim in
the release ledger.]

- 起草: 研究者 2026-07-15。実行役: エンジニア 新セッション（機械実行と記録のみ・判定は 研究者）
- 位置づけ: B4 判定（`data/t02_b4_judgement_2026-07-15.md`）は**確定済みで本測定によって変更しない**。B5 は論文化に向けた (i) Gemma R 救済値の正規化 (ii) N 拡張（15→~22）と**同サイズ対照**の追加。B5 の解析条件は本書 §4 に事前固定する。
- 計算資源: **H200 1 台**（Command-R 35B bf16 ≈70GB のため 80GB 級では A probe が危険——B2 の yi-34b 前例）。
- 運用規律: B1/B2 指示書（`t02_partB_b1b2_instructions_2026-07-11.md`）の共通規律を継承——sha256 記録・RUN_LOG は事実のみ。出力先 = `data/t02_b5_2026-07-15/`。[remaining operational-discipline items omitted]

## 1. Job 1 — R probe 修正版の診断 + Gemma-3 5 点の R 再測定

probe は修正済み: `experiments/t02/scripts/r_delta_probe.py`（2026-07-15 研究者・非有限 head-row の除外 + `n_valid_per_head`/`nonfinite_head_rows` 計数。合成 NaN 4 ケースの単体検証済み・clean 系は旧実装と数値一致）。

1. **診断（最初に 1 回・gemma-3-4b）**: 長さ 2048 で 1 forward し、層別に `nonfinite_head_rows` を出力。**確認事項 = (a) 非有限が SWA 層のみに出るか（global 層 = 6 の倍数層はゼロか）(b) window(1024) 超の query 位置に限られるか (c) 非有限の実体（nan か ±inf か）**。数行のログで良い——これは「attention 重み自体が非有限」という 研究者 の推定の一次確認。結果が推定と食い違っても続行し、事実だけ RUN_LOG に記録（解釈は 研究者）。
2. **再測定**: gemma-3-270m / 1b / 4b / 12b / 27b の 5 点を B2 と同一条件（neutral・lengths {1K,2K,4K}・4 seq・query 64 位置）で再実行。出力 = `r_delta_<model>_v2.json`（旧ファイルは上書きしない）。
3. **整合チェック**: qwen3-1.7b を 1 点だけ再実行し、旧 `r_delta_qwen3-1.7b.json` と R が一致することを確認（full-attention では修正の影響ゼロのはず・不一致なら停止して報告）。

## 2. Job 2 — 横断層増員（7 モデル・R/A/erank の 3 因子）

各モデルで B2 と同一仕様の 3 probe を実行: `r_delta_probe.py`（修正版）・`a_retrieval_probe.py`（NIAH 4K・100 instances）・`erank_pr_probe.py`（neutral 100 prompts）。cka_band は**測らない**（B4 裁定で W 主変種 = 全層平均のため不要・band 感度は既存点のみで足りる）。

| 優先 | モデル（HF ID） | 狙い | y（公開行の閾値法再計算・85.6 連続超過） |
|---|---|---|---|
| 1 | aws-prototyping/MegaBeam-Mistral-7B-512k | **同サイズ対照**（Mistral-7B アーキ・v0.2 と対） | 32K |
| 2 | princeton-nlp/Llama-3-8B-ProLong-512k-Instruct | **同サイズ対照**（Llama3-8B アーキ） | 32K |
| 3 | gradientai/Llama-3-8B-Instruct-Gradient-1048k | **同サイズ対照**（同上・低 effective 側） | 16K |
| 4 | internlm/internlm2_5-7b-chat-1m | claimed 1M / effective 4K の乖離極端点 | 4K |
| 5 | LargeWorldModel/LWM-Text-Chat-1M | claimed 1M / effective <4K の下端極端点 | <4K（censored） |
| 6 | CohereForAI/c4ai-command-r-v01 | 35B・サイズ上端の追加 | 32K |
| 7 | （任意）FILM-7B = In2Training/FILM-7B | 同サイズ対照 3 点目（author-reported 行 = 開示つき） | パック §3 の行から 研究者 が再計算 |

- y の出典 = `data/t02_ruler_public_pack_2026-07-11.md` の NVIDIA 測定行（4/7 は上表に転記済み・FILM のみ author-reported）。**実行側は y に触らない**（研究者 が判定工程で確定）。
- 注意: internlm は `trust_remote_code=True` が要る見込み（probe の fallback 経路が対応済み）。gated/取得不能のモデルが出たら**代替せず skip して記録**（fallback 連鎖は組まない——同サイズ対照という選定理由が代替では満たせないため）。
- 出力 = `r_delta_<m>.json`・`a_retrieval_<m>.json`・`erank_pr_<m>.json`（命名は B2 と同一規則・新規モデルは v2 サフィックス不要）。

## 3. 回収と後処理

- 全 JSON + RUN_LOG を `data/t02_b5_2026-07-15/` にローカル回収・`SHA256SUMS.txt` 照合。
- HF キャッシュ削除（手動）。

## 4. B5 解析条件（事前固定・研究者 が実行）

**測定開始前に本書で凍結。データを見てからの変更はしない。**

1. **主解析 = B4 と同一仕様の再回帰**（`b4_regression.py` の MODEL_SPECS・LOFO-CV・permutation 1000 を変更なしで N≈22 に適用。Gemma R は v2 の全層平均を主・global 層限定を感度）。**報告様式 = B4 判定（機構>無機構 不成立）は不変のまま提示し、B5 は「拡張での頑健性確認」**。反転条件 = 機構最良が全ベースラインに勝ち、かつ（機構最良 − 最良ベースライン）の paired permutation p < 0.05 の両方が成立した場合のみ「拡張で逆転」と報告する（片方のみなら「不確定・N 依存」）。
2. **同サイズ対照解析（新規・記述統計）**: params がほぼ同一のグループ内（G1 = {mistral-7b, MegaBeam, (FILM)}・G2 = {llama-3.1-8b, ProLong, GradientAI-8B}・G3 = {internlm2.5-7b, LWM, mistral-7b, MegaBeam の 7B 帯}）で、各因子の順位が effective の順位と一致するか（グループ内 Kendall τ・N が小さいので検定でなく記述 + 全開示）。**サイズが弁別できない場所で因子が弁別できるか**が問い——B4 の「因子はサイズ代理」への最鋭の追試。
3. censored（LWM・既存 3 点）の扱い・y の再計算規約は B4/B3 から不変。
4. Gemma R v2 と救済値（global 層限定）の差分を開示表にする（判定書の訂正履歴と接続）。

## 5. 想定コストと中断規則

[cost thresholds omitted --- operational]
- 個別モデルの probe 失敗（アーキ非互換等）は長時間格闘しない——skip + 記録（B2 の GLM 教訓）。

## 6. Amendment 1（2026-07-17 研究者 裁定・Job 1 診断結果を受けた測定コード修復）

**経緯**: Job 1 診断（§1-1・計画済みの判断点）で、NaN の真因が測定コード側と確定した——`attn_weights` の実 dtype は bf16（probe の「fp32」注釈が誤り）で、`CLIP_FLOOR=1e-45` は bf16 キャストで 0.0 になりクランプが無効（bf16 最小 subnormal ≈9.2e-41）。sliding window 外の列は softmax 後に厳密ゼロ → `log(0)=-inf` → `0·(-inf)=nan`。ゼロ行は有限なので 07-15 の non-finite ガードは検出できない。**これは測定定義の変更ではなくバグ修復**（クランプは元々 log(0) 防止のために存在し、bf16 で無音で機能停止していた）——事前登録の凍結対象（判定条件 §4）は不変。

**修復**（`r_delta_probe.py`・研究者 実装済み・要計算資源同期）:
1. `CLIP_FLOOR = 1e-45 → 1e-38`（bf16 で表現可能・非ゼロ。正常域の非ゼロ値には無影響 = full-attention 測定は v1 と数値同一のはず）。**dtype は変更しない**（bf16 演算のまま——B2 の既存 18 モデルとのビット互換を保つため fp32 キャストは採らない）。
2. hook 冒頭に fail-closed assert（`CLIP_FLOOR` が `aw.dtype` で 0 に落ちたら即死）。
3. 誤っていた fp32 注釈の訂正 + 経緯コメント。

**受け入れゲート（fail-closed・全て通るまで Gemma v2 データを解析に使わない）**:
- (G1) 計算資源上で bf16 表現の実測: `torch.tensor(1e-38, dtype=torch.bfloat16).item() > 0` かつ `torch.zeros(4,dtype=torch.bfloat16).clamp(min=1e-38)` が全要素 1e-38 近傍・さらに合成ゼロ入り bf16 行で H が有限になること。
- (G2) qwen3-1.7b 整合チェック（§1-3）: 修正版 probe の R per-layer が旧 `r_delta_qwen3-1.7b.json` と一致（期待 = ビット同一。相対 1e-6 超の乖離が 1 層でもあれば停止・報告）。
- (G3) gemma-3-4b: global 層（full-attention 層）の R が B2 記録値と G2 と同じ許容で一致。
- (G4) gemma-3-4b: sliding 層が全て有限・`clip_count > 0`（クランプが実働している証拠）・per-head H ≤ log(window)+0.1。
- 判定・解釈は 研究者。ゲート結果は数値のまま RUN_LOG へ。

**開示**: (a) 本 Amendment は Job 1 診断データを見た後の裁定（診断は §1-1 で計画済みの判断点）。(b) sliding 層の δ は floor 支配（≈ log(a_max)+87.5）となり解釈不能——δ は B4 因子でなく P0-d アンカーも不成立済みのため判定に影響なし・報告時は sliding 層 δ を「window 打ち切りの artifact」とフラグ。(c) 真因の記録は 3 訂目: B4 開示 3 の原記述「0·log0 未ガード」が**機能的には正しかった**（ガードは存在したが bf16 で不活性 = 実質未ガード）——07-15 の訂正「NaN は attention 重み自体に含まれる」は誤りで、本 Amendment で再訂正。(d) 07-15 の単体検証（合成 NaN 4 ケース）が本バグを見逃した理由 = テストが fp32 テンソルで書かれていた——**テストは配備時 dtype で書く**（教訓）。

### Amendment 1 補遺（2026-07-17 研究者・G2 の字義不合格を受けたゲート較正）

G1 通過（bf16 実測 3 点。付記 = fp16 では 1e-38 が 0 にアンダーフロー——本測定は bf16 固定ロードのため無関係・assert が防護）。G2 は字義不合格: qwen3-1.7b R 0.855312→0.854720（相対 0.07%）・per-layer H_mean 最大相対差 0.476%（len2048 layer18）。ただし B2 は A100・B5 は H200——**GPU アーキ差の丸め蓄積と修正由来が未分離**のため、G2 を二段に較正する:

- **G2a（修正の単離・決定打）**: 同一 H200 上で v1（CLIP_FLOOR=1e-45 のみ戻した版）と v2 を qwen3-1.7b で直接比較。**期待 = ビット同一**（1e-38 クランプは正常域の非ゼロ値に無影響のはず）。ビット同一なら「0.48% は GPU 差・修正は full-attention の数値を変えない」が事実として確定 → G2 通過扱い。**同一でなければ停止・値を報告**（qwen の attention に 1e-38 未満の値が実在する想定外——調査要）。
- **G2b（GPU 差の開示）**: A100↔H200 の相対差（R 0.07%・per-layer ≤0.5%）を RUN_LOG に数値のまま記録。B4 再回帰への影響 = R のモデル間差は数十% 級であり 0.1% 級の測定系差は無視可能——ただし「B2=A100・B5=H200 の混在」を解析時開示に追加（研究者）。
- **G3 較正**: gemma-3-4b global 層の対 B2 比較は cross-GPU のため許容を相対 ≤1% へ緩和。修正単離側は G2a が担う（可能なら gemma-3-4b でも v1/v2 同一 GPU ビット同一を確認——診断 1 forward で安価）。
- G4 は不変。
