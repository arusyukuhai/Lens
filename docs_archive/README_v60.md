# v60 — String-side Genetic Inference + Spearman/Recovery Pareto

v59 の all-sample pairwise rank readout と v58 の linkage/QD/operator search を残したまま、**実際に入力文字列側を探索して復元できるか**を第二目的に追加した版です。

## 何を追加したか

各 outer generation で、通常の Spearman 評価後に独立した短い code snippet を数個用意します。snippet には既知の位置だけ人工ノイズを入れます。

```text
clean text
   ↓ corruption (positions are recorded)
noisy text
   ↓ inner genetic search: only corrupted positions may change
candidate strings
   ↓ current Replace population scores candidates
best-scoring inferred strings
   ↓ compare to hidden clean bytes at corrupted positions only
recovery accuracy
```

未変更部分を分母へ入れると accuracy が不当に高くなるため、第二目的 `inference_accuracy` は **人工的に壊した byte の復元率だけ**です。

clean byte は mutation proposal や candidate fitness には一切使わず、最終採点にだけ使います。

## Inner GA

既定値は以下です。

```text
--inference-cases 3
--inference-population 12
--inference-generations 4
--inference-elites 3
--inference-span 192
--inference-noise 0.05
--inference-ensemble 64
--inference-rotate-every 2
--inference-budget-ratio 0.75
```

候補文字列は corrupted loci だけを gene として扱います。mutation は corpus byte distribution と現在の局所 context から byte を提案し、clean target は見ません。crossover も corrupted loci だけで行います。

candidate pool は全 genome 共通です。上位 Spearman genome 群だけが search generation 中の candidate を rank 化して投票し、その平均 rank で shared candidate population を進化させます。raw readout scale の大きい genome が投票を独占しないよう、tie-safe な平均順位を使います。

探索が終わったら **最終 candidate pool だけ** を全 genome で一度評価します。各 genome の第二目的は、その genome 自身の readout が最終 pool 内で最も高く評価した candidate の復元率を case 平均した値です。したがって `inference_accuracy` は genome ごとに異なりますが、中間世代を全450 genomeで評価する無駄はありません。

### エリート保存

文字列側GAには `--inference-elites`（既定3）を追加し、各 inner generation で上位候補を **byte-for-byte そのまま次世代へコピー**します。elite copy 自体には crossover / mutation を掛けません。候補数が小さくなった場合でも最低1枠は offspring 用に残すため、budget縮小時に elite だけで population が埋まって探索停止することもありません。

## Pareto selection

outer population は

```text
objective 1: Spearman fitness  (maximize)
objective 2: inference_accuracy (maximize)
```

の2目的を NSGA-II 型 non-dominated sorting + crowding distance で選択します。

- elite survivor: **明示的な survivor elitism**。Pareto knee / best Spearman / best inference を優先的に固定し、残りを Pareto rank / crowding 順で `--elites` 枠まで無改変コピー
- tournament parent selection: Pareto rank / crowding 順
- QD cell representative: Pareto comparator
- linkage elite set: Pareto 順

HoF は従来の「rank representation の長期 archive」という役割を保つため、admission 自体は top Spearman candidates を使います。

`best_minimal_gp.json` は従来互換で best-observed Spearman、`latest_minimal_gp.json` は現在の Pareto front の knee（Spearman と inference のバランス点）を保存します。

## 2倍を超えにくくする予算制御

checked revision では inner GA の仕事量を **full-population trajectory equivalent** で見積もります。

```text
cases * candidate_population * (1 + generations * ensemble_size / population_size)
```

`1` は最終 candidate pool を全 genome で採点する1回分です。search generation は上位 ensemble だけを評価します。`--inference-budget-ratio` には引き続き 0.85 の hard cap を掛けます。

デフォルト population=450, ensemble=64, cases=3, candidates=12, generations=4 なら、

```text
3 * 12 * (1 + 4*64/450) = 56.48 equivalent trajectories
outer = 4 * 50 = 200 trajectories
nominal extra ~= 28%
```

となります。inference snippet は既定192 bytesで通常trajectoryより短いため、初版v60よりかなり余裕を持って2倍未満を狙える構成です。

Apple MPS では同じ persistent evaluator / Global Rule Pool を再利用し、active sample prefix に加えて、**同じ search ensemble の rule/index pack を inner generation 間で再利用**します。search ensemble は1回、最後の全population採点でもう1回だけpackします。

## グラフ

`training_saturation.png` は左軸が従来どおり

```text
-log2(1 - Spearman)
```

右軸が

```text
corrupted-byte recovery accuracy [0, 1]
```

です。右軸には

- `inference raw` — 現在の Pareto-knee genome の実測 recovery（gray）
- `inference best` — その世代の population 内の最良 recovery
- `inference MA(200)` — raw recovery の移動平均

を重ねます。

history CSV には `inference_raw`, `inference_best`, `inference_best_ever`, `inference_mean`, `inference_seconds`, `inference_model_jobs`, `inference_equivalent_work`, `inference_search_rounds`, `pareto_front_size` 等も保存します。

## checkpoint compatibility

v59以前の version=1 checkpoint をそのままロードできます。新しい inference/Pareto arrays が無い旧 checkpoint は NaN/default で読み込み、最初の v60 generation で再評価します。

## 推奨実行

```bash
python3 main.py --backend mps \
  --load minimal_gp_checkpoint.npz \
  --checkpoint minimal_gp_checkpoint_v60.npz \
  --save best_minimal_gp_v60.json \
  --current-save latest_minimal_gp_v60.json \
  --history-csv fitness_history_v60.csv \
  --plot-prefix training_v60
```

負荷をさらに下げるなら例えば:

```bash
--inference-budget-ratio 0.50 --inference-generations 3
```

A/B 用に第二目的を完全に切る場合:

```bash
--no-inference
```

## 検証

```bash
python3 main.py --self-test
python3 -m unittest -v \
  test_improvements.py \
  test_v58_search.py \
  test_v59_rank_readout.py \
  test_v60_inference.py
```

v60 では追加で以下をテストします。

- inference work hard cap
- mutation が corrupted loci 以外を変更しないこと
- 2目的 Pareto front/rank
- checkpoint pack/unpack で inference/Pareto metadata が保存されること
- CPU smoke training で inner GA + Pareto breeding が通ること

この環境では Apple MPS 実機を持たないため、Metal kernel の実機 benchmark は未実施です。host orchestration と CPU path はテスト済みです。


## Checked revision

追加監査で見つかった修正点の詳細は `AUDIT_v60.md` を参照してください。

## v60.1: `inferBest=0.000` 修正

elite 保存後にも `inferBest=0.000` が固定されるケースを再現し、原因を修正しました。
最終候補選択で `np.argmax` を使っていたため、複数候補が同点のとき常に index 0、つまり
「何も直していない noisy string」が選ばれていました。rule firing ベースの readout では
候補間の完全同点が珍しくないため、これは強い no-op バイアスになります。

v60.1 は最大スコア同点集合を均等に扱い、候補配列の順番ではなく「同点なら一様に選ぶ」と
仮定した期待 recovery を第二目的にします。clean target は同点集合の決定には一切使いません。

さらに exact-byte recovery の探索が疎すぎたため、checkpoint に保存される n-gram pool から
軽量な前後文字 transition proposal を作り、corrupted loci を round-robin で探索します。
mutation は原則1文字だけにして、正しい文字の候補生成率と score attribution を改善しました。
デフォルト inner generations は 4 -> 6 ですが、従来の work cap はそのまま有効です。

ログには以下も追加されています。

```text
infEns=... infOracle=... infMove=...
```

`infOracle` は「最終候補群の中に実際に正解文字を含む候補があったか」の診断値だけで、
GA や Pareto 選択には使いません。`infOracle>0` なのに `inferBest=0` ならモデル側のランキングが
原因、両方0なら candidate proposal/search 側が原因、と切り分けできます。

## v60.2: log-uniform inference mutation radius

String-side GA mutation no longer uses the v60.1 mostly-1-locus heuristic. For a case with `k` known corrupted loci, a normal mutation now draws

`round(exp(uniform(0, log(k))))`

clamped to `[1, k]`, then mutates that many distinct corrupted positions. If the round-robin `focus_position` is active, that locus is guaranteed to be included and the remaining loci are sampled without replacement. `force_changes` remains an explicit override used only for duplicate-recovery paths. No clean/unmodified position can enter the mutation set.
