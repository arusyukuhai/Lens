# v59 — All-Sample Pairwise Rank Ridge + Full Spearman

v58 の Budget-Neutral Linkage / QD / Adaptive Search を維持したまま、readout と fitness の目的関数を順位最適化へ揃えた版です。

## 変更の中心

従来は既定 `--cases 3 --samples 36`、合計108 trajectoriesのうち、`sample_id % 3 == 0` の36本だけで raw cleanliness を dual Ridge 回帰し、残り72本だけで held-out Spearman を計算していました。

v59 の既定 `--readout-mode pairwise-all` では holdout を廃止します。

```text
3 cases × 36 trajectories = 108 trajectories

Pairwise Rank Ridge:
  caseごとに36本をtarget rank順へ並べる
  adjacent 35 pairs / case
  3 × 35 = 105 pair constraints
  108 trajectoriesすべてがpair graphに参加

Spearman fitness:
  36 trajectories / case 全部
  3 caseのSpearman平均
```

したがって、同じ108 trajectoriesすべてを readout fitting と fitness の両方に使います。

## Pairwise objective

予測は従来どおり

```text
prediction = overflow_baseline + X @ w
```

です。各case内でtarget rankが隣接する2 trajectoryを `hi`, `lo` とすると、

```text
(X_hi - X_lo) @ w
  ~= normalized_rank_margin - (baseline_hi - baseline_lo)
```

をL2正則化付きRidgeで解きます。

全ペア `36 choose 2` を使うと1case630 constraintsになりreadoutが重くなるため、rank chainの隣接35本だけを使います。これでも全36 trajectoryが少なくとも1 constraintへ入り、推移律を通じてcase全体の順序を学習できます。tiesはaverage rankを使い、同順位間はmargin=0になります。

## 速度対策

105×1500 pair matrixに対して105×105 Gramを全1500 ruleで作ると旧readoutよりhost負荷が増えます。v59では全105 pair・全1500 columnsをまず一回だけ走査し、normalized pairwise covarianceで上位256 rule columnsを選び、その256列だけに対して **exact dual Ridge** を解きます。

既定値:

```text
--rank-max-features 256
```

重要なのは、feature screening前のrelevance計算には105 pair全部が使われるため、108 trajectoriesの参加条件は変わらないことです。

1500-rule相当の疎な合成 firing matrix でのhost microbenchmark例:

```text
legacy 36-row exact ridge : 約0.7〜1.0 ms / genome
v59 105-pair screened rank ridge : 約1.4〜1.5 ms / genome
```

計測ノイズはありますが、450 genomesでの追加host時間は概算0.2〜0.4秒/世代程度です。Metal rewrite evaluationが数十秒級なら小さい割合です。実機値はMacで `readout=...` ログを確認してください。

## 目的関数の整合

v58以前:

```text
fit: raw cleanliness MSE + L2
select: held-out Spearman
```

v59:

```text
fit: case-local pairwise rank constraints + L2
select: same case内全trajectoryのSpearman
```

したがって、readoutもselectionも「値そのもの」ではなく順位を重視します。

## 注意: in-sample fitness

v59の既定方式は、同じ108 trajectoriesをfitとSpearman採点の両方に使います。そのため従来のheld-out fitnessより楽観的になり、個々のrolling caseへの過学習は増え得ます。

ただしrolling evaluation自体は残っており、case slotは世代をまたいで順次入れ替わります。完全固定データへのfitではありません。

外部generalizationを測る場合は既存 `audit_generalization.py` を使うか、A/B用に旧方式へ戻せます。

```bash
--readout-mode legacy-holdout
```

## 推奨実行

既存checkpointはそのままロードできます。

```bash
python3 main.py --backend mps \
  --load minimal_gp_checkpoint.npz \
  --checkpoint minimal_gp_checkpoint_v59.npz \
  --save best_minimal_gp_v59.json \
  --current-save latest_minimal_gp_v59.json \
  --history-csv fitness_history_v59.csv \
  --plot-prefix training_v59
```

明示する場合:

```text
--readout-mode pairwise-all
--rank-max-features 256
```

## v58探索器は維持

- Sparse linkage learning
- Budget-neutral / deferred optimal mixing
- MAP-Elites-style quality diversity
- Adaptive operator bandit
- legacy crossover arm
- precise rule mutation
- rolling HoF
- Global Rule Pool / differential pack / MPS evaluator

いずれも追加GPU fitness evaluationを発生させません。

## 検証

```bash
python3 main.py --self-test
python3 -m unittest -v test_improvements.py test_v58_search.py test_v59_rank_readout.py
```

確認済み:

- existing tests: 14/14 PASS
- v59 rank-readout tests: 3/3 PASS
- 合計17/17 PASS
- 3×36=108 trajectoriesでCPU smoke training 3 generations: PASS
- 3×36から105 adjacent pairs生成: PASS
- 108 trajectoryすべてがpair graphへ参加: PASS
- caseを跨ぐpairなし: PASS
- full-36 Spearman / case: PASS
- checkpoint / rolling evaluation / HoF / v58 searchとの互換: PASS

Apple MPS実機はこの環境にはないため、Metal kernel自体には変更を加えていません。
