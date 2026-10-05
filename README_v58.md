# v58 — Budget-Neutral Linkage / QD / Adaptive Search

v57 の高速な MPS 評価器、Global Rule Pool、差分 pack、rolling evaluation、HoF、epsilon-lexicase、precise mutation を維持したまま、探索器を大きく作り直した版です。

## 狙い

従来は 1500 個の ordered rewrite rules を一つの巨大な genome として、two-point crossover と local / balanced / explore mutation で探索していました。これは個々の mutation は高速ですが、「離れた位置にある複数 rule が一緒に働く」場合の linkage を学習しません。

v58 は以下を統合します。

1. **Sparse linkage learning**
   - 上位 genome の既存 ridge readout weights だけから rule-position 間の依存を推定。
   - 既定では上位96 genome、重要128 lociだけを使い、最大96 modulesを作ります。
   - 1500×1500 の巨大な相関行列は作りません。
   - 8世代ごとに再学習するため、CPU側の追加負荷は小さいままです。

2. **Budget-neutral / deferred optimal mixing**
   - GOMEA の発想で、学習した非連続 module を donor からまとめて移植します。
   - donor と base の既存 readout mass から有望な module を選ぶため、追加評価なしで proposal を絞ります。
   - strict GOMEA のように「1 module移植するたびに full fitness evaluation」は行いません。これは現在の MPS では評価回数を数倍〜数十倍にするためです。
   - offspring は従来どおり次世代で一度だけ評価されます。

3. **MAP-Elites-style quality diversity**
   - すでに評価済みの population から sparse grid を構築。
   - descriptor: active-rule sparsity / latent embedding / moved embedding / important-rule pattern length / wildcard fraction / case specialty。
   - cell ごとの best を少数 next population に残し、異なる解き方を fitness 一本で消さないようにします。
   - 別枠の GPU 評価はありません。

4. **Adaptive operator bandit**
   - arms: `mix_local`, `mix_balanced`, `local`, `balanced`, `explore`, `legacy`。
   - 子 genome に親の case score と case serial を runtime metadata として持たせ、次世代の通常評価後に「同じ rolling case だけ」で child-parent delta を計算します。
   - 非定常 EMA + UCB で有効な探索 operator に予算を寄せます。
   - `legacy` arm として v57 の two-point crossover + precise mutation を残しているため、新方式が弱い局面では旧方式へ自動的に戻れます。

5. **既存 precise mutation を維持**
   - rule 内部の pattern / replacement mutation、swap、block move/duplication、embedding mutation は削除していません。
   - linkage mixing は「どの module を組み替えるか」、precise mutation は「module 内部をどう変えるか」という二階層探索です。

## 速度設計

最重要制約は「GPU fitness evaluation budget を増やさない」です。

評価対象は従来と同じく

```text
population + sampled HoF
```

だけです。linkage learning、QD grid、bandit は次の population を作る host-side 処理であり、追加の rewrite trajectory evaluation を呼びません。

既定値も整理しました。

```text
--hof-size 1024
--hof-eval 32
--hof-inject 4
--plot-every 25
```

元アーカイブ実コードに残っていた `hof-size=10000 / hof-eval=128 / hof-inject=128` は README と矛盾し、探索枠と評価時間を圧迫するため修正しています。

手元の 450×1500 checkpoint を使った host-side マイクロベンチでは、Sparse linkage 再学習は約 0.006 秒、QD grid は約 0.02 秒でした。この環境には Apple MPS がないため、実機 Metal の generation time は Mac 上で確認してください。ログには `search=...s` を追加しているので、探索器の host overhead を直接監視できます。

## コーパス修正

既存コードでは `max_chunk` より長い local/streaming chunk が crop されず、コメントアウトされた crop の直前で `continue` して丸ごと捨てられていました。v58 では本来の仕様どおり、長い chunk をランダム contiguous window に crop します。長いコードだけが学習データから消えるバイアスを避けます。

## 推奨実行

既存 checkpoint はそのままロードできます。

```bash
python3 main.py --backend mps \
  --load minimal_gp_checkpoint.npz \
  --checkpoint minimal_gp_checkpoint_v58.npz \
  --save best_minimal_gp_v58.json \
  --current-save latest_minimal_gp_v58.json \
  --history-csv fitness_history_v58.csv \
  --plot-prefix training_v58
```

新しい探索器の主要既定値:

```text
--qd-bins 6
--qd-inject 24
--qd-parent-rate 0.08
--linkage-refresh 8
--linkage-elites 96
--linkage-loci 128
--linkage-modules 96
--linkage-max-module 16
--operator-ucb 0.0003
--operator-epsilon 0.04
```

### A/B test

linkage mixing を実質無効化:

```bash
--linkage-modules 0
```

QD survivor injection を無効化:

```bash
--qd-inject 0 --qd-parent-rate 0
```

HoF 評価をさらに減らす:

```bash
--hof-eval 16
```

## ログ

通常 generation 行に以下が増えます。

```text
link=96 qd=... search=0.0xxs
```

breeding 行では:

```text
mix_rows=...
ops[mix_local:.../mix_balanced:.../local:.../balanced:.../explore:.../legacy:...]
```

CSV には `qd_cells`, `linkage_modules`, `linkage_loci`, `operator_updates`, `operator_reward_mean`, `search_model_seconds` と各 arm の累積 count / reward / success が保存されます。checkpoint format 自体は v57 互換のままです。

## 検証

実施済み:

```bash
python3 main.py --self-test
python3 test_improvements.py
python3 test_v58_search.py
```

- 既存 v57 tests: 9/9 PASS
- v58 search tests: 5/5 PASS
- CPU の複数世代 smoke training: PASS
- checkpoint save/resume: PASS
- smoke run で HoF 評価 budget 上限を確認
- 既存 MPS orchestration mock test: PASS

実機 Apple GPU はこの環境にはないため、Metal kernel 自体は変更せず、既存 test の CPU reference / mocked MPS consistency を維持しています。
