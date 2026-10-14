# Lens Serial GA（Nim）

アップロードされた `replace(4).nim` をもとにした既存の最適化済み `replace_core.nim` を継承し、**一直線のReplacerのみ**に統一した実験版です（互換用 `--arch=chain` は受け付けます）。`diff3` は**ネットワーク内・推論時には使いません**。遺伝子として保持した短い置換パターン／置換文字列を交叉するときのみ使います。

## 実行

Nim 2.x推奨・追加Nimパッケージ不要。`gnuplot`は画像生成時のみ外部コマンドとして呼び出します。

```bash
nim c -d:release --opt:speed lens_ga.nim
./lens_ga --self-test
./lens_ga --corpus=github-code.txt --generations=100 --nodes=1296 \
  --population=16 --elites=3 --offspring=24 --immigrants=2 \
  --train-chunks=12 --anchor-chunks=2 --rotate-chunks=4 --rotate-stride=1 \
  --valid-chunks=2 --test-chunks=2 --selected-features=16 \
  --plot-every=5 --out=lens_training --save=lens_serial_ga.json
```

学習済みの再評価：

```bash
./lens_ga --load=lens_serial_ga.json --nodes=1296 --evaluate --corpus=github-code.txt
```

短い動作試験：

```bash
./lens_ga --corpus=github-code.txt --nodes=32 --generations=2 \
  --population=4 --elites=1 --offspring=4 --immigrants=1 \
  --train-chunks=4 --anchor-chunks=1 --rotate-chunks=2 \
  --valid-chunks=1 --test-chunks=1 --noise-levels=6 \
  --selected-features=4 --plot-every=1
```

コーパスは従来どおり `===SPLIT===` で分割したUTF-8テキスト。文字列は**バイト列**として扱い、マスク記号に整数256を使います。128バイト（既定）に切り出してから段階的にマスクします。`--noise-levels=0` なら各文字を1文字ずつマスクして全段階を生成します。

## 遺伝子と変異

1個体は1296個（`--nodes`で変更可能）の `(pattern, replacement)` 列です。各遺伝子は先頭から順に、一度ずつ文字列へ適用されます。特徴ベクトルの各座標は **置換できたか否かではなく、実際に実行した非重複置換回数**。同一文字列への置換も計数します。最大出力長を超えた置換は棄却・0件として処理します。

- **Point**：1カ所の規則文字列内部の編集。置換、挿入、削除、隣接交換、短い部分文字列の反転・複製、ワイルドカード／キャプチャ参照の導入・除去、コーパスからの短い断片挿入など。
- **Span**：複数の規則に対して、複数回の内部編集をまとめて実行。
- **GeneSplice**：親2個体の同じ位置のパターン文字列および置換文字列で切断点を選んで接ぎ合わせる。
- **Block**：連続する1〜64個の置換規則を、もう一方の親から移す。任意で点変異を追加。
- **Diff3**：異なる親3個体の規則文字列に3-way diffを適用。`D(a,b,c)` の競合位置は元の `a` に戻す。これは**数値的DEそのものではなく**、`a→b` の編集を `c` に移植する**差分進化に着想を得た離散編集交叉**。
- **Immigrant**：新規ランダム個体を毎世代流入させて多様性を維持。

交叉では両親／3親が実際に異なる遺伝子位置を優先して選択。自明に同じになる場合は内部突然変異にフォールバックします。操作ごとの改善割合に応じて選択重みを緩やかに適応変更します。

パターンの負数はワイルドカード、置換側の負数はキャプチャ参照（-1〜-15のみ）。**`-16`以下のsort/reverse/算術などの命令は生成・使用しません**。交叉後はワイルドカード参照を修復し、再コンパイルします。

## 評価とローテーション

- `--train-chunks` のうち末尾 `--anchor-chunks` は固定Anchorとして確保し、それ以外を `--rotate-chunks` 幅のローリング窓で世代ごとに切替。
- `--rotate-stride=1`なら訓練チャンクを順番に巡回。一般に全チャンクを公平に巡回するには、strideと回転プールのチャンク数を互いに素にしてください。
- 遺伝的選択のfitnessは `0.35 * rotatingTrainPearson + 0.65 * anchorPearson`。Anchorは学習用チャンクから別分割した固定例。これにより世代間でまったく異なる窓のスコアだけを見る問題を緩和します。
- 親3個体のトーナメント選択、上位エリート維持、Hall-of-Fame再投入、候補のfitness順採択、重複ゲノム除去、毎世代のImmigrant導入を行います。
- `--valid-chunks` は**監視専用**で、遺伝的採択やChampion選出には使用しません。`--test-chunks` は最終評価専用で、選択には一切使いません。
- Championは、固定Anchor上で観測したPearsonの最高値で保存・更新（Anchorへの適合は起こりえます）。保存時・終了時は**全訓練チャンク**から改めてOLS係数を計算。回帰にRidge正則化は使用せず、前向き列選択＋QRベースの解析的OLSを採用します。既定16列。
- ノイズ系列は実行開始時に生成して固定し、世代ごとにマスク位置の乱数が入れ替わることで評価がぶれるのを防ぎます。

## 性能上の工夫

子個体は親をベースに生まれるため、変更された規則より前の逐次処理は同じです。**64段ごとの中間文字列スナップショット**を保存し、子個体の評価では最初の変更位置以前の特徴量とスナップショットを親から再利用します。これは毎回全1296段を再計算するより効率的です。世代ごとに訓練チャンクが変わった場合はキャッシュを作り直します。diff3を推論経路で用いることはありません。

## gnuplot

デフォルト `--out=lens_training` に以下を出力します。

- `training.tsv`：世代ごとの `train_q`, `anchor_q`, `valid_q`, `best_anchor_q`。各値は **`-log2(1 - Pearson)`**。
- `training.gp`：gnuplotスクリプト。
- `training.png`：ローテーションTrain／固定Anchor／監視Validation／Anchor最高値。
- `training_detail.png`：AnchorとValidationを独立のY軸で見やすく描画。

`gnuplot` が存在しない場合でも学習・TSV出力は継続します。後から次を実行してください。

```bash
cd lens_training
gnuplot training.gp
```

`-log2(1-r)` は `r=1` で発散するため、画像用には `1-1e-12` で上側だけクリップします。Pearson自体はクリップ値を適合度に用いません。

## 保存形式

`LensSerialGA-v1`のJSONを出力。旧`LensTripleGate-v1`の`arch=chain`形式は読み込み可能ですが、`arch=grid`は読み込み拒否します。JSONの重みは元データの訓練チャンクで再推定したOLS重みです。

## 未検証事項

この実行環境にはNim/gnuplot実行バイナリがなく、実コンパイル・学習・画像出力は未検証です。`--self-test` は置換回数、変異・交叉、逐次計算キャッシュと完全再計算の一致、OLS、モデル保存を検査します。研究用途の探索原型であり、安定収束や新規構造の性能優位は実験で確認する必要があります。
