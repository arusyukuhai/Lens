# Lens Serial: 微分可能な直列置換による破損率学習

添付 `cms(9).nim` の課題を、勾配学習するPyTorch実験実装へ移したものです。
**1500規則は1500段の直列接続です。規則をグループ化・並列適用しません。**
規則 i は規則 i-1 の出力を受け取ります。**文字集合は常に0〜255の256バイトです。**
各規則の置換先も既存256バイトから学習し、規則ごとの専用中間記号は導入しません。
中間教師は使わず、最終的な文字の残存率だけを教師にします。

## 元のLensとの対応

| 項目 | この実装 |
|---|---|
| 学習入力 | `===SPLIT===` で区切るUTF-8等の**生バイト**文書。各文書は既定2000バイトで切る |
| 拡散 | ランダム順に位置を1回ずつ別の文字へ変更。観測する中間段階を重複なしでランダム抽出 |
| 端点 | 原文・全位置破損を必ず含める |
| 教師 | `1 - 実際に変更した文字数 / 原文長` |
| 規則 | 常に256バイトの固定アルファベット。各規則の左右・中央・置換先を学習 |
| 置換 | 左から非重複で適用し、学習した0〜255の1バイトとキャプチャを出力。次の規則へ直接渡す |
| 特徴 | 各規則の期待発火回数÷**元の入力長**。既定1500次元 |
| 回帰 | `1500 → 2048 → 2048 → 1`、隠れ層はSwiGLU |
| 探索 | GA/Ridgeの代わりに、全規則とMLPをAdamWで同時学習 |
| 並列化 | 独立した文章・拡散段階をバッチ化。規則の適用順序は維持 |

**完全なNim版の移植ではありません。** 現段階の規則文法は
`[左の記号, 中央の記号またはSTAR, 右の記号]` です。
中央が通常記号なら3記号一致、STARなら0〜`--max-capture`記号をキャプチャし、
最初の右区切りで終了します。既定の最大キャプチャは4。
長さ12までの任意パターン・2個のワイルドカード・無制限のキャプチャ長・sort/reverse等は未実装です。
**この256バイト固定文法の離散化後の意味論**について参照実装と一致をテストしています。

確率入力の連続実装は、周辺分布と期待位置による近似です。
入力の全256成分を使いますが、相関を含む文字列全分布の厳密な期待値ではありません。
非整数長の存在マスクと位置配置に5次smoothstepを用います。
学習中に文字列・規則をargmax化せず、層間のdetachや打ち切り逆伝播もしません。
深い連続置換には位置・長さ誤差や勾配の減衰が蓄積する可能性があります。

## インストール

Python 3.10以上、PyTorchが必要です。CPU動作確認環境はPyTorch 2.14.1+cpu。

```bash
python -m pip install -r requirements.txt
python test_lens.py
```

CUDA版PyTorchは使用するGPU環境向けにインストールしてください。
Apple SiliconではMPSが利用可能なPyTorch環境で `--device mps` を指定します。
明示指定したデバイスが使えなければエラーにし、CPUへ黙って切り替えません。
この作成環境はCPUのみで、**CUDA/MPSの実行・速度・メモリは未検証**です。

## 学習

元の形式のコーパスを指定します。検証・テストは文書単位で分割し、完全重複文書も除きます。
学習集合のバイトだけから置換ノイズ用アルファベットを作ります。
規則は学習文書から採った3-gramで初期化し、置換先バイトは0〜255からランダム初期化します。
正解規則・中間教師は与えませんが、
完全な一様初期化からの獲得を実証したものでもありません。

CUDA例（まず短めの入力で実際のメモリを確認し、その後2000等に上げてください）:

```bash
python train.py --corpus github-code.txt --out run_cuda --device cuda \
  --rules 1500 --hidden 2048 --chunk-len 512 --max-capture 4 \
  --batch-chunks 4 --levels 4 --microbatch 2 --accumulate 2 \
  --checkpoint-every 25 --steps 2000 --reg-max .02 --reg-cycles 4
```

MPS例:

```bash
python train.py --corpus github-code.txt --out run_mps --device mps \
  --rules 1500 --hidden 2048 --chunk-len 512 --max-capture 4 \
  --batch-chunks 2 --levels 4 --microbatch 1 --accumulate 4 \
  --checkpoint-every 25 --steps 2000 --reg-max .02 --reg-cycles 4
```

`batch-chunks × levels × accumulate` が通常の1更新の標本数です。
各文書が短く、異なる段階をその数だけ取れない場合はlevelsを減らします。
`microbatch` は同時にデバイスへ流す標本数です。余裕があれば増やすことでバッチ並列度を上げられます。
CPUデータ生成は文書と拡散段階をテンソル化し、CUDA転送時はpin memoryと非同期転送を使います。
MSEの勾配は各microbatchの実際の標本数で重み付けするため、分割で目的が変わりません。

`--checkpoint-every 25` は**25規則ずつ直列に実行した区間を再計算する**指定です。
25規則を同時適用する意味ではありません。0で再計算なし。
小さくしすぎると保存する境界が増え、大きくしすぎると再計算中の一時メモリが増えます。

`--amp` は対応CUDA GPUでMLPだけをBF16にします。確率計算・prefix scan・位置配置はFP32です。
MPSはFP32 eagerを使います。CUDA/CPUでは `--compile-head` でMLPだけのコンパイルを試せます。
1500規則全体を巨大な一つのコンパイルグラフにはしていません。
forward内に各文書のPythonループ・CPUへの往復はありません。
データ生成と規則の初期化・エクスポートはCPUを使います。

## 損失と振動する離散化正則化

既定値は

\[
\mathcal L=\operatorname{mean}(\hat y-y)^2+\lambda(t)\mathcal R,
\qquad \mathcal R=\operatorname{mean}_{\text{rule slots}}(1-\|p\|_2^2).
\]

実装では各規則の「外側2スロットの平均」「中央スロット」「置換先スロット」を
それぞれ同じ重みで平均します。
この多項式正則化はone-hotで0です。規則を離散化へ促し、**入力の不確かさ自体は罰しません**。
softmaxパラメータの有限値では厳密なone-hotにはならず、最終エクスポートでargmaxします。

学習進捗を `t=step/(steps-1)`、warmup比率を `w` として

\[
u=\operatorname{clip}((t-w)/(1-w),0,1),\qquad
\lambda(t)=\lambda_{\max}u\sin^2\!\left(\pi(k+\tfrac12)u\right).
\]

既定 `w=.1, k=4, λmax=.02`。包絡線は徐々に上がり、その中で正則化を強めたり弱めたりします。
最後は必ず `λmax` で終わります。係数は各更新中は定数で、モデルの2階微分を妨げません。
温度は既定1で固定です。正則化係数の振動と温度の変化を混同しない設計です。
`--temperature` で固定温度を変更できます。

MSEはPearsonの中心化・分散正規化を省き、既知の残存率への較正を直接学習できます。
両者とも標本数に対してO(B)で、損失変更による大幅な高速化を実証したわけではありません。
計算量の中心は1500段の置換です。Pearsonは検証時に全評価行で計算します。
`--loss pearson` も使えますが、microbatch分割と勾配蓄積では元の全バッチPearsonに一致しないため、
`microbatch >= batch-chunks*levels, accumulate=1` を必須とします。
滑らかなロバスト損失は `--loss pseudo_huber`。既定MSEもこちらもC∞です。

## 出力・再開・離散実行

- `last.pt`: 最終の連続学習モデル、optimizer、データ乱数・順序・位置。検証ごとに更新。
- `config.json`, `split.json`: 設定と文書SHA256による分割記録。
- `train.jsonl`: 教師損失、正則化係数、不純度、勾配ノルム。
- `eval.jsonl`: 固定検証文書のsoft/hard評価。検証で早期停止・モデル選別はしません。
- `rules.json`: 離散化した1500段の規則。
- `results.json`: テスト文書のsoft/hard MSE、MAE、Pearson。

同じ設定・同じコーパスで `--resume run_cuda/last.pt` を付けると続きから実行します。
再開時に総更新数やアニーリング設定を変更することは許しません（係数の履歴が変わるため）。
実機・デバイスを変更した場合までビット一致を保証するものではありません。

```bash
python predict.py run_cuda/last.pt input.txt --device cuda --soft
python predict.py run_cuda/last.pt input.txt --device cuda
python predict.py run_cuda/last.pt input.txt --device cpu --integer
```

既定predictは離散規則で実行します。`--integer` はPythonの整数置換参照経路を使います。
置換と発火回数カウントは整数演算ですが、特徴量の正規化・MLPは浮動小数点です。
出力は較正前の実数スコアで、[0,1]へのクリップは行いません。

`--hard-finetune 50` を学習時に追加すると、最後に規則を固定し、整数化した規則の特徴で
MLPだけを追加学習します。中間教師は使いません。結果は別の `hard_finetuned.pt` に保存し、
連続学習用の `last.pt` を上書きしません。これはsoft/hardの差をMLP側で調整する処理であり、
離散化で失われた情報を必ず回復できるわけではありません。

## 性能の測定

```bash
python benchmark.py --device cuda --rules 1500 --hidden 2048 --length 512 --batch 2
python benchmark.py --device mps --rules 1500 --hidden 2048 --length 512 --batch 2
```

warmupを除き、デバイス同期を挟んだforward+backward時間を測定します。
**1500段の依存関係は消せないので、通常のMLP並みのGPU効率は保証しません。**
位置走査は小さな状態遷移行列を平衡木で合成し、下りでは状態ベクトルだけを伝搬します。
総演算量O(B L S³)、依存深さO(log L)です。Pythonで各位置を逐次走査しません。
これは規則の適用順序を変えない、各規則内部のテンソル演算です。

入力長L、バッチB、規則数N、最大消費長Sとして、全段でおおむね
`O(N B L (256 + S³))` の演算量です（低次項は省略）。
**語彙増大による N² 項はなくなりました。** 各段の文字分布は常に
`[B,L,256]` で、規則パラメータも語彙サイズに関して `O(256N)` です。
ただし、規則間の直列依存と走査・パッキングにかかる計算は残ります。

CPUでの短い学習試験は再実行可能です。古い版の学習結果や重みは、
置換先の定義が異なるため新しい版の検証結果として使えません。

## 参照したPyTorch公式文書

- https://docs.pytorch.org/docs/stable/checkpoint
- https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- https://docs.pytorch.org/docs/main/notes/mps.html

## 走査の高速化更新

旧走査との数値比較は `verify_scan_upgrade.py` で再実行できます。
高速化した走査はこの版でも引き続き使用します。規則の直列接続と
非重複走査は維持していますが、置換先を256バイトへ固定したため、
**旧版チェックポイントとは互換ではありません**。浮動小数点の演算順序は変わります。

```bash
python verify_scan_upgrade.py --device cpu --output scan_upgrade.json
# 実機では --device cuda または --device mps
```

## 256バイト固定版への変更（重要）

- 以前: 規則 `i` は毎回固有の `256+i` を生成し、後段ほどアルファベットが大きくなる。
- 現在: 規則 `i` は確率ベクトル `replacement ∈ Δ²⁵⁵` を持ち、同じ256バイトのいずれかを出力する。hardモードでは `argmax` の1バイトを出力する。
- 追加で学習するのは置換先バイトの256 logits。中間教師や専用マーカーは不要。
- `rules.json` の `marker` は廃止し、`replacement_byte` と `replacement` を出力する。
- 意味論は意図的に変更しています。**バイトの再利用に伴う衝突はあり得ます**。固有の中間記号と同じ識別能力を保持するわけではありません。
- 旧 `last.pt` / `hard_finetuned.pt` は新しい `model.py` に直接ロードできません。新規学習が必要です。

`FIXED_256.md` に変更理由・検証をまとめています。
