# Lens — 旧Replaceに準拠した直列学習器（GPU / MPS / CPU）

**これは旧版の置換デモだけを再配布したものではありません。**
`train.py` は任意個数の直列Replaceルールを学習するLens学習器です。学習用の可微分ルール列、
256バイトの文字表現、原文長で割った各ルールの発火回数、`R → 2048 → 2048 → 1` のSwiGLUヘッド、
GPU/MPS対応バッチ学習、梯度チェックポイント、固定検証/テストセット、再開用チェックポイント、
ノイズ率予測を備えています。

## 最も重要な区別：厳密性

| 処理 | 旧Replaceに対する保証 | 微分可能 | 用途 |
|---|---|---|---|
| `scalable` の **hard / 離散推論** | 旧 `discrete_replace.py` と同じ仕様 | ルールパラメータには不可 | 通常の推論/検証 |
| `scalable` の **soft 学習** | 旧ワイルドカードへのC²平均場近似。一部の相関は保存しない | 2階まで | CUDA/MPS、多規則の学習 |
| `exact-small` 学習 | 有限サポート上の**全相関付き確率分布**を正確に伝播 | 2階まで | 数学的な検証・ごく小規模の学習 |

**1500ルールの完全な確率的pushforwardを高速・厳密に計算する実装ではありません。**
一般に規則と出力文字列の候補が組合せ爆発します。`exact-small` は状態数上限を超えたら明示的に停止します。
`scalable` では文字列長の有限テンソル容量超過時にエラーを返し、黙って切り捨てません。

## 旧ワイルドカードの仕様（旧版へ復帰）

- 1規則の置換元・置換先はそれぞれ**可変長**。`--max-pattern` / `--max-replacement` で長さ上限を設定。
- パターンでは `-1`, `-2`, … が**出現順のキャプチャ**。構文オペコードであり、257番目以降の文字ではない。最大15個を設定可能（`--max-stars`）。
- 置換先の `-1` は最初の捕捉、`-2` は2番目。複数回参照・順序入れ替えに対応。
- `ab* -> Q` は `abXYZ` を `Q` に。**末尾ワイルドカードは残りをすべて捕捉**する。
- `a*1b*2 -> *2*1` は `a12b34 -> 3412` に。区切りの `b` は最初の出現を選ぶ。
- 左から非重複置換。全部ワイルドカードだけの置換元は恒等変換。変換を**1規則ずつ直列に**適用。
- `sort`, `reverse`, `inverse_sort` 拡張は含めない（添付旧版の捕捉/置換意味論へ戻す）。
- 捕捉**文字数**の上限は設けない（softの系列テンソルの確保上限は別）。

## インストール

```bash
pip install -r requirements.txt
```

## 直列Lensを学習する（scalable）

まず小さな設定で動作を確認してください。

```bash
python train.py --mode scalable --synthetic --device cpu \
  --rules 2 --hidden 16 --max-pattern 2 --max-replacement 2 \
  --max-stars 1 --chunk-len 16 --capacity-factor 2 \
  --batch-chunks 2 --levels 2 --valid 2 --test 2 \
  --checkpoint-every 1 --steps 2 --eval-every 2 --out smoke
```

実際のデータ `github-code-dataset.txt` を使う場合（MPS例）：

```bash
python train.py --mode scalable --device mps --corpus github-code-dataset.txt \
  --rules 1500 --hidden 2048 --max-pattern 4 --max-replacement 4 \
  --max-stars 3 --chunk-len 128 --capacity-factor 2 \
  --batch-chunks 2 --levels 4 --microbatch 0 --checkpoint-every 8 \
  --steps 2000 --eval-every 100 --out lens_run
```

**注意**：このMPSの1500規則は起動形式を示すもので、速度・VRAMを保証しません。
深い直列構造と可変長捕捉の仕事量は非常に大きいので、
まず `--rules 2 --chunk-len 16` 程度から環境別に計測し、徐々に増やしてください。
`--microbatch 0` はバッチ全体を一括演算し、正の値で明示的に分割します。
`--capacity-factor` を増やすと伸長を許せる代わりに必要メモリが増えます。

**学習データ形式**：`===SPLIT===` 行で分離したUTF-8 / バイト列文書。別々の文書をtrain/validation/testに分けます。
ノイズは同じ文書に対しランダムなバイト置換を入れて生成し、残存率を回帰します。
これは**百科事典の意味知識の直接学習ではありません**。目的関数を変更しなければ、規則を増やしても事実知識の取得は保証されません。

出力：`last.pt`（再開可能モデル/optimizer）、`rules.json`（学習後の旧Replace離散規則）、
`train.jsonl`, `eval.jsonl`, `results.json`, `config.json`, `split.json`。
通常の予測は旧Replaceの離散推論を使用します。

```bash
python -m lens_gpu.predict lens_run/last.pt article.txt --device mps
```

## 厳密な小サポート直列学習（exact-small）

```bash
python train.py --mode exact-small --demo --device cpu --steps 10 --out tiny_exact
# カスタムデータ・ルールの支持集合で学習
python train.py --mode exact-small \
  --dataset examples/dataset.json --rules-json examples/rules.json \
  --steps 10 --out tiny_exact
```

このモードは前段の曖昧な捕捉結果の**文字列全体の同時分布**と、規則ごとの**発火履歴**を保持します。
非線形のMLPには `head(E[features])` ではなく正しい `E[head(features)]` を使います。
反面、全256文字・長文・1500ルールに対する計算量上の解法ではありません。
このモードは `scalable` の大規模モデルとチェックポイントパラメータを共有しません。

## テスト

```bash
python tests/test_serial_end_to_end.py
python tests/test_exact.py
python tests/test_vertex_within_bound.py
python -m lens_gpu.test_old_wildcard
python -m lens_gpu.test_vertex_random
python -m lens_gpu.test_priority
```

旧版の数値確認用コード：`reference_exact/discrete_replace.py`, `reference_exact/replace_exact.py`,
`reference_exact/replace_fixed.nim`。

CUDA/MPS実機はこの配布作成環境では未検証です。`scalable` は2階微分可能ですが、
全ての操作で数値条件に依らず安定したHessianが得られると保証するものではありません。
