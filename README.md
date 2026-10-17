# Lens v8 — Sparse Capture Transducer (identity / reverse / sort / inverse_sort)

**目的:** Lens v7 の旧Nim式・無制限ワイルドカード捕捉を保ち、置換先に捕捉ごとの3種類の操作を追加。以前の密な入力位置×出力位置の対応行列を取り除き、メモリを幾何候補と系列長に対し概ね線形に抑える。

## 置換先の文法

置換元の捕捉は従来どおり `*1, *2, ..., *n`、文字は256バイト。置換先では各捕捉について、`identity(*i)`, `reverse(*i)`, `sort(*i)`, `inverse_sort(*i)` の **4種類**から選べる。追加は `n*3`、合計は `256+4n` 種類のRHSトークン。

**トークンID (学習内部):** `0..255` は文字。`256+4*i+op` は捕捉 `i` (0-based) に対する `op=0` 通常 / `1` reverse / `2` 昇順sort / `3` 降順inverse_sort。

**離散エクスポートのopcodes:** `-(1 + 16*op + capture_id)`。例: `-1` は `*1`、`-17` は `reverse(*1)`、`-33` は `sort(*1)`、`-49` は `inverse_sort(*1)`。捕捉 `*2` はそれぞれ `-2,-18,-34,-50`。最大15捕捉 (置換元の純STARは無効) に対応。旧版の `-1..-15` は互換。

`A*1B*2C -> reverse(*2) sort(*1)` の例: 入力 `AxyBdcC` → `cdxy` （後半の `xy` は昇順にソートされた結果）。捕捉は先頭側の一致から、各区切り文字列の**最初の出現位置**まで。末尾の `*` は残り全部を捕捉。非重複で左から順に置換する。

`inverse_sort` は **降順ソート**であって逆置換ではない。UTF-8のバイト値に対するソートなので日本語として有効な符号化を保証しない。

## 数学的実装の分担

- **離散:** `lens_gpu/old_replace.py`。旧Nim式ワイルドカードに、逆順 / 昇順 / 降順の操作を追加。ソートはバイト値の比較、捕捉長制限なし。hard推論には整数CPUインタプリタを使用。
- **C²近似学習:** `lens_gpu/unbounded_dp.py`。First-hit DPで長さ候補を列挙せずに区切り探索。捕捉は再利用/複製/入れ替え可能。`identity`・`reverse` は捕捉元をC²二点補間で散布する。`sort`・`inverse_sort` は捕捉の確率的**期待ヒストグラム**を累積し、五次smoothstepを用いてソート後のバイト分布を構成する。離散One-Hot極限ではソートと一致する。
- **厳密確率的pushforward:** 小さな静的サポートに限り `ExactSerialPushforward(..., capture_transforms=True)` が、変換済み捕捉を含む全文字列の同時確率を厳密に伝播。元の `capture_transforms=False` は旧API互換。**通常のGPU学習版は厳密確率分布ではなく平均場近似**で、捕捉の複製における相関を完全には保持しない。

## メモリ改善

従来のsoft出力は `[batch,input_position,output_position]` の密なコピー係数行列を作り、最後に256チャンネルを掛けていた。新版では文字分布を2点scatterで直接出力へ加算。**入力×出力の密行列を生成しない**。

- 捕捉ごとの累積ヒストグラムは `torch.cumsum` で `O(B*L*256)`。
- 変換される参照もジオメトリ軸 `H` と元位置軸の二方向で小さなブロックに分割し、展開テンソルを目標約400万スカラーに制限。これは**目安**で、中間テンソルと勾配バッファが別途存在する。
- 貪欲非重複選抜: `L>=96` なら密な `[B,L+1,L+1]` 三角行列に代え、直接散布する `stream_select` を使う。32ステップ単位で再計算チェックポイント。密演算との差と2階微分を検証済み。`L<96` は高速な密三角解法を維持するが、**これは元の入力×出力コピー行列ではなく、非重複の遷移行列**。

**重要:** メモリの改善と計算時間の高速化は別。完全にソフトな規則で全部の置換元・捕捉候補を扱うため、`O(L^2)` の仕事量が残る。特に `max_pattern`, `max_stars`, `max_replacement` が大きいとCPU/MPS/CUDAで非常に遅くなる場合がある。新しいGPU専用融合カーネルを導入したわけではない。1500規則・2000バイトでの実行を保証しない。実機CUDA/MPSの速度/メモリ未測定。

旧版同様、捕捉長自体には上限はないが、系列全体のテンソル容量 (`--capacity-factor`) は有限で、超過時はエラー。

## 実行例

```bash
pip install -r requirements.txt
python -m lens_gpu.train --synthetic --device cpu --rules 2 --hidden 8 \
  --max-pattern 2 --max-replacement 2 --max-stars 1 \
  --chunk-len 12 --capacity-factor 2 --batch-chunks 1 --levels 2 \
  --steps 2 --valid 2 --test 2 --checkpoint-every 1 \
  --out small_test
```

MPS/CUDAでは最初は `--rules 2`・`--chunk-len 16`・`--max-pattern 2` で検証し、メモリと時間を測りながら増やすこと。

```bash
# v7の重みのみ移行（optimizerは移行しない）
python -m lens_gpu.upgrade_weights v7_checkpoint.pt v8_initial_weights.pt

# tests
PYTHONPATH=. python tests/test_capture_transforms.py
PYTHONPATH=. python tests/test_identity_equivalence.py
python -m lens_gpu.test_old_wildcard
python tests/test_exact.py
```

## 互換性

- 新規モデルの `format_version=8`。v7のoptimizer付き再開は不可。v7からの `upgrade_weights` は新トークンを抑えた**重みだけ**の移行で、再学習が必要。
- old-Nim意味論 (first delimiter, trailing STAR, no-op pure STAR) は保持。変換の新たなオペコードはv8の拡張であり、オリジナル `reference_exact/replace_fixed.nim` 自体には実装されていない。
- v7の学習ヘッド・256バイト・直列規則・checkpoint再計算は継承。

## 数値検証

CPU (PyTorch) で、離散変換、各種捕捉、旧版のsoft identity-only計算との出力と入力勾配の一致、ストリーミング非重複DPと密解法との一致と2階微分、soft変換演算の1階/2階数値微分、厳密pushforwardの変換、短い学習・評価をテスト。CUDA/MPS実機は未検証。
