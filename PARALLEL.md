# GPU/MPS バッチ並列化・高速化（256バイト固定版）

この版は以前の `Lens_Serial_256_Fixed_GPU_MPS.zip` を直接改善したものです。
規則は必ず **規則0の出力→規則1→…→規則1499** の順番で作用します。
規則ごとの入力依存性は壊せないため、1500規則を同時に適用する実装にはしていません。

## 最大の改善: ミニバッチの暗黙の直列化を解除

旧版の既定 `--microbatch 2` では、`--batch-chunks 8 --levels 4` を指定しても
32件を一度にGPUへ送らず、2件×16回に分割して直列 forward/backward していました。
この版の既定 **`--microbatch 0` は全行一括**です。
メモリが足りなければ `--microbatch 2`, `4`, `8` 等を明示して分割します。
`--accumulate` は勾配蓄積であり、指定回数だけ逐次実行します。

> 計算時間の比較では、1ステップの秒数とともに **行/秒 = (batch-chunks × levels × accumulate)/秒** を見てください。
> バッチを増やすと1ステップの秒数が伸びるのは普通ですが、行/秒が伸びれば有効な高速化です。

## 規則内部の並列化

1. **1500規則分のsoftmaxをまとめて実行。** 左・右・中央・置換先のパラメータを規則軸にstackし、規則ごとのCUDA/MPS softmax起動を大幅に削減。`state_dict` は前版と互換です。
2. **ワイルドカードの全長候補を同時評価。** 長さごとのPythonループを除去し、`unfold` のviewと `cumprod` に変更。候補間の最初の一致の条件は同一です。
3. **キャプチャ保持を並列計算。** Pythonの長さごとの `sum`/`shift` の反復をやめ、逆向き `cumsum` とまとめた `gather` に変更。
4. **連続座標パッキングを1回のscatter_addで実行。** 元の2回の逐次 `scatter_add` を、一括で2×Lのソースへ散布する方式に変更。メモリ転送とkernel-launch回数のトレードオフあり。
5. **非重複マッチ選択は並列prefix scanのまま。** 既存のバランス木により位置軸は O(log L) 依存深さ。バッチと文字チャネルはすべてテンソル演算です。
6. **AdamW更新をCUDA fused実装に。** CUDAでは既定で`fused=True`を使い、多数の小さい規則パラメータの更新をまとめます。MPSは安全のため既定ではscalar経路ですが、PyTorch版に応じて`--optimizer-kernels foreach`を試せます。旧経路への切替えは`--optimizer-kernels scalar`。
7. **オプション `--compile-rule`（CUDAのみ）。** 1個の共通 `serial_step` を `torch.compile` でfusionし、1500規則が同じcompiled関数を再利用します。MPSでは使用せず通常のGPUテンソル計算を使います。PyTorch/CUDA依存の実験機能であり、初回コンパイルは遅くなります。2階微分が必要ならeagerを使用してください。

## 実行例

CUDA（まずは短いシーケンスでメモリ確認）:

```bash
python train.py --corpus github-code-dataset.txt --device cuda --out run_cuda \
 --rules 1500 --hidden 2048 --chunk-len 512 \
 --batch-chunks 4 --levels 4 --microbatch 0 --checkpoint-every 25 \
 --steps 2000
```

CUDA + compiled kernel（GPUとPyTorchの互換性を確認）:

```bash
python train.py --corpus github-code-dataset.txt --device cuda --out run_cuda_compile \
 --rules 1500 --chunk-len 512 --batch-chunks 4 --levels 4 \
 --microbatch 0 --compile-rule --checkpoint-every 25
```

MPS（GPUメモリ不足なら `--microbatch 4` → `2` に調整）:

```bash
python train.py --corpus github-code-dataset.txt --device mps --out run_mps \
 --rules 1500 --chunk-len 512 --batch-chunks 4 --levels 4 \
 --microbatch 0 --checkpoint-every 25
```

## バッチサイズ別ベンチマーク

```bash
python benchmark_scaling.py --device mps --rules 16 --hidden 64 \
  --length 256 --batches 1,2,4,8,16 --iterations 5
```

`baseline` は元の256固定モデル、`optimized` は今回版。同じ設定でforward+backwardを実測し、
`seconds_median` と `samples_per_second` を記録します。CPU環境では16サンプルで改善を観測していますが、
バッチが小さい場合は遅くなるケースもあり、**GPU/MPS実機での速度向上は保証できません**。
CUDAの速度・メモリは `--compile-rule` の有無も比較してください。

## 精度、メモリ、互換性

- 256バイト固定、ワイルドカードの最初の右区切り、左から非重複置換、各規則の発火率÷元長、
  元のC²近似の連続パッキング、SwiGLUを維持。
- `model_baseline.py` を比較用として収録。既存モデルとforward/gradientの一致をテスト済み。
- 2階微分は通常のPyTorch eagerモードで検証（CUDA compile経路の2階微分は未保証）。
- `checkpoint_every` は逆伝播時の再計算区間サイズです。GPUメモリはおおむね
  バッチサイズ×文字長×256×floatサイズに比例して増加し、checkpoint境界と区間内の活性にも比例します。
  全行一括がOOMになる場合は `--microbatch` を明示して分割する必要があります。
- モデルのパラメータ名・形状と `format_version=2` は前版と同一。旧学習の再開では
  **`--microbatch` 等の学習設定を以前と一致させること**（新既定は0、旧既定は2）。
- この環境にはCUDA/MPS GPUがないため、ハードウェア固有の実行は未検証です。
