# Lens v10 — tqdm進捗表示を追加（v9高速化版ベース）

今回の変更は **進捗の可視化のみ** です。自己回帰、450ルール、256トークン、`===SPLIT===` 分割、1500バイト以上のチャンク除外、可逆LUT、遺伝的アルゴリズム、モデル/チェックポイント形式はv9と同じです。

## セットアップ

ZIP内の `main.py`, `gpu_replace_persistent.py`, `native_cpu.py`, `replacer_native.cpp` を**すべて**同じフォルダへコピーしてください。`replacer_native.cpp` を入れ替えると、CPUでは初回評価時に変更を検出してC++を自動再コンパイルします。

```bash
python -m pip install tqdm
python main.py --backend mps --local-corpus github-code.txt \
    --population 450 --rules 450 --cases 8 --max-chunk 1500
```

CPUネイティブ並列評価:

```bash
python main.py --backend cpu --cpu-workers 0 \
    --local-corpus github-code.txt --population 450 --rules 450 --cases 8
```

進捗表示が不要なら `--no-tqdm` を指定してください（この場合は `tqdm` のインストールは不要です）。

## 表示内容

- **初期集団を生成**: 初回起動時に何個体を生成したか。
- **Lens 学習 gen=N**: 世代数、世代/秒、直近のAccuracy、評価時間、世代全体の推定残り時間。
- **gen=N 評価/mps / 評価/cpu**: 今世代に要求された個体のうち評価が終わった数、個体/秒、残り時間。
- **gen=N 次世代作成**: 次の世代の子個体生成進捗。
- 通常の `gen=... acc=... mean=... seconds=...` ログも維持します。

CPUでは **C++側のスレッドが1個体の全文評価を終えたごとに** tqdmへ通知します。既存のC++全個体一括並列処理を細かな小バッチへ分割しません。正確にキャッシュから再利用した個体も進捗に加算します。

MPSでは **1バッチの計算結果をGPUから読み戻した後** tqdmを更新します。GPUカーネルの実行途中までは計測できないため、バッチが長時間かかる場合は進捗がその間止まります。さらに細かな表示にしたい場合は、`--mps-batch 16` や `--mps-batch 8` を試せますが、実行速度が遅くなる可能性があります。進捗表示自体のためにGPUを余分に同期させる処理は入れていません。

速度が異常に遅い場合は、数世代の `seconds=` を比較し、`--backend mps` と `--backend cpu` の実測値を比べてください。tqdmは処理速度を表示するだけで学習を自動高速化するものではありません。

## テスト

```bash
python -m unittest -v test_progress test_optimized test_autoregressive test_fastfix test_corpus_split
```

38件の回帰テストを実行: C++並列コールバック、同一ゲノムのキャッシュ計数、Pythonフォールバック、模擬MPSバッチ、tqdmの有効/無効によるモデル一致、旧来の自己回帰評価テストなど。

Apple MPS実機でのシェーダーコンパイル・実時間測定は未検証です。
