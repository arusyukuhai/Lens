import json
from pathlib import Path
root=Path(__file__).resolve().parent
r=json.loads((root/'scan_upgrade.json').read_text())
s='''# 意味論を保つ走査高速化

旧実装の倍々合成を、区間行列の平衡木＋状態ベクトルの下向き伝搬へ置き換えました。
行列の合成順は常に「右区間 @ 左区間」です。各位置へ入る状態は、直前までの
遷移を初期状態e0に作用させたものなので、旧実装と同一の数式を計算します。
2の冪でない長さは単位行列で埋め、結果を元の長さに切ります。

- 走査の総演算量: O(B L S³ log L) → O(B L S³)。
- 依存する計算の深さ: O(log L)を維持。
- 木に必要な行列数: O(L)。下りは行列積ではなく行列×ベクトル。
- 1500規則の直列順、確率入力、既存の連続近似、正則化係数は変更なし。
- 古い重みをそのまま読み込めます。浮動小数点の加算順序は変わるのでビット一致ではありません。

## 検証

既存6テスト群に合格。加えて長さ1/2/3/7/16/31/129、状態数1/3/6の21条件で、
旧実装と出力・1階勾配・2階微分由来のベクトルを比較しました。
'''
s+='最大絶対差（出力/1階/2階）: '+str([max(c['errors_output_grad_hvp'][i] for c in r['checks']) for i in range(3)])+'。\n'
s+='''
## 速度

CPU1スレッド。同じプロセス・同じ入力と重みで比較。デバイス同期を挟み、warmup除外。
走査単体は5回の中央値、モデル全体は2回の中央値。すべてforward+backwardです。

| 走査入力（B,L,S） | 旧ms | 新ms | 倍率 |
|---|---:|---:|---:|
'''
for x in r['scan_forward_backward']:
 s+=f"| {x['batch']}, {x['length']}, {x['spans']} | {1000*x['seconds']['old']:.3f} | {1000*x['seconds']['new']:.3f} | {x['speedup']:.2f}× |\n"
s+='\n| モデル（規則数/入力長/隠れ幅） | 旧秒 | 新秒 | 倍率 |\n|---|---:|---:|---:|\n'
for x in r['model_forward_backward']:
 s+=f"| {x['rules']}/{x['length']}/{x['hidden']} | {x['seconds']['old']:.3f} | {x['seconds']['new']:.3f} | {x['speedup']:.2f}× |\n"
s+='''
1500規則の測定は16バイトの短い入力です。長文1500段の速度を示すものではありません。
CUDA/MPSはこの環境では利用できず未計測。GPUでも同じPyTorchテンソル経路を使用します。
走査単体の倍率をモデル全体の倍率として解釈しないでください。
文字の確率チャネル増加に伴うN²項は今回の変更では解消していません。

再現: `python verify_scan_upgrade.py --device cpu --output scan_upgrade.json`
GPU/MPS実機では `--device cuda` / `--device mps` に変更。
旧走査は `scan_reference.py` に比較専用で保持し、通常の学習は新走査を使います。
今回の作業では再学習はせず、同一重みにおける出力・勾配一致を検証しました。
'''
(root/'SCAN_UPGRADE.md').write_text(s)
print(s)
