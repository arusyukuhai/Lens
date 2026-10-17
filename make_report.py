"""Rebuild the report and optional scientific training plot from saved measurements."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
r=json.loads((root/'demo/results.json').read_text())
b=json.loads((root/'serial1500_cpu.json').read_text())
rows=[]
for label,key in [('連続規則','test_soft'),('離散規則・MLPそのまま','test_hard'),('離散規則・MLPのみ20更新','test_hard_after_head_finetune')]:
    v=r[key];rows.append(f"| {label} | {v['mse']:.6f} | {v['mae']:.6f} | {v['pearson']:.6f} |")
text='''# 実測結果

## 機能検証

`test_lens.py` の6テスト群に合格しました。

- 非重複走査を逐次DPと比較し、1階・2階の数値微分も一致。
- 全成分が未確定の入力分布・連続長についてgradcheckとgradgradcheckに合格。
- 規則パラメータについて2階微分が有限。
- 離散規則と純Python整数実装の発火特徴が65入力で一致。
- 前段が生成した記号を後段が使うケースも一致。
- バッチと個別処理、チェックポイント有無の勾配が一致。
- ノイズの実変更数と教師が一致。破損位置は拡散につれて単調に増える。
- 正則化係数が上下動し、開始0・終了λmaxになる。

## 1500段・指定MLPの動作確認

CPU1スレッド、入力16バイト、バッチ2、キャプチャ上限1、
**1500段直列 → 1500特徴 → SwiGLU 2048 → SwiGLU 2048 → 1**。
'''
text+=f"\nパラメータ数 {b['parameters']:,}。warmup後のforward+backwardは **{b['seconds'][0]:.3f}秒**（1回測定）。\n"
text+=f"最初の規則の外側logits勾配ノルムは {b['first_rule_grad_norm']:.3e}、最後は {b['last_rule_grad_norm']:.3e}。\n"
text+='''逆伝播の経路は存在しますが非常に小さい値です。1500段で良好に学習できた証拠ではありません。
この計測は短い入力に限定し、2000バイト×1500段の実用性能を示していません。
CUDA/MPS実機はありません。GPU速度の推測値を掲載していません。

## 小規模な学習試験

人工的なコード風文書64件、各96バイト。重複排除後に学習48・検証8・テスト8文書。
テスト32行は8文書×4拡散段階であり、32独立文書ではありません。
16段直列、MLP隠れ幅128、キャプチャ上限2、200更新。
バッチ4文書×4段階=16行、microbatch4。教師は残存率のみ。
単一seed。実際の利用コーパス・1500規則での学習成績とは分けて扱ってください。

| テスト方式 | MSE | MAE | Pearson |
|---|---:|---:|---:|
'''
text+='\n'.join(rows)+'\n'
text+=f"\nテスト集合の平均値を常に返す定数のMSEは {r['test_soft']['constant_mean_mse']:.6f}。この平均値は比較指標だけに使い、モデルへ与えていません。\n"
text+=f"連続学習と途中検証に {r['training_seconds']:.2f}秒（CPU）。最終テスト・追加20更新はこの時間に含みません。\n"
text+='''
離散化直後にはMSEが悪化しました。規則を固定してMLPだけ調整すると改善しましたが、
連続版と完全に同じ挙動へ収束したわけではありません。
追加学習の有無・更新数は事前に設定し、テスト値で選択していません。

最初の40更新だけの短い試験（`smoke/`）も保存しています。
そのテストMSEは約0.280で定数基準より悪く、短い試験を成功例として扱っていません。
この後の200更新試験は設定を変えた追加の開発実験です。同じ分割・seedを使用したので、
テスト値は独立した最終ベンチマークではなく開発中の診断結果です。
正則化なしとの比較は行っておらず、改善を振動正則化の効果と断定できません。

## 再現コマンド

```bash
python test_lens.py
python benchmark.py --device cpu --rules 1500 --hidden 2048 --length 16 --batch 2 --max-capture 1 --checkpoint-every 25 --iterations 1 --threads 1 --output serial1500_cpu.json
python train.py --synthetic --out demo --device cpu --rules 16 --hidden 128 --max-capture 2 --chunk-len 96 --steps 200 --batch-chunks 4 --levels 4 --microbatch 4 --checkpoint-every 4 --eval-every 100 --log-every 20 --threads 1 --lr .001 --hard-finetune 20
```

`demo/last.pt` は連続学習、`demo/hard_finetuned.pt` は離散規則用MLP調整後の重みです。
`training.png` は保存ログから作成。プロット再生成だけはmatplotlibも必要です。
'''
(root/'RESULTS.md').write_text(text)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from train import anneal
    history=r['history']; steps=r['args']['steps'];args=r['args']
    fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True,layout='constrained')
    axes[0].plot([x['step'] for x in history],[x['loss'] for x in history],marker='o',label='Training MSE (changing batches)')
    val=[json.loads(line) for line in (root/'demo/eval.jsonl').read_text().splitlines()]
    axes[0].plot([x['step'] for x in val],[x['validation_soft']['mse'] for x in val],marker='s',label='Fixed validation MSE')
    axes[0].set_ylabel('MSE');axes[0].set_yscale('log');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].plot(range(steps),[anneal(i,steps,args['reg_max'],args['reg_cycles'],args['reg_warmup']) for i in range(steps)],color='#9b51e0',label='Regularization coefficient')
    axes[1].set_ylabel('lambda');axes[1].set_xlabel('Optimizer update');axes[1].grid(alpha=.2)
    fig.suptitle('Serial Lens: 16 rules, synthetic corpus (CPU)')
    fig.savefig(root/'training.png',dpi=160);plt.close(fig)
except ImportError: pass
