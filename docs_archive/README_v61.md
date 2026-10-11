# v61 — Steady-state diff3 Differential Evolution

外側の探索を世代交代型 GA / NSGA-II から、**population=450 を固定した steady-state differential evolution** に変更した版です。旧アルゴリズム本体は `evolve_generational_legacy()` として残してありますが、CLI の既定実行は v61 DE です。

1 sweep では population の全 target をランダム順で1回ずつ処理します。各 step で target `a` 以外から `b,c` を選び、各 Rule を1遺伝子として `b -> c` の構造差分を diff3 で `a` に適用して trial `a'` を作ります。

```text
population (450)
   ↓ target a を1体選択
random distinct b, c
   ↓ categorical diff3: merge-base=b, donor=c, target=a
trial a'
   ↓ 同じ freshly-rotated evaluation set で a と a' をペア評価
Spearman(a), inference(a)
Spearman(a'), inference(a')
   ↓
a' が両方とも strict に上昇した場合だけ即時 a を置換
```

diff3 conflict は token 単位で無理に混ぜません。同じ Rule に `a` 側の独立編集と `b->c` 側の編集が衝突した場合、その **Rule 全体を a / b / c のいずれかから直接選択**します。embedding は permutation/injective 制約があるため位置ごとの diff3 を掛けず、従来の embedding mutation/crossover 経路だけで変更します。

diff3 trial の後には独立確率で、通常の2点 crossover、既存 structural mutation、単純な Rule-string cut/paste を追加できます。diff3 が完全な no-op になった場合だけは、strict 改善判定で絶対に勝てない無駄評価を避けるため local mutation を1回強制します。

既定の DE 関連設定は以下です。

```text
--population 450
--de-case-refresh 3
--de-inference-refresh 3
--de-crossover-rate 0.30
--de-mutation-rate 0.35
--de-splice-rate 0.20
--de-splice-max-rows 8
--de-audit-every 1
```

`--de-case-refresh 3` により **target 個体を1体処理するたびに outer evaluation case を異なる3 slotまとめて交換**します。`a` と `a'` は交換後のまったく同じ case set で必ず再評価されるため、rolling data の難易度差で trial だけが有利/不利になることはありません。inference set も既定では同様に3 case更新します。

survivor 条件は通常の Pareto dominance より厳しく、inference 有効時は

```text
Spearman(a') > Spearman(a)
AND
inference_accuracy(a') > inference_accuracy(a)
```

の **両方を strict に満たした場合だけ**即時置換します。片方が同値でも不採用です。`--no-inference` 時だけは Spearman strict improvement 単目的へ退化します。

`--de-audit-every` の common-snapshot audit はログ・plot・best checkpoint を450体間で比較可能にするためだけの診断です。survivor 判定には一切使いません。実際の探索選択は全て target ごとの `a vs a'` ペア比較で完結します。

また v61 で、既存の corpus loader が `max_chunk` より長い chunk を crop する前に除外していた条件順序のバグも修正しました。

---

詳細な v60 inference/readout の背景は `README.md` を参照してください。
