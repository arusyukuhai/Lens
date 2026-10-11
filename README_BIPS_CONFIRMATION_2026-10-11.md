# Lens: BIPS two-stage fresh-row confirmation (2026-10-11)

## Motivation from user 199-generation run

The latest training log has best `Pearson=0.976212` and best reconstruction `inferBestEver=0.282`. It recorded 2,240 CEGAR concrete evaluations and 558 internal validation passes; this does **not** prove the BIPS path family generalizes. In several generations `valPearsonGain` approached 2.0 on tiny held-out-source mini-probes (sometimes only 8 or 12 rows). Across 199 generations the inference recovery score fluctuates. The best-ever score is not a substitute for an independent evaluation corpus.

## New behavior

1. CEGAR still mines and tests 2–4-rule paths using exact Native execution, first-sweep source repair, small cross-original screen, capped adaptive checks, and optional per-stage interventions.
2. **Second, finalist-only confirmation.** The `--bips-confirm-budget` highest-ranked screened macros are run on additional, **unused by that candidate** trajectory rows. Readout fit uses rows from its original source; Pearson and MSE are computed on other-original rows, without refitting on validation. The population's separate final rolling source remains untouched. The confirmation pass uses max `--bips-confirm-probes` rows, evenly spread through the remaining corruption trajectory.
3. If the fresh-row Pearson gain is not positive, or relative MSE reduction is below `--bips-confirm-min-mse-ratio`, the candidate is rejected from the *validated* macro lane. Unconfirmed candidates outside the confirmation budget are no longer mislabeled as validated winners. The pre-existing **explicit exploration** lane is not blocked; as before, its child gets an ordinary next-generation evaluation.
4. `--bips-cegar-require-validation-firing` (optional, default off) can require the downstream rule to fire in the other source. This conservative gate can reject useful source-specific or later-sweep chains; it is NOT necessary for all tasks and may reduce discovery throughput.
5. Confirmation scores cap the influence of very large micro-Pearson gains on finalist ranking. The basic `--readout-mode linear-pearson` is unchanged; the inner `--inference-mode ga`, CPU Native, LUT, legacy -2/-3 checkpoint execution, HOF and six directed mutation operators are unchanged.

## Command

```sh
python main.py \
  --backend cpu-native --learning-mode guided \
  --readout-mode linear-pearson --inference-mode ga \
  --bips-share .35 --bips-depth 4 \
  --bips-cegar-budget 12 --bips-cegar-probes 16 \
  --bips-cegar-adaptive-probes --bips-cegar-adaptive-max-probes 24 \
  --bips-confirm-budget 2 --bips-confirm-probes 64 \
  --bips-confirm-min-mse-ratio .01 \
  --bips-causal-budget 2 \
  --population 96 --elites 40 --rules 1500 --hof-size 4096
```

Switch off fresh-row confirmation to replicate the previous causal/adaptive path:

```sh
--bips-confirm-budget 0
```

If your corpus contains too few *training originals*, `confirmInsufficient` will rise, and screening-only validation will not gain a misleading label. If this makes macro learning too restrictive, either provide more training originals/trajectories or explicitly disable confirmation.

## Logging

`bips-cegar ... confirm=X/Y confirmReject=N confirmInsufficient=N confirmRows=N confirmPearsonGain=G confirmMSEGain=G noValFire=N`

CSV: `bips_confirm_checked`, `bips_confirm_accepted`, `bips_confirm_rejected`, `bips_confirm_insufficient`, `bips_confirm_probe_rows`, `bips_confirm_best_pearson_gain`, `bips_confirm_best_mse_gain`, `bips_confirm_best_mse_gain_ratio`, `bips_cegar_validation_no_fires`.

## Five-arm comparison

`compare_bips_modes.py` adds `bips_confirmed` as fifth mode; the previous four arms explicitly disable confirmation, so their semantics do not silently change. Use a truly never-trained corpus with `--audit-corpus` for the frozen-readout Pearson/MSE audit, multiple RNG seeds, and matched wall-clock budgets.

## What is **not** solved

- These are **different noise rows**, not independent original texts beyond the earlier cross-original split. Both batches are taken from the same finite set of training originals. Repeated selection across generations may still overfit them. Do not claim real heldout generalization without `audit_generalization.py` on an untouched corpus.
- Confirmation is a gate, not a new generator of genuinely cooperative rules. It can improve precision of admissions at the expense of recall and CPU time. It does **not** guarantee improved Pearson or byte restoration.
- Causal leave-one-stage-out still measures interventions against parent row contents and is sensitive to probe context. The inference GA budget clamp persists (`infPlan` is what actually ran).
- Small synthetic smoke tests can have zero confirmed macro wins. This is expected, not a proof that BIPS works poorly on real data.
