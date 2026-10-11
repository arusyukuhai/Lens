# Lens / Replacer — Causal BIPS + Adaptive CEGAR (2026-10-11)

This is the complete-source successor to `Lens_Pearson_BIPS_CEGAR_Efficient_20261011_full.zip`. It keeps ordinary Linear Regression + Pearson, internal GA string-recovery learning, Native CPU executor, embedding LUT, HOF, and backward compatibility with existing checkpoints. No new hidden-memory capture opcodes are generated.

## Changes

**1. Fix strongest-candidate bookkeeping.** The old code retrieved a candidate with `best_by_path.get(p.path_key)` but wrote it using `(p.anchors,p.origin_case)` as the key. After stage-aware anchor repairs, the lookup and write keys differed, causing worse later placements to replace the best child. Now the canonical source-path identity is used uniformly, including the exploration lane and deduplication. Existing regression tests were updated where they incorrectly presumed the chosen placement must be the front window; score determines placement.

**2. Counterfactual full-program macro ablations.** After a BIPS candidate passes original-disjoint micro-validation, optionally intervene on the top `--bips-causal-budget` macros (default 2). For every rule in each 2–4-rule candidate, restore only that row of the original parent genome and rerun the entire Native Replacer. Refit a bounded OLS readout on the *same candidate-construction original* and compare heldout-original MSE with the full chain. Positive leave-one-stage-out MSE delta means that restoring/removing this stage worsens the macro's validated performance; negative means the stage may be harmful; near-zero indicates potential redundancy. A conservative bounded score modifier prefers paths supported by complementary stages. **This is an intervention-based contribution diagnostic, not proof of mechanistic/physical causality or true generalization.** An original row restored may itself match text, so it is specifically a *reversal to the parent* rather than a no-op substitution.

**3. Budgeted, adaptive Native probes.** A tiny probe has high variance and can yield implausibly large `valPearsonGain` near +2. If the gain is suspiciously large (>0.65) or barely above the minimum acceptance threshold (less than 0.10 above), and MSE gate is positive when near threshold, optionally re-evaluate the **unchanged** child on up to `--bips-cegar-adaptive-max-probes` rows (default 24), still train/validation-disjoint by original. This extra exact evaluation counts against the existing `--bips-cegar-budget`, with at most `--bips-cegar-adaptive-expansions` (default 2) expansions per generation. A weak expanded result can reject an initially promising candidate. No withheld original from the population-level audit is used in synthesis or selection.

**4. Consistent logging.** BIPS console summary includes `adaptive`, `causal`, `allStages`, `redundant`. History CSV adds:

- `bips_cegar_adaptive_expansions`, `bips_cegar_adaptive_extra_rows`
- `bips_causal_macros_checked`, `bips_causal_stage_interventions`
- `bips_causal_all_stages_helpful`, `bips_causal_redundant_stages`, `bips_causal_best_min_stage_gain`

**5. Four-arm comparison utility.** `compare_bips_modes.py` runs `guided_only`, `bips_unvalidated`, `bips_fixed_keys` (no causal/adaptive), and `bips_causal_adaptive`. Fix population > elites. Use **multiple seeds** and match wall-clock budgets as well as generation counts before making performance claims. Operator paths can consume RNG differently despite identical initial seeds. `--audit-corpus` accepts a truly separate, never-trained corpus and runs a frozen-readout Pearson/MSE audit after the arms. The existing `audit_generalization.py` supports CPU Native and defaults to Pearson (optionally Spearman).

## Suggested command

```bash
python main.py \
  --backend cpu-native --learning-mode guided \
  --readout-mode linear-pearson --inference-mode ga \
  --bips-share 0.35 --bips-depth 4 \
  --bips-crossfit --bips-cegar-source-gate \
  --bips-cegar-budget 12 --bips-cegar-probes 16 \
  --bips-cegar-adaptive-probes \
  --bips-cegar-adaptive-max-probes 24 \
  --bips-cegar-adaptive-expansions 2 \
  --bips-causal-budget 2 \
  --population 96 --elites 40 --rules 1500 --hof-size 4096
```

Disable the new scoring selectively:

```bash
--bips-causal-budget 0 --no-bips-cegar-adaptive-probes
```

Comparisons across runs:

```bash
python compare_bips_modes.py \
  --train-corpus /path/to/train.txt \
  --audit-corpus /path/to/NEVER-TRAINED.txt \
  --output comparative_runs --generations 100 \
  --population 96 --elites 40 --rules 1500 --seed 17
```

## Validation and boundaries

- `python -m pytest -q`: **153 passed, 4 skipped, 10 subtests passed** (no MPS here).
- `python main.py --self-test`: **PASS**.
- 4 generations of training on a synthetic corpus with 32 genomes, 128 rules, 3 cases, 14 trajectories/case, Native+inner reconstruction GA, elites=8, and BIPS+causal interventions completed. Logs show one macro intervention check in generation 0, and that BIPS children were added to next-generation candidates. This is **not** proof of improved real-corpus performance.
- Four-arm 3-generation synthetic comparison completed with all exit statuses 0. The new mode attained a higher best Pearson than the other three arms in this **single, small run**, but no claim of reliable speed or accuracy superiority follows.
- Frozen-readout, fully separate corpus Pearson audit executed successfully (CPU Native), retaining all models' fixed weights.

### Limitations

- Tiny in-training cross-original validation may still have high variance. Repeated adaptive selection can itself overfit the candidate-validation original. Never use it as the final generalization metric; use truly fresh corpus in `audit_generalization.py`.
- Per-stage causal ablation is *local* and budgeted. It does not exhaustively test every subset of a chain or estimate Shapley values. A stage that seems redundant under a bounded OLS probe can be necessary elsewhere or in later sweeps.
- `--bips-cegar-budget` caps concrete candidate checks *including adaptive expansions*, but causal macro interventions are separate additional Native calls, bounded by `--bips-causal-budget * path_depth`. These increase per-generation work.
- Group-aware 32-feature readout selection is **not** implemented here: rule/macro provenance needs to persist safely through ordinary mutation, crossover and checkpoint serialization. The current max-feature limit is unchanged; do not treat `active=32/1500` as the number of rules which fired.
- First-sweep source feasibility gates can discard later-sweep-only chains. Disable with `--no-bips-cegar-source-gate` when testing completeness.
- Internal reconstruction GA retains auto-budget clamping. A requested 128 iterations may execute fewer actual iterations; see `infPlan` in logs.
