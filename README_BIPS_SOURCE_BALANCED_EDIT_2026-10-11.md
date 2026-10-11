# Lens / Replacer — Source-balanced BIPS and edit-sensitive macro evaluation

This revision is built on `Lens_Pearson_BIPS_FreshConfirm_20261011_full.zip`.

## Root cause demonstrated by the user's ~174-generation run

- The default `--cases 2 --samples 100` reserved **one** case as the outer held-out Pearson evaluator. Therefore BIPS saw **one training original**, but final macro confirmation required **two different training originals**: one for synthesis/readout fitting, one for cross-original evaluation. The two requirements were mutually inconsistent.
- The supplied training log recorded 1,979 BIPS CEGAR concrete checks, 508 BIPS offspring, but **zero final confirmation probe rows** and zero causal interventions. Those 508 offspring were from the explicitly unconfirmed exploration lane.
- Even correctly measured global cleanliness correlation is not itself a direct measurement of the ability to distinguish a *correct local edit* from an *incorrect local edit* of the same noisy text. The inner string GA relies on exactly that ordering.

## Fix A: source-level verification without more trajectories

When the outer learning mode is guided/hybrid, BIPS share and source-confirmation are enabled, at least three eligible corpus chunks are available, and `--cases=2`, the new `--bips-auto-case-balance` (default enabled) rebalances **2 × N → 3 × round(2N/3)**. For example, 2×100 (200 trajectories) → 3×67 (201 trajectories):

1. Original 0: proposal synthesis and readout fitting.
2. Original 1: validation and fresh confirmation on *disjoint* corruption rows.
3. Original 2: normal population-level Pearson evaluation; excluded from BIPS proposal mining, CEGAR, fresh confirmation, and edit preference.

Both of the first two cases may swap roles according to which source a macro was mined from. Every rule chain is still run in the **actual** Replacer (including intermediate rewrites, hidden-memory semantics, and multiple sweeps). Nothing bypasses the ordinary next-generation fitness evaluation.

The loader prints `init: BIPS case rebalance ...`. To preserve an exact old two-case benchmark (including its unavailable cross-original confirmation), use `--no-bips-auto-case-balance`, or choose `--cases 3` manually. Short corpora with fewer than three originals cannot support this split, and the code prints a warning rather than pretending otherwise.

## Fix B: a task-aligned counterfactual preference for macro selection

`--bips-edit-contrast-weight 0.15` (default) and `--bips-edit-contrast-pairs 3` add a **bounded, soft macro-ranking term** after a macro already passes final confirmation:

1. In a *validation original different from the macro's synthesis original*, identify the clean control, and select noisy examples with known differing bytes.
2. On each example construct **two edit candidates at exactly the same byte position**: the correct one-byte repair and an incorrect same-position byte replacement.
3. Execute both child and parent *as complete Replacer programs*, keeping the readout fit restricted to the synthesis original.
4. Compare their preference margin for the correct edit. Prefer macros that improve the inner GA's ability to select the correct change, with a small positive reward for absolute correct-direction preference.
5. Add a bounded preference term to the ranking score. It does **not** replace the ordinary linear regression + Pearson fitness, does **not** modify the inner string GA, and is not a supervised denoising pretraining loop.

New statistics: `editPairs`, `editGain`, CSV: `bips_edit_contrast_pairs`, `bips_edit_contrast_best_gain`, `bips_edit_contrast_positive`.

This uses oracle edits from training/validation originals for *candidate selection*, **not** to update regression weights from validation rows. Because the same originals may be reused across generations, only a separate never-trained corpus is sufficient to establish genuine generalization. It is still possible for validation-based repeated model selection to overfit.

## Recommended command

```sh
python main.py \
  --backend cpu-native \
  --learning-mode guided --readout-mode linear-pearson \
  --inference-mode ga \
  --cases 2 --samples 100 --bips-auto-case-balance \
  --bips-depth 4 --bips-share 0.35 \
  --bips-cegar-budget 12 --bips-cegar-probes 16 \
  --bips-cegar-adaptive-probes \
  --bips-confirm-budget 2 --bips-confirm-probes 64 \
  --bips-edit-contrast-weight 0.15 --bips-edit-contrast-pairs 3 \
  --bips-causal-budget 2 \
  --population 96 --elites 40 --rules 1500 \
  --hof-size 4096
```

Compare ordinary confirmed BIPS and edit-aligned confirmed BIPS using the added sixth arm of `compare_bips_modes.py`. If you pass `--cases=2` into the ablation, the script balances **all** arms to three original cases so guided-only and BIPS variants receive an identical number of trajectories.

## Verification

- Existing CPU native, Guided/Diff3, Pearson, hidden memory and reproduction behavior is covered by the project's regression tests.
- New tests cover source-disjoint confirmation, correctly preferred edit candidates, score-term integration and CLI balancing of the fixed trajectory budget.
- One small 5-generation native/GA integration smoke run completed. Importantly, its confirmation checks reached `confirmRows=64` in a generation (previously always zero). The small artificial corpus did **not** demonstrate a confirmed macro win; its confirmed candidate was rejected on additional evidence, which is an intended use of the gate.

## Limitations / interpretation

- The proposed edit preference is a bounded **surrogate** for full GA recovery: real inference may involve many coupled edits, rearrangements, and context-dependent transformations.
- Source-disjoint training-time confirmation with two original chunks remains a low-sample proxy. It is not an entirely independent audit dataset; the latter must be supplied to `audit_generalization.py`.
- Automatically switching from 2×100 to 3×67 keeps total trajectories almost unchanged, but changes per-case statistical reliability. Compare using several seeds and matched wall time.
- `--inference-budget-mode auto` may still reduce the inner GA to ~6 iterations in the user's 96-genome setup, so reconstruction accuracy is a combination of representational ability and inference search budget.
- No improvement in unseen-corpus Pearson, actual reconstruction accuracy, or training throughput is claimed without matched benchmarks.
