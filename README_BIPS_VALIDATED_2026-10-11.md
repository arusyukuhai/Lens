# Lens / Replacer — Cross-Original BIPS with Validation-Aware CEGAR

**Date:** 2026-10-11. Based on `Lens_Pearson_BIPS_CEGAR_20261011_full.zip`.

This is a refinement of the **outer structural learner**, not a new learning target. The existing exact Native CPU evaluator, ordinary least-squares + Pearson outer evaluation, inner denoising GA, 512-token LUT, checkpoint schema, legacy opcodes, and 4096-capacity HOF remain intact.

## Why this version exists

The earlier CEGAR proxy selected 2–4-rule pathways primarily by *final reader firing* and a small residual correlation on the **same probes** that were used to verify firing. This is easily misled:

- A chain can fire while destroying important existing rules' signals.
- The rule's final firing bit is only one part of its effect on the entire Replacer trace.
- A rule chain can seem predictive on the original from which it was invented but fail on another.
- The old 12-evaluation budget, split over three placements and three depths, typically tried only **one unique rule chain per depth**.
- Rejecting every statistically uncertain chain also prevents evaluation of potentially useful multistep pathways.

## Changes

### 1. Original-group crossfitting (`--bips-crossfit`, default enabled)

With at least three rolling originals, the first `cases-1` sources are available to the BIPS proposer; the *last* rolling original remains exclusively for the standard population fitness. Each crossfit pathway is constructed and ranked using only **one** of the proposer originals. The subsequent CEGAR evaluator uses a *different* original for confirmation. Sources are interleaved to avoid favoring a particular original.

This is a selection safeguard, **not** a completely independent generalization estimate: the champion's old readout was fitted using the existing training sources, and repeated candidate selection still uses the CEGAR confirmation source. Keep a final never-used test corpus for claims about true generalization.

Disable to compare against original joint-source mining: `--no-bips-crossfit`. With fewer than three originals the system falls back to the original mining mode and uses within-source row splits where necessary (not a source-disjoint validation).

### 2. More transferable anchor grammar

The producer and consumer anchors may now be one byte wide (as well as longer forms). Single-byte anchors can transfer across originals when their longer context is source-specific. Intermediate 2-, 3-, and 4-rule pathways still capture gaps with ordinary `-1` and pass private markers through unused internal LUT token IDs. New pattern-side `-2/-3` writes are **not** generated.

A short anchor can also be less selective; the held-out validation and structural checks are necessary to reject excessive false firing.

### 3. Full-phenotype, refitted OLS preflight

For each candidate, Native C++ evaluates the **whole child program** on cross-original probe sets. We then refit tiny ordinary least-squares readouts (up to six training-only features, intercept included) separately for the parent and child. Their held-out-original **Pearson change and calibrated squared-error change** determine confirmation.

This is substantially more faithful than ranking only by the final rule's activation and frozen-parent residual. The micro-readout is still a small-budget approximation; the following generation's full population evaluation is definitive.

A confirmed pathway requires strictly positive Pearson improvement and nonnegative squared-error improvement on the inner confirmation original, after the concrete-versus-abstract mismatch gate. Adjust the Pearson threshold with `--bips-cegar-min-pearson-gain`.

### 4. Allocate time to genuinely distinct programs

The default 12 Native probes now reserve roughly 70% of the budget for distinct pathways and the remainder for window changes / CEGAR-style refinement. Rejected execution windows remain concrete counterexamples; the loop tries alternative placements and diagnoses producer versus downstream failures. This remains **CEGAR-inspired** and is not a complete formal abstract interpreter.

### 5. Bounded uncertainty-aware exploration

With short probe sets, lack of a positive change is not decisive evidence that a multistep pathway cannot improve on a larger corpus. At most `--bips-cegar-exploration 0.25` of BIPS offspring slots (subject to candidate eligibility) are reserved for concrete-working paths with no large detected Pearson/MSE deterioration. These **exploration** children are not described as validated successes, and receive full evaluation in the next generation. Set `--bips-cegar-exploration 0` to disable.

### 6. Audit fields

In addition to prior `bips-cegar` diagnostics, the log now includes:

- `valid=accepted/checked`: positive inner-group Pearson **and** squared-error checks;
- `validReject`: candidate pathways that did not pass the micro-readout confirmation;
- `explore`: unconfirmed, not-clearly-harmful chains admitted for full next-generation evaluation;
- `valPearsonGain` and `valMSEGain`: largest inner-group observed improvements;
- CSV fields `bips_cegar_validation_checked`, `bips_cegar_validation_accepted`, `bips_cegar_validation_rejected`, `bips_cegar_exploration_accepted`, `bips_cegar_heldout_pearson_gain`, `bips_cegar_heldout_mse_gain`.

`rejected_windows` may include exploratory admissions that failed the **strict** confirmation gate, so it should not be interpreted as the count of offspring thrown away.

## Suggested command

```bash
python main.py \
  --backend cpu-native \
  --learning-mode guided \
  --readout-mode linear-pearson \
  --inference-mode ga \
  --cases 3 --samples 100 \
  --bips-share 0.35 --bips-depth 4 \
  --bips-crossfit \
  --bips-candidates 32 --bips-sources 24 \
  --bips-cegar-budget 12 \
  --bips-cegar-probes 16 \
  --bips-cegar-windows 3 \
  --bips-cegar-exploration 0.25 \
  --population 96 --rules 1500 --hof-size 4096
```

**Tip:** `--inference-budget-mode auto` may clamp a requested 128 denoising-GA generations to only a handful. Use `--inference-budget-mode direct` only when you knowingly want to pay the full GA compute budget.

## Verification

- `python -m pytest -q`: 144 passing tests, 4 skipped MPS-dependent tests, 10 passing subtests in the development environment.
- Three-generation CPU Native + inference GA smoke completed using 3 sources and 30 corruption samples per source.
- One smoke run saw 1 exploration-path child at generation 0 and 2 at generation 2; stringent confirmation accepted 0 in that tiny test. This does **not** establish higher real-world accuracy and underlines the need for larger source diversity.
- Tests include crossfit source isolation, rejection of antialigned label signals on a distinct source, native parity of one-byte-anchored multirule chains, and old CEGAR placement regression.

## Important scientific limitations

- This is **not** a proof-producing CEGAR engine, nor a general symbolic preimage solver for all Replacer opcodes.
- Multi-rule construction is still constrained to ordered anchor/capture chains with 2–4 rules. Hidden-memory producer instructions are not added.
- Held-in-discovery source validation is useful but is not an unbiased final test: candidates are repeatedly selected using that source. Reserve never-used corpus sources for the final comparison.
- A pathway's predictive benefit and denoising benefit are different; the inner reconstruction GA is unchanged.
- Crossfit and micro-OLS add cost and can reject all BIPS candidates in small or heterogeneous source sets. The fallback six single-rule strategies remain enabled.
- The current data does **not** demonstrate that this new design is consistently faster or achieves higher validation Pearson than the previous build. Run a fixed-seed, fixed-corpus, multi-seed A/B evaluation under equal wall time, then a final independent-source test before claiming improvement.
