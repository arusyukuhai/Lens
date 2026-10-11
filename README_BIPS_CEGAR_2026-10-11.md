# Lens / Replacer — 4-rule BIPS and bounded counterexample refinement

2026-10-11. Derived from `Lens_Pearson_BIPS_MultiRule_20261011.zip`.  This release **does not** change the learned target, inner reconstruction GA, Pearson/ordinary linear regression, LUT format, HOF limit, or stored checkpoint schema. Code and tests are included.

## Why this revision exists

Earlier BIPS approximated a chain's usefulness by detecting a sequence of raw-input anchors.  But the actual genome is an *ordered, destructive rewriting program*: upstream legacy rules can erase those anchors or transform intermediate tokens.  A highly ranked abstract chain could therefore never activate. The earlier implementation also synthesized only 2- and 3-rule chains, and a handful of candidate placement windows tended to cluster near the end of the genome.

This revision adds a bounded **abstraction → concrete counterexample → placement refinement** loop. It is **CEGAR-inspired**, not a complete formal CEGAR/SYNGAR or model checker: finite training examples do not prove a rule program correct, and the abstract predicate is a heuristic rather than a verified over-approximation.

## Changes

1. **2–4 rule program composition.** `PathProposal` supports up to two independent intermediate constraints (`middle`, `bridges`) along with source/sink constraints. Path mining uses corruption witnesses and ordered non-overlapping literals. Depth-balanced quotas preserve exploration of 3/4-rule macros in spite of stronger-looking two-rule proxy scores.
2. **Backward requirements with forward witness support.** A path is represented as an ordered anchor sequence, not separate frequency-based n-grams. The chain is compiled using private latent marker tokens from the existing 512-token internal space. Every intermediate consumer captures and preserves untouched gap text with ordinary `-1`. No newly generated pattern-side `-2/-3` or hidden-memory writes.
3. **Counterexample-guided placement refinement.** First score the joint raw-input anchor occurrences. Build complete candidate children at diverse locations (tail, middle, front), then evaluate their actual last-chain-rule firing with the full existing genome, repeated sweeps and Hidden Memory rewrite phases on a *small training-only probe subset*. If raw anchors predict firing but the concrete program does not, record that contradiction and reject the placement if mismatches exceed the configurable threshold; retry the same rule path at a different window. The source-stage versus later-stage failure is also counted.
4. **Concrete residual-guided ranking.** After a path is checked, its *actual observed* firing bitset on probes, rather than the abstract n-gram bitset, is correlated with frozen-readout residuals. Each path's best tested placement competes for the limited BIPS child quota. **Full population evaluation next generation remains the authoritative fitness measure**; a mini-probe score is not itself accepted as the model's Pearson value.
5. **Budget control.** Only `--bips-cegar-budget` complete-genome probe calls per generation; per-probe training inputs are limited by `--bips-cegar-probes`. Depth exploration and alternative windows are capped, so the additional cost is explicit. `--bips-cegar-budget 0` restores the previous BIPS child creation path (with the enhanced 4-rule grammar still available).
6. **Measurements and attribution.** CSV fields `bips_cegar_evaluations`, `bips_cegar_rejected_windows`, `bips_cegar_refined_windows`, `bips_cegar_false_positives`, `bips_cegar_false_negatives`, `bips_cegar_probed_inputs`, `bips_cegar_best_gain`, `bips_cegar_blocked_source`, `bips_cegar_blocked_later` and 2/3/4-depth delayed offspring rewards. Existing `bips_added` and guided proposal counters are preserved. The console now explicitly prints `bips-cegar tested=... accepted=...` (the older `guided=...` summary is **only for the six single-rule proposal lanes** and does not include BIPS children).

### Running

```bash
python main.py \
  --backend cpu-native \
  --learning-mode guided \
  --readout-mode linear-pearson \
  --inference-mode ga \
  --bips-share 0.35 \
  --bips-depth 4 \
  --bips-candidates 32 \
  --bips-sources 24 \
  --bips-cegar-budget 12 \
  --bips-cegar-probes 10 \
  --bips-cegar-windows 3 \
  --bips-cegar-max-mismatch 0.35 \
  --population 96 --rules 1500 --hof-size 4096
```

For a strict before/after comparison, vary **only** `--bips-cegar-budget 0/12` on identical seeds and a fixed corpus. The unrefined baseline can use `--bips-depth 3` to match the old model. Compare delayed `bips_prev_improved_d*`, out-of-source Pearson, inference accuracy, and total wall-clock time; avoid comparing Pearson values from different rotating case sets as if the dataset were unchanged.

### Tests

`python -m pytest -q`

- Four-stage marker passing and C++/Python parity;
- Exact counterexample from an earlier rule deleting the required suffix;
- Rejection of two blocked windows and success after moving the path in front of the interfering rule;
- Held-out source isolation (changing the last source and its labels does not affect the probe candidates);
- Regression tests for the previous 2/3-rule BIPS semantics, old checkpoint opcodes, CPU native and inner inference GA;
- Three-generation training smoke with `--bips-depth 4` plus the inner reconstruction GA.

### Caveats / next research steps

- The synthesis grammar is still anchored by literal input substrings plus gap-preserving wildcards, not full symbolic preimage computation; it does **not** synthesize arbitrary operator sequences or derive Hidden Memory channels.
- The predicate abstraction is a *heuristic*. A finite mismatch threshold is not a sound correctness proof. Other rule interactions can occur outside the training probes.
- Candidate mini-probe OLS residual gains are approximate, and overlap/Pearson evaluations are deferred to the next generation. This guards against treating the proxy as a full-genome fitness metric but does **not** resolve statistical multiple-testing or out-of-distribution generalization by itself.
- CEGAR calls add workload. They do not guarantee higher wall-clock performance or learning accuracy. If large real input sequences make this expensive, reduce `--bips-cegar-budget` (e.g. 4) and/or `--bips-cegar-probes` (e.g. 6).
- A more complete next version would use **typed symbolic intermediate state** (literal interval predicates + capture equalities + hidden memory content) and perform backward preimage computation with proof-producing counterexamples. This release intentionally stops short of making an unsupported formal correctness claim.
