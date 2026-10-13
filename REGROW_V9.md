# Lens — Counterfactual Dependency-Aware Rollback (experimental v9)

This is an **experimental** refinement of the v8 All-Pairs Readout ZIP. It
changes **only structural rollback/regrow**; all-pairs rank fitting, residual
screening for ordinary beam search, original-source validation, frozen champion,
4096-token skip, and the `-log2(1-Spearman)` plot are unchanged.

## Motivation

The previous dependency-aware rollback ranked every affected downstream rule
solely by the drift of its firing / local edit signature. High drift is not
necessarily high **repair value**. It then returned only one maximum-Search
candidate, even when an alternative structural repair might generalize better.

## New three-stage decision

1. **Detect nonadjacent affected rules:** Compare the full original and
   modified rule chains on balanced *training* noise snapshots. This is a
   behavioral impact approximation, not guaranteed causal discovery.
2. **Probe counterfactual rescue:** For the strongest affected targets, replace
   only the target LHS **temporarily** with a guaranteed-unseen fresh literal;
   replay all remaining rules and refit the readout using Train, then compare
   their Search stability merits. This *disabling probe* measures the benefit
   of removing that rule's effect in the root-modified chain, not the effect of
   repairing it. Always reserve one target for highest observed drift, so
   a useful synergistic rule is not lost merely because masking it hurts.
3. **Mechanistic LHS proposals:** Evaluate a small high-frequency/stratified
   pool of mined LHS against the target rule's **old Train firing signature**.
   Retain half the final shortlist with the best F1 of matching old firings,
   half for frequency/diversity. Candidates still require FULL Train fit +
   Search validation. This step is *optional* since a desirable repair might
   use a novel firing pattern.

Keep up to three distinct Search-selected finalists, including candidates
up to 0.02 below the original Search merit, for the existing independent
Acceptance + Anchor gate to adjudicate. The protected Held-out is NEVER used
by the rollback controller. Capping finalists matters: testing too many
candidates against the acceptance data can overfit that gate.

The external `rollbackRegrow(...) -> State` API is preserved as a compatibility
wrapper preserving nondecreasing raw Search Spearman for legacy callers.
The main learner calls `rollbackRegrowCandidates(...) -> seq[State]`.

## Usage

New runs retain the v8 **opt-in** default for expensive regrowth. To enable:

```sh
nim c -d:release --opt:speed cms.nim
./cms --corpus:github-code.txt --out:lens_regrow_v9 --iterations:100 \
  --regrow-positions:1 --regrow-patience:5 --regrow-every:4 \
  --regrow-depth:2 --regrow-targets:2 --regrow-beam:2 \
  --regrow-probe-targets:4 --regrow-match-pool:64 \
  --regrow-finalists:3 --regrow-finalist-slack:0.02
```

To compare against **drift-only** v9 in an otherwise identical new run, use
`--regrow-probe-targets:0 --regrow-match-pool:0 --regrow-finalists:1
--regrow-finalist-slack:0`.

To resume an older checkpoint, specify the new switches after `--resume`:

```sh
./cms --resume:lens_run/checkpoint.json --iterations:100 \
  --regrow-positions:1 --regrow-probe-targets:4 \
  --regrow-match-pool:64 --regrow-finalists:3
```

Saved options from the old checkpoint still apply unless overridden. The
checkpoint's newly missing search options are filled from the v9 defaults.

## Bounded cost / audit

- Only `--regrow-probe-targets` affected indices receive temporary masking
  and a refitted full-chain Search evaluation **per path/wave**, with a cache.
- `--regrow-match-pool` patterns receive a TRAIN-only batched presence test
  before an unchanged maximum `--repair-candidates` expensive evaluations.
- Mining remains bounded by `--regrow-mines` and all proposal states have to
  pass the existing independent source-wise acceptance gate.
- Logs include `maskGain`, `maskProbes`, `maskCacheHits`, and finalist count.

For performance evaluation, measure wall-clock time *per accepted improvement*,
not just mean iteration time. For scientific accuracy, compare two independent
training runs using the same dataset partition and random seed, with an
untouched external test corpus; don't tune on the protected heldout graph.

## Validation

Run:

```sh
nim c -r -d:release test_regrow_v9.nim
nim c -r -d:release test_validation.nim
nim c -r -d:release test_rollback_policy.nim
```

The local authoring runtime did **not** have Nim installed; the code was
statically inspected, but compilation and real-corpus accuracy gains are
not claimed. These are hypotheses, not established performance results.
