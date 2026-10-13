# Lens v10 — Precision-first Dependency Regrow (experimental)

Built on v9 Counterfactual Regrow (indent-fixed). This release deliberately
changes the structural repair mechanism, **not** the Replacer semantics,
all-pairs readout, source-disjoint acceptance/anchor gate, fixed held-out,
or `-log2(1-Spearman)` plot.

## 1. Fixed a regression in nonlocal dependency detection

`lens_dependency.nim::affectedRuleOrder` used to skip the revised pipeline's
transformation whenever a rule at or before `changedIndex` fired in *both*
chains. The resulting replay compared unequal histories and could report
spurious nonlocal dependencies. Now BOTH chains replay every rule, regardless
of firing agreement, and only the post-root rules contribute to the drift
measurement.

A dedicated regression test reproduces a *never-firing root edit* followed by
active prefix-dependent rules: there must be **zero** affected nodes.

## 2. Joint counterfactual probes for repair interaction

For the top N (`--regrow-pair-probes:3`) empirically affected rules, evaluate
small **pairs** of temporarily disabled rules. Each trial fully replays the
edited chain, refits the readout on Train and scores on Search. Probe results
are memoized within one rollback transaction and do not see Acceptance,
Anchor, or protected Held-out.

Let `g(i) = Merit(mask(i)) - Merit(path)` and
`g(i,j) = Merit(mask(i,j)) - Merit(path)`. Define:

    synergy(i,j) = max(0, g(i,j) - max(g(i),g(j))) if g(i,j)>0, else 0

This is **an empirical repair heuristic**, not a proof of causality or a
prediction of the exact benefit of editing either rule. It gives a limited
chance to consider joint effects invisible to single-rule probes, while still
reserving a slot for the strongest directly observed firing/edit drift.

With `--regrow-pair-probes:0`, behavior reverts to single-target v9 scoring
(apart from the essential dependency-trace bug fix).

## 3. More room for compensating downstream edits

The ROOT proposal frontier is allowed up to `2 * --regrow-roots` states
before post-root repair, instead of a single narrow frontier. This preserves
some candidates that temporarily lose Search score before a downstream rule
is fixed. It does not increase the number of eventual independent Acceptance
finalists (`--regrow-finalists` remains 3 by default).

## Recommended settings (precision prioritized)

    nim c -d:release --opt:speed cms.nim
    ./cms --corpus:github-code.txt --out:lens_v10 --iterations:100 \
      --regrow-positions:1 --regrow-patience:5 \
      --regrow-depth:2 --regrow-roots:4 --regrow-targets:2 \
      --regrow-beam:2 --regrow-probe-targets:4 \
      --regrow-pair-probes:3 --regrow-match-pool:64 \
      --regrow-finalists:3

For cheaper testing, set `--regrow-pair-probes:0` and
`--regrow-roots:2`; for ablation, compare 0 vs 3 vs 4 pair probes with the
**same corpus split, seeds, anchor/acceptance, and initial model**. When
resuming from v9, explicitly pass `--regrow-positions:1` if the checkpoint
saved the previous opt-out value.

Tests:

    nim c -r -d:release test_regrow_v10.nim
    nim c -r -d:release test_regrow_v9.nim
    nim c -r -d:release test_validation.nim

## Limits / research ethics

- Pairwise mask rescue can misrank actual repair gain; the benefit is not
  guaranteed. Repair candidates still require full-chain recomputation and
  the unchanged independent Acceptance+Anchor promotion gate.
- Regrow remains disabled by default for legacy checkpoints that saved
  `--regrow-positions:0`; turn it on explicitly as shown above.
- The development environment does not include Nim or the original training
  corpus. Compilation, speed, and *real-world precision improvement* are
  therefore not yet demonstrated. Do not treat the numerical heuristic as an
  experimental accuracy result.
