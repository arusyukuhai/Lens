# Lens v8: residual-guided structural search

## The actual problem

v7's fixed-heldout graph **does improve** (roughly `-log2(1-rho)=5.4` late
in the run), but a few fixed chunks cannot demonstrate robust generalization.
The very high and volatile *search* curve is not itself a learning target.
Continuing to tune rollback/beam alone does not address the proposal bottleneck.

## Three-layer pipeline

1. **Proposal stage:** mine clean transformed chunks to discover candidate wildcard
   LHS strings. Rather than enumerating every possible suffix gap, optionally
   use a *bounded, short-gap-heavy, start-jittered logarithmic sampling* per wildcard. The
   candidates and approximate frequency ranks **change**: this is not an
   exact optimization of the old mining algorithm.
2. **Supervised screening:** evaluate all mined LHS patterns for presence on
   training noise rows, build trajectory-centered *rank residuals*, and keep
   at most 256 strongest residual predictors, plus 48 exploration probes.
   SEARCH wildcard matching, multiscale regression, Spearman evaluation,
   phi and RHS expansion are only performed for these retained candidates.
   Exact SEARCH matching on the shortlist is tested against full matching. Unselected
   candidates may be valuable; the exploration reserve mitigates this.
   This stage uses **training data only**. Held-out and acceptance data are
   never used for candidate screening.
3. **Paired acceptance by original source:** reuse the existing independent
   rotating acceptance and anchors. In addition to their mean Spearman
   improvements, the new model must improve at least half of the original
   source chunks in both pools (default `--min-source-win:0.5`). Multiple
   noise trajectories of one source are averaged before being counted.
   A small number of sources still causes high statistical uncertainty.

v8 makes legacy dependency-aware rollback *opt-in* for **new runs** by
`--regrow-positions:0`; the implementation remains available. v7 checkpoints
retain their stored settings on resume. This is not a removal of the Lens
Replacer or wildcard-to-RHS learning; it refactors its proposal and selection.

## Commands

New run (recommended for comparing with v7):

```sh
nim c -d:release --opt:speed cms.nim
./cms --corpus:github-code.txt --out:lens_v8_run --iterations:100 \
  --mine-start-stride:2 --mine-gap-samples:6 \
  --screen-candidates:256 --screen-explore:48 \
  --min-source-win:0.5 --regrow-positions:0
```

Exact previous mining and all-candidate ranking (for ablation):

```sh
./cms --corpus:github-code.txt --out:lens_v8_exhaustive \
  --mine-start-stride:1 --mine-gap-samples:0 \
  --screen-candidates:0 --min-source-win:0
```

Compare v8 defaults to ablation runs with the same seed and same held-out
chunk IDs. Use a separate **unseen** external corpus for final evaluation.
Repeatedly interpreting the same protected held-out can guide human choices
and eventually overfit even if the code never uses those metrics.

## Tests

```sh
nim c -r -d:release --opt:speed test_v8_architecture.nim
nim c -r -d:release --opt:speed test_miner_fast.nim
nim c -r -d:release --opt:speed test_v7_multiscale.nim
nim c -r -d:release --opt:speed test_validation.nim
```

## Limits

- A bounded gap sampler is **not exact** and could miss useful long-range
  dependencies. Set `--mine-gap-samples:0` to restore full enumeration.
- Rank-residual screening is a first-order proxy; candidates that only help
  after an RHS rewrite may be dropped. Exploration probes remain available.
- Requiring broad per-source wins reduces spurious promotions but can also
  delay legitimate specialized improvements. Set `--min-source-win:0` to disable.
- The existing fixed-heldout counts only 8 source chunks by default. We do
  not claim accuracy or runtime improvements without actual Nim benchmarks
  on the source corpus.
