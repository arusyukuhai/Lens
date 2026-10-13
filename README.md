> Accuracy-oriented all-pairs rank readout experimental branch. See [DENSE_RANK_V9.md](DENSE_RANK_V9.md). New runs default to `--rank-scales:0` with shortlist readout tuning; historical mode: `--rank-scales:4 --readout-ridge-probes:1`. No measured accuracy/speed uplift is claimed.

# Lens v8 — Residual-Guided Structural Search

**Start here:** [V8_REDESIGN.md](V8_REDESIGN.md) explains the new bounded
wildcard-gap proposal, training-only rank-residual screen, original-source
consensus acceptance, commands, and important limitations. The existing
Replacer rule semantics, multiscale regression, JSON checkpoint format and
-log2 Spearman plot remain intact. Legacy Rollback is opt-in for new runs.

---

# Lens v7 — multiscale rank training / balanced sources / gated readout

**Start here:** [V7_DESIGN_AND_ABLATION.md](V7_DESIGN_AND_ABLATION.md)

The v7 learner is a structural change, not another search hyperparameter tweak:
- **Multiscale rank pairs** (1,2,4,8) instead of only adjacent noise stages, fitting a closer surrogate to sequence-wide Spearman.
- **Equal source-path contribution** to regression regardless of trajectory length.
- **Gated readout-only improvement** so a validated champion may improve its weights without adding or deleting rules. Rejected updates leave rules and weights untouched.
- **Bounded two-path noise trajectories** and optional sampled n-gram starting positions to control time/memory. Exact mining remains `--mine-start-stride:1`.
- More source chunks for validation. Original source-tokenization, RHS rules, dependency-aware rollback, protected holdout and log2 chart remain.

## Recommended new run

```sh
nim c -d:release --opt:speed cms.nim
./cms --corpus:github-code.txt --out:lens_run_v7 --iterations:100 \
  --rank-scales:4 --balanced-rank-fit:true \
  --mine-start-stride:2 --readout-refresh-every:4
```

Run the new parity test first: `nim c -r -d:release test_v7_multiscale.nim`.

**Scientific status:** this release has not been Nim-compiled or benchmarked in the authoring environment. The v6/v7 differences are hypotheses to test; the improved generalization curve is *not* established. Source-level tests and the independent linear algebra check are provided.

---

> **v6 Generalization-first / CacheFast**
> Start with [OPTIMIZATION_V6.md](OPTIMIZATION_V6.md) for the reproducible shuffled source split, shared training-derived noise alphabet, exact presence/cache optimizations and cross-chunk robustness ranking. Existing v5 checkpoint resumes without changing source order or noise protocol. New `--out` is recommended to evaluate v6 honestly.

# Lens v5 — Frozen Champion & Robust Validation

This release is based on **Lens Rollback Acceptance Gated v4** and preserves
its nonlocal Dependency-Aware Regrow, beam search, wildcard rules, 4096-token
cap, and checkpoint compatibility. The important changes are *evaluation*
and *generalization* corrections, not simply more beam width or a more
expensive rollback procedure.

## Why this changes the design

Previous v4 code ran `accepted.evaluate(train, selection, ridge)` at the start
of **every iteration**. Thus even when `promoted=false`, the incumbent's
regression weights were refitted on a different 3–4-chunk training window.
The saved model and protected heldout score changed while the rule sequence
was identical. This is a correctness bug in the meaning of "accepted
champion", not ordinary training variance.

In v5 the champion's weights are **frozen until a new candidate is promoted**.
The exploratory beam may be refitted every iteration, but the accepted model
is not. Its weights are serialized in a **format-3 checkpoint** and restored
without refitting. Old formats 1 and 2 are supported, but because they did
not store the champion's weights, the first resumed iteration performs one
explicit migration refit. Later rejected iterations do not change them.

## Promotion: three safeguards

1. A candidate must outperform the SAME frozen incumbent on a rotating
   acceptance batch, with the same source chunks and corruption realizations.
2. A candidate must *also* improve on a separately reserved anchor batch of
   independent source chunks. The anchor batch refreshes every N iterations
   to reduce repeated overfitting to one tiny validation set. Reserved anchor
   pool sources never appear in rotating acceptance, although different
   anchor batches come from the same designated reservoir.
3. Both candidate correlations must exceed `--accept-floor` (default 0.0):
   improving from -0.42 to -0.28 is not a meaningful promotion by default.
   Additional rules also incur the existing `--rule-penalty`.

Note that *reusing any validation pool for many model-selection steps still
leaks selection information*. Protected heldout is never used for promotions,
rollbacks, or hyperparameter selection. Even 8 heldout chunks are only a
small diagnostic sample. The input format lacks repository identifiers,
so source-chunk separation alone cannot guarantee repository-disjoint data;
near duplicates may remain across partitions.

## Data and speed

- Keep `--window:3 --replay-chunks:1` as the **mining** corpus.
- Add `--fit-replay-chunks:4` training-only, widely spaced corpus chunks;
  these are *not* mined for n-grams and are used only for regression fitting.
- The new fit-only trajectories use `--fit-replay-steps:64`, keeping cost
  bounded even for 4096-token chunks. This reduces training-window fragility.
- Search and rotating acceptance default to **3 and 4 source chunks** rather
  than 1 and 2. Their trajectories use `--validation-steps:96` instead of
  all single-character stages, so verification costs less than before.
- Anchors use `--anchor-steps:96`; protected heldout uses
  `--monitor-steps:96`. `--noise-steps:0` still means **all stages in the
  primary training chunks**; no implicit training-stage subsampling.
- Rollback is still gated on stalled model promotions, now 5 consecutive
  failed robust promotions by default. The existing nonadjacent causal-drift
  heuristic and bounded regrowth stay in place.

## Recommended: clean baseline run

```sh
nim c -d:release --opt:speed cms.nim
./cms --corpus:github-code.txt --out:lens_run_v5 --iterations:100 \
  --max-chunk-tokens:4096 --beam:4 --lookahead:2 \
  --search-chunks:3 --accept-chunks:4 --anchor-pool:16 \
  --anchor-chunks:4 --anchor-refresh:8 --holdout:8 \
  --fit-replay-chunks:4 --fit-replay-steps:64 \
  --validation-steps:96 --anchor-steps:96 --monitor-steps:96
```

To resume an older checkpoint (while preserving its old partition parameters
where explicitly stored), overwrite the modules and rebuild, then use:

```sh
./cms --resume:lens_run/checkpoint.json --iterations:100
```

The first V1/V2-resumed iteration necessarily re-fits the legacy champion
because its checkpoint did not contain weights. The old output directory's
`training_v2.tsv` is untouched; v5 logs go to **training_v3.tsv** with
columns: iteration, frozen champion train rho, frozen champion search rho,
rotating acceptance rho, anchor acceptance rho, protected heldout rho,
baseline rotating rho, baseline anchor rho, rule count, promoted, seconds,
regrow-proposal. `training.png` is regenerated from v5's TSV. The anchor
series can jump at refresh boundaries; the heldout series uses identical
fixed source trajectories and *should be exactly flat on rejected updates*.

Use `--anchor-pool:0 --anchor-chunks:0` to disable the secondary anchor gate,
though doing so weakens model selection. To disable rollback entirely for an
ablation study, use `--regrow-every:0` or `--regrow-positions:0`.

## Validation

```sh
nim c -r -d:release test_generalization_v5.nim
nim c -r -d:release test_validation.nim
nim c -r -d:release test_rollback_policy.nim
nim c -r -d:release test_lens.nim
```

Tests include anchor vs. rolling disjointness, two-source-group promotion
requirements, rejection of negative-only "improvements", and an exploratory
refit that leaves champion coefficients unchanged. All previous v4 tests
are included. **The provided environment does not have Nim or gnuplot**;
compilation and accuracy/speed benchmarking have not been performed here.
Please run the tests locally and compare v4 and v5 from new runs under
identical corpus, random seed, and protected heldout group before concluding
that generalization has improved.


## Plot styling update (log2 reference-chart edition)

This variant modifies only `cms.nim`'s `emitPlot` visualization string.
It replaces the two-panel raw-Spearman plot with one scientific-style 1600x870
figure (light grid, bottom-left boxed legend, moderate line/marker weights).
**Each displayed series** is transformed using `-log2(1 - Spearman rho)`;
negative rho values remain negative and rho=1 is capped only for visualization.
The source file `training_v3.tsv`, fitting objective, search metric, validation
acceptance rule, checkpoint format and rule-search behavior are unchanged.
The series are fixed held-out, rotating anchor, rotating acceptance, and search
validation. Unlike the fixed held-out series, rotating series are measured on
changing source chunks and their iteration-to-iteration values are not a
strictly comparable generalization curve.

After rebuilding `cms.nim`, plots are regenerated as part of each iteration.
To replot an existing run immediately, enter the run output directory and run:

```sh
gnuplot training.gnuplot
```

For an existing run, first rebuild and run at least one iteration to write the
new `training.gnuplot`; or copy `examples/training.gnuplot` into the run folder
that contains `training_v3.tsv`, then execute `gnuplot training.gnuplot`.
