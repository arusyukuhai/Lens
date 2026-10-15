# Lens: Count-based Incremental OLS Rule Search (Nim)

This project is a replacement for the original `cms(7).nim` mining-only script,
using the user's `replace(9).nim` and `fast_count_min_sketch(7).nim` as a base.
This revision adds a deterministic frequency-weighted hash alphabet and fresh RHS tokens.

## Contents

- `token_codec.nim`: corpus-frequency-weighted byte-to-token map, keyed hashes, frequency-weighted in-alphabet noise and fresh rule-token namespace, stable map persistence.
- `cms.nim`: corpus loading, deterministic diffusion, pattern mining, count-feature
  OLS, FWL candidate score, removal score, chained replacement proposals,
  one-chunk rotation, simple width-4 survivor-beam expansion, fixed holdout diagnostics,
  and guaranteed alternating fresh/capture RHS generation.
- `replace.nim`: original optimized matcher and replace engine; adds an integrated
  count+rewrite operation and reusable `SeqReplaceProgram` with cached compilation,
  scratch and output buffer.
- `fast_count_min_sketch.nim`: frequency sketch with fused `addAndEstimate` for the bounded heavy-hitter miner.
- `run.sh`: build and launch script.

## Compile and self-test

Install Nim 1.6+ or 2.x and optionally gnuplot (with pngcairo):

```bash
nim c -d:release --opt:speed cms.nim
./cms --self-test
```

The self-test checks exact FWL increment gains and OLS deletion losses against
full OLS fits, and tests fused count-and-rewrite semantics for literal and
wildcard patterns, as well as frequency-tokenization, noise ordering and chained rewrite behavior.

**Validation caveat:** the delivery environment lacked Nim and gnuplot, so
these Nim self-tests and the generated gnuplot PNG could not be executed here.
Algebraic formulas were independently checked against NumPy least squares.

## Run

```bash
./cms --corpus=/path/to/github-code-dataset.txt \
  --out=lens_run --chunks=40 --train=3 --valid=2 \
  --chunk-len=256 --noise-steps=0 \
  --steps=100 --max-rules=24 --beam=4 \
  --swap-depth=3 --token-seed=1729
```

Or use `./run.sh --corpus=... [other flags]` to build and run.

### Frequency-weighted hash tokenization (preprocessing)

Before generating diffusion rows or applying any learned rule:

1. The loaded corpus chunks are counted **byte-by-byte** (0..255, including UTF-8 bytes).
2. Bytes are ordered by **decreasing observed frequency**, with seed-keyed hash tie breaks.
3. Each byte is mapped to a **distinct opaque positive token** in a reserved range.
   Each token's hash-jittered numeric interval has a span **proportional to its
   observed frequency**. Hence sorting captures now reflects frequency-ranked
   tokens, not ASCII order. This is not frequency-proportional *sampling*;
   the map is injective and preserves equality.
4. The mapping is recorded in `token_map.tsv` and reused with `--resume`.
   To explicitly reuse it elsewhere: `--token-map=path/to/token_map.tsv`.
   Old ASCII-valued `rules.tsv` files **cannot be resumed** with this encoding.

Every newly mined rule's RHS contains a **fresh hash marker** in a disjoint
namespace, with a coarse `log2(candidate occurrence frequency)` numeric bucket.
Literal source segments are no longer copied into RHS! **The RHS ALWAYS
alternates newly hashed literal markers and captured wildcard material**, with
markers also at both ends. For example, a two-wildcard pattern emits
`[fresh_A, -1, fresh_B, -2, fresh_C]` (or the swapped-capture equivalent).
Zero wildcards emit `[fresh_A]`. One wildcard yields `[fresh_A, -1, fresh_B]`,
`[fresh_A, -16, fresh_B]` (sort), `[fresh_A, -32, fresh_B]` (reverse), or
`[fresh_A, -48, fresh_B]` (inverse_sort). New markers use deterministic
frequency-bucketed hashes with distinct salts for each gap. Captures remain so
that sort and swaps can affect future rule inputs. `inverse_sort` outputs ranks
(typically small integers). The self-test checks the RHS layout and actual
rewritten sequences. Legacy `[fresh, captures...]` rules supplied with
`--resume` are automatically upgraded and a warning is printed because their
rewrite behavior necessarily changes.

`--noise-steps=0` generates one incremental substitution per character, plus
its original state; fewer levels can be configured to reduce memory/time.
The deterministic noise transformation makes identical chunk/stage pairs
reusable. **Processing order for every diffusion row:**

`original bytes -> frequency-weighted hash encoding -> per-position existing-character noise ->
rule 1 -> rule 2 -> ... -> rule N -> feature-count OLS`

At each rule the non-overlapping match **count is recorded immediately before
that rule rewrites the sequence**. Each subsequent rule receives the actual
preceding rule's output. Nothing resets to the original between rules. The training window rotates forward exactly **one chunk per step**;
the last `--valid` chunks form a fixed held-out set, when available.

## Model mechanics

- The linear regression uses a free intercept plus **number of nonoverlapping
  matching occurrences** of each rule. Counts are measured immediately before
  the rule is applied, at that rule's position in the rewrite pipeline.
- New candidates are mined from up to eight deterministically distributed
  snapshots across the **entire rotated diffusion window**, including later
  noise stages and other chunks; the existing mining budget is shared among
  them. Previously, candidates were mined only from `texts[0]`, systematically
  excluding patterns found exclusively in other snapshots.
- **CMS-backed bounded heavy-hitter selection:** Every observed pattern updates
  a Count-Min Sketch, even when the pattern is not currently retained. A
  min-heap holds up to `--mine-tracked` candidate keys, prioritized by their
  sketch-estimated counts. When a previously untracked pattern becomes more
  frequent than the heap minimum, it evicts that minimum. The heap updates in
  `O(log mineTracked)` per changed retained key, and the minimum is refreshed
  lazily to account for sketch collisions. The sketch's counter update and
  estimate now share a single hash traversal (`addAndEstimate`). CMS estimates
  are used **only for mining preselection**; accepted OLS feature columns are
  measured using exact non-overlapping occurrence counts on all diffusion rows.
- New candidates are evaluated against the **fully transformed text** after all
  existing rules. The exact fixed-design FWL gain is
  `(z^T residual)^2 / (||z||² - ||Q^T z||²)`. Only shortlisted candidates
  produced by the bounded pattern miner are compared, not *all* possible rules.
- OLS uses two-pass reorthogonalized modified Gram-Schmidt. Full-rank deletion
  loss uses `beta_j² / ((X^T X)^(-1))[j,j]` via an inverse triangular `R`;
  a singular design triggers exact reduced-model fits to avoid an invalid inverse.
- For a pattern with one wildcard the branch variants include identity,
  `sort`, `reverse`, and `inverse_sort`. For two or more wildcards all pairs
  of captures may be swapped, along with the identity choice.
- **Simple width-4 beam (no lookahead):** a newly appended rule's match
  count is independent of its RHS operation. For a pattern with one wildcard,
  the learner emits all four possible RHS operators (`plain`, `sort`,
  `reverse`, `inverse_sort`). Their *immediate* OLS Pearson is identical.
  These distinct alternatives may therefore survive together in the beam.
  At the NEXT rotated training window, **every surviving branch** applies its
  own complete sequential rewrite chain, mines/tries independent Add and
  Replace proposals, and is ranked by the actual Pearson from that step.
  This is ordinary beam search, with NO speculative next-rule probes,
  lookahead budgets, tolerance windows, or operator-specific bonuses.
  Width 4 is the default (`--beam=4`); it is a cap on the surviving branches,
  not a promise that each operator survives if higher-scoring branches compete.
- **Fast last-rule insertion:** each shortlisted pattern's FWL gain produces
  its exact current-window OLS Pearson once; the score is shared among its
  RHS variants. For insertion *between* existing rules, the full rewrite
  chain is still refitted because all later features can change.
- `sort` and `reverse` are identity transforms when the wildcard captures a
  single token. Their effect is on downstream *token sequences*, not on the
  current rule's occurrence count. `inverse_sort` maps captured tokens to their
  stable sorted ranks. The self-test explicitly checks output differences and
  that later count features can distinguish transformation variants.
- On **every step**, independently search for the best **addition** and the
  best **replacement** using exactly the same rotating training rows and
  incumbent model. Replacement first targets the rule with the smallest
  conditional deletion loss, then (if the first replacement does not improve)
  follows the rule whose count column changed most, up to `--swap-depth`.
  **Choose replacement only when its full-pipeline refitted OLS Pearson is
  strictly greater than the best addition's Pearson** and it also improves the
  incumbent by `--min-gain` (internal default: `1e-9`). Ties favor addition.
  If both fail to improve, keep the original rules. After `--max-rules` is
  reached, addition is unavailable, so an improving replacement may be used.
  Console output includes `[compare] baseline=... add=... replace=... chosen=...`
  for each step (`n/a` means no improving proposal).
- **Important:** the deletion loss and candidate gain are exactly valid only
  for their fixed current feature matrix. Deleting or inserting a rewrite rule
  changes downstream tokens, so every speculative swap and its acceptance is
  scored by rebuilding the full pipeline, not by trusting the incremental gain.
  During mid-chain replacements, a zero-gain candidate can still change
  later feature columns, so FWL gain is no longer a hard exclusion condition.
  The limited shortlist combines top-gain and top-frequency patterns and then
  evaluates each full pipeline. This is still a heuristic, not a proof that
  the globally best replacement has been found.
- Pattern candidates are deduplicated before scoring; compiled
  pattern/replacement buffers are reused across rows, and diffusion rows have
  a bounded LRU cache. Mining samples starts and has configurable search budgets.
  Consequently the search is a *heuristic beam-like search*, not an exhaustive
  globally optimal search.

## Output

- `lens_run/token_map.tsv`: deterministic original-byte to weighted-hash-token
  map (all 256 bytes), including frequencies and seed.
- `lens_run/rules.tsv`: latest accepted rules, comma-separated integer patterns
  and RHS separated by a tab. Pass this via `--resume=...` on a later run.
- `lens_run/training.dat`: columns `step -ln(1-train_pearson)
  -ln(1-heldout_pearson) n_rules train_pearson heldout_pearson seconds
  -ln(1-train_pearson_MA15) -ln(1-heldout_pearson_MA15)`.
  **Moving average is computed on raw Pearson first**, then transformed with
  `-ln(1 - mean(Pearson))` (not the average of the transformed metric).
  This is a trailing 15-step average; steps 0–13 use all available observations.
- `lens_run/training.gnuplot`: matplotlib-like visual styling, white background,
  soft grid, raw train/held-out lines in light blue/green and their 15-step
  moving averages in darker, thicker blue/green lines.
- `lens_run/training.png`: rendered automatically at each step **if gnuplot is
  installed**. If not installed, the `.dat` and `.gnuplot` outputs are kept.

The *training* Pearson equals `sqrt(R²)` for intercept-including OLS;
held-out Pearson is computed directly from held-out predictions with the
training coefficients, so it is not constrained to improve.

## Caveats / recommended controls

- Cost is dominated by rescanning candidate patterns and evaluating
  speculative mid-chain replacement pipelines. The former two-ply preview
  has been removed completely. `--beam` directly limits how many parents
  expand at each step (default 4); reduce it to 2 for faster training.
  Last-rule RHS variants reuse one FWL result rather than each rerunning OLS.
  Because both Add and Replace compete whenever eligible, increasing beam
  width still raises computation. Current-window scores are refreshed after
  the single-chunk rotation, not carried over as if the data were unchanged.
- `--mine-budget` / `--candidates` / `--mine-starts` can be increased for broader
  search or decreased to improve throughput. `--max-gap` bounds wildcard
  search; large gaps in the original unbounded miner are therefore omitted.
- For large chunks, full one-character diffusion uses quadratic memory in
  chunk length. Set `--noise-steps=128` or reduce `--train` if memory is limited.
- This uses training-set Pearson as the acceptance condition, matching the
  requested objective. No lookahead evaluation is used; the held-out set
  does not affect rule selection. Beam alternatives may be
  individually worse than the best current branch but are not marked as
  accepted winners unless they improve against the current incumbent. The held-out curve must be watched for overfitting.
- Diffusion replaces a position with a **different, corpus-observed character's
  hashed alphabet token**, sampled in proportion to the observed byte frequencies
  conditional on differing from the source byte. It never introduces a new
  noise-only token. The sampling is deterministic for each seed/chunk/position,
  uses a cached cumulative frequency index, and runs **before** any rules.
  If the corpus contains only one distinct byte, such corruption is impossible
  and the sampler raises `ValueError`. The initial length is preserved unless
  downstream rewriting changes it. Existing token maps are still readable,
  but continuing a run trained under the old OOV-noise distribution is not
  equivalent to continuing the old experiment; a fresh run is preferable.

## Audit notes (2026-10-15)

Earlier fixes distributed mining over representative diffusion snapshots and
stopped rejecting mid-chain replacements purely for zero immediate FWL gain.

The subsequent CMS repair removes the "first `mineTracked` patterns win forever"
behavior. Sketch estimates now directly control promotion and eviction in a
bounded min-heap, including when the candidate was first observed after the
tracking table filled. A new self-test checks late heavy-hitter promotion,
heap/table index consistency and stable tracking capacity.

**Remaining approximation:** CMS may overestimate frequency due to hash
collisions, and the bounded mining budget samples only part of the possible
pattern space. Consequently the candidate set is not guaranteed to contain
the globally best rule. `bestPatterns` recomputes the real occurrence counts
before using any candidate for OLS. Neither this nor prior changes imply
out-of-sample improvements.

**Build verification:** No Nim compiler or gnuplot was available in the
execution environment. The Nim build and end-to-end self-tests have not been
run here; use `./cms --self-test` after compiling locally.
