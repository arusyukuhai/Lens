# Lens: online residual-Pearson progressive pruning

Base: `Lens_DualGrid_Fast_FineEveryGen.zip` (same Nim modules, original dual-grid diffusion, Coarse -> Fine reranking **every generation**, independent `--eval-every`, Ridge/OLS, fingerprint, delta-suffix).

## Implementation

Candidate miner -> existing small-window heuristic screen -> **progressive pruning** -> exact full-window Ridge/OLS -> existing Coarse/Fine branch evaluation.

For each candidate pattern, observe progressively more diffusion rows. Each observed pair is

```
x = nonoverlapping_match_count / original_pre_noise_token_length
residual = y - baseline_ridge_prediction
```

A numerically stable Welford online update maintains `(n, meanX, meanR, sumDeltaX2, sumDeltaR2, sumDeltaXR)` in O(1) arithmetic per observed row and candidate. Branches are ranked temporarily using `abs(Pearson(x, residual))`, **not** a Ridge score, and **not** a confidence bound. Because diffusion states are correlated and a new rule's gain depends on other rule features, this is a heuristic whose pruning can remove a valuable candidate.

Rows are processed in breadth-first midpoint order throughout the entire rotated training set, to cover different chunks and noise-rate ranges early instead of visiting all clean/small-noise stages first. Snapshots themselves are NOT changed. When a checkpoint is reached the candidate pool is reduced (`--progressive-factor`) while protecting a small number of top-frequency candidates. The default progression with 96 screened candidates and plenty of rows is about `96 -> 48 -> 24` after 64 and 128 rows. Each surviving candidate's **already computed** match density values are carried into the full-window vector, so these rows are not matched again. Exact full-window `candidateGain` then uses the existing Ridge/OLS math.

Cached feature vectors from sibling beam parents are read rather than rematched. Cache semantics, the deterministic CMS frequency-derived RHS salts, and `--fine-top` are preserved.

## CLI

```
--progressive=1              # new: defaults on; use 0 to reproduce original candidate scoring
--progressive-start=64        # evaluate at least this many distributed diffusion rows before first prune
--progressive-keep=24         # never prune below max(this, required beam selection count)
--progressive-factor=0.5      # retain at least this fraction of remaining candidates per checkpoint
```

There is still the **older** `--screen-rows` / `--screen-candidates` prescreen before this stage. In profile builds, `[profile]` now shows `screened_candidates`, `exact_candidates`, `progressive_rows`, `progressive_pairs`, `progressive_full_pairs`, `progressive_pruned`, and `progressive_reused_pairs`. `exact_total_rows` counts rows remaining to be scanned *after* progressive sample reuse; `exact_match_seconds` excludes progressive matching (counted in `screen` time).

## Speed and precision tradeoffs

- `--progressive-keep=48` or `--progressive-start=128`: safer but slower than 24/64.
- `--progressive=0`: disable NEW pruning; the old approximate `screenPatterns` stage remains. To disable both filters use `--progressive=0 --screen-rows=0`.
- For a controlled comparison, use the same corpus, fixed seed, original initialization, and separate output directories; compare held-out performance and wall-clock time over multiple seeds.
- Branch-level full-rule re-encoding (`rankBranches` / `suffix_encode`) is **not** pruned by this revision. Only rule-pattern candidate full-row matching is optimized. It may remain the dominant runtime. Fine branch reranking continues every generation and is exact for the Coarse finalists it receives.
- The score is not directly Pearson of `x,y`; it uses residuals from the existing regression, which prioritizes features that could explain presently unexplained signal.

## Verification

- Updated Nim `--self-test`: online Pearson, spread-out order, count-vector reuse, pruning enable/disable, cached vector consistency.
- `verify_progressive.py`: independent Python numerical checks and source wiring checks.
- Previous four Python verifier scripts retained; all passed.
- **Nim compiler is unavailable in the provided environment**, therefore Nim compilation/self-test execution and a measured speedup remain unverified.
