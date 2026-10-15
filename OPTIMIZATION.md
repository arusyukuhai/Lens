# Lens Firing Density Ridge — exact speed optimizations

**Revision basis:** `Lens_Firing_Density_Ridge.zip` (not any later Dual Batch or geometric-mean branch).

This release preserves firing density (`match_count / original_chunk_length`), the clean-fraction target, feature standardization for Ridge, the existing `--ridge-lambda`, beam width, rule-search policy, candidate screening, CMS, sequential rewriting, noise, and plotting. **No geometric-mean pair features** have been introduced.

## Changes

1. **Ridge sibling-prefix Gram reuse.** When `rankBranches` already reuses the pre-exchange text/feature prefix, it now computes that fixed prefix's feature means, standard deviations, and standardized Gram submatrix once per unique shared prefix in the rotated training window. The block is cached
  inside the existing bounded prefix-state cache (at most two stored states),
  so later sibling groups with the same prefix can reuse it too. Each
  candidate's Ridge solver reuses this block and computes only the remaining (changed) Gram entries. This saves repeated ~`O(N * prefix_rules²)` work per sibling. The same accelerated path is used for both ordinary suffix ranking and delta-suffix ranking. There is no approximation: the Ridge normal equations and Cholesky solver are unchanged. No savings when `prefix_rules=0` or prefix-state reuse is disabled.

2. **Less work in incremental Ridge Pearson.** For each fitted Ridge baseline, cache the prediction/target centered covariance and prediction norm while predictions and residuals are already being computed. Appending one candidate column now updates these two scalar statistics using the one-column Schur complement. This removes a redundant `O(P²)` Gram multiplication and temporary beta array from *every shortlisted candidate* (the block inverse multiplication and counting cost remain).

3. **Self-test for the prefix optimization.** `--self-test` compares full-vs-cached-prefix Ridge beta, residual and Pearson at multiple lambda values. The original fast-append-vs-full-Ridge regression tests remain in place.

## Should you switch back to OLS?

**Not for speed alone.** `--ridge-lambda=0` already switches to QR-based OLS, so run an actual matched A/B benchmark. With many rules, both methods have substantial `O(N P²)` work in this implementation. Ridge also provides stabilizing regularization when feature columns become highly dependent. OLS can be faster or slower depending on data, conditioning, and candidate workloads.

### Benchmark / profile

```bash
nim c -d:release --opt:speed -d:lensProfile -o:lens_train cms.nim
# Supply the same corpus and training configuration for each experiment:
./lens_train --corpus=github-code-dataset.txt --out=prof_ridge \
  --steps=10 --ridge-lambda=0.01 --plot-every=0
./lens_train --corpus=github-code-dataset.txt --out=prof_ols \
  --steps=10 --ridge-lambda=0 --plot-every=0
```

Compare the `[profile-rank] ridge`, `[profile-evaluate] ridge`, `[profile] mine/screen/exact` and `[profile-step]` fields. Use identical corpus, seed, beam, rows, and rule budget. After the first update, the two methods may select different rules, so their times at later steps are *not* a pure solver microbenchmark. A stronger controlled test resumes from the same initial `rules.tsv` and `token_map.tsv` (with separate `--out`) and measures step 0.

The uploaded run log uses `internalPearson` / `graphTrainPearson` labels, while this baseline ZIP logs `trainPearson`. Therefore its timing trend should not be treated as an isolated profiler trace from the exact source being optimized.

### Cost caveat

The dominant worst-case costs may still be pattern matching, re-encoding the suffix, the `O(NP²)` *non-shared* Gram entries for early-rule exchanges, or Cholesky/inverse computation. This patch guarantees **less duplicate arithmetic**, not a specific wall-clock speedup. Actual speed must be measured after a local Nim build.

## Validation status

- Algebraic candidate-update identity compared in 300 NumPy randomized trials to a full Ridge solve (agreement within `1e-9`).
- Nim self-test extended to check prefix-cache fit equivalence.
- This environment does not have a Nim compiler or gnuplot, so neither compilation nor end-to-end speedup has been confirmed here.

## Fingerprint size A/B

The candidate `exact` bottleneck spends time counting 96 patterns across
10–12M source tokens per stage. The existing 512-bit token-presence bitmap
can saturate with many distinct tokens, forcing full pattern matches that
cannot succeed. The default bitmap has been enlarged to 8192 bits, with a
`-d:lensFingerprint512` legacy-compatible benchmark option. A 2000-distinct-
token row fills about 98% of a 512-bit one-hash filter but only 22% of an
8192-bit filter under approximately uniform hashing. The real benefit
depends on token diversity and absent literal incidence. This optimization
is exact (only false positives are possible), unlike lowering
`--screen-candidates`, which changes search quality.
