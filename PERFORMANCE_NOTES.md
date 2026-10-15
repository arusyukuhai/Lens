# Lens Dual-Batch Ridge: CPU hot-path optimization

**Basis:** This release modifies `Lens_Dual_Batch_Ridge.zip`, the latest
persisted source when the optimization was made. Both the rotating internal
search batch (`--train`) and independent fixed graph-fit/held-out batches
(`--graph-train`, `--valid`) remain separate.

## Exact-model fast paths

1. **Sparse Ridge append evaluation.** Candidate count/density vectors often
   have zero entries in most diffusion rows. When at most 25% of entries are
   nonzero, compute the Ridge covariance of candidate `z` and an existing
   predictor `x` from

   `sum_{z_i != 0}(z_i*x_i) - sum(z)*mean(x)`.

   For this candidate, the new work scales with `nnz(z)*#rules` rather than
   `#rows*#rules`. The same Schur-complement Ridge update still gives the
   Pearson of the augmented design. Dense candidates take the original
   centered path for numerical stability. The coefficient units, feature
   normalization, penalty, response, and acceptance rules are unchanged.

2. **Faster dense Gram construction.** Build `X_center^T X_center` using

   `sum(x_i*y_i) - n*mean(x)*mean(y)`.

   This moves mean subtraction out of the innermost loop. It is algebraically
   the same Gram matrix, subject to ordinary floating-point rounding.

3. **Skip redundant final-rule RHS evaluations.** If a set of beam children
   shares every earlier rule and the last **matching pattern**, variations
   in the last replacement output cannot change the current feature columns.
   Reuse its exact fitted Pearson. An output-cap guard keeps the legacy
   rewrite error checks for potentially oversized outputs. These variants
   are still retained separately so their different outputs influence later
   rule searches.

4. **Profiler counters.** With `-d:lensProfile`, log the number of sparse
   versus dense candidate updates and the number of last-RHS scores reused.
   Retain the existing mining, screening, Ridge, suffix, and step timing logs.

## Build and compare

```bash
nim c -d:release --opt:speed cms.nim
./cms --self-test
# Use your existing dataset and settings. Output directory must be different
# if you compare two runs.
./cms --corpus=github-code-dataset.txt --out=lens_fast_test \
  --train=8 --graph-train=4 --valid=2 --steps=100

# Separate profiling build: costs extra instrumentation overhead
nim c -d:release -d:lensProfile --opt:speed cms.nim
./cms --corpus=github-code-dataset.txt --out=lens_fast_profile \
  --train=8 --graph-train=4 --valid=2 --steps=10
```

Check `[profile] mine/screen/exact`, `[profile-rank] suffix_encode/ridge`,
`[profile-evaluate] encode/ridge`, `[profile-candidates]`, and `[profile-step]`.
The true end-to-end speed-up depends on how sparse selected patterns are and
how much time goes into actual rewrites versus Ridge calculations. Do not
interpret mathematical equivalence as a measured speed-up.

### Optional speed/quality controls (do change workload)

- `--plot-every=5` reduces plotting overhead without affecting fit selection
  (plot cadence only).
- `--delta-suffix=1` enables the existing omission-aware suffix shortcut
  for sibling replacements, but **benchmark and validate it against the
  default `0` on your dataset** before relying on it.
- Reducing `--screen-candidates`, `--beam`, `--mine-budget` or noise steps
  makes training faster but may change learned rules and accuracy.

## Verification limitations

The mathematical identities were independently checked on 240 randomized
NumPy cases (up to 900 rows and 28 existing features), with tiny
floating-point differences. The existing Nim `--self-test` also checks that
cached and uncached 4-way RHS evaluation yield equal Pearson; extra sparse
append-versus-full-refit tests were added. The container has no Nim toolchain
and no training corpus, therefore it was **not possible to compile the Nim
code or measure a full Lens training speed-up** here.
