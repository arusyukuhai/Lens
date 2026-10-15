# Lens: pairwise geometric-mean Ridge experiment

This release starts from the preceding **Lens_Ridge_BeamSearch** implementation.
The new default `--geometric-pairs=1` adds a nonlinear interaction feature
`g_ij = sqrt(count_i * count_j)` for **every unordered pair of distinct rules**.
The original counts remain in the model.  We do not introduce `i=i` square
features because `sqrt(x_i*x_i) = x_i` for nonnegative match counts.

The fitted model is

`prediction = intercept + sum(beta_i * count_i) + sum(gamma_ij * sqrt(count_i * count_j))`.

Ridge normalizes **each design column** by its sample standard deviation,
fits an unpenalized intercept and penalizes the standardized coefficients.
This is explicit feature-map Ridge (not kernel Ridge); the nonlinear feature
map is fixed while the regression remains linear in its learned coefficients.

## Correct add/delete behavior

- **Add candidate:** Adding rule `z` contributes `z` and **all** interactions
  `sqrt(z * old_rule_j)`. A Ridge block-Schur update evaluates the exact
  in-sample Pearson **after jointly fitting this entire feature block**.
  The block score is not the old one-column FWL statistic. With
  `--ridge-lambda=0`, exact expanded OLS refitting is used instead.
- **Delete/replace:** Dropping original rule `j` removes its count feature
  **and every pair that touches `j`**. For positive lambda, grouped removal
  is evaluated by inverse-Gram block formulas, measuring the actual fitted
  Pearson of the reduced fixed design. For lambda zero, it refits OLS.
  This identifies the weakest *rule*, not the weakest feature column.
- **Mid-chain replacement:** Inserting a different rule changes outputs for
  later rules. Candidate screening is still heuristic, but acceptance is
  based on full sequential re-encoding and re-fitting of the complete
  expanded design. Failed replacement chains are rejected as before.
- **Held-out prediction:** Applies the training coefficients to exactly the
  same ordering: first all rule counts, then interactions in `(i,j)` order
  with `i<j`. No fit is performed on the held-out samples.

## Build and run

```bash
nim c -d:release --opt:speed cms.nim
./cms --self-test
./cms --corpus=github-code-dataset.txt --out=lens_geometric_run \
  --ridge-lambda=0.1 --geometric-pairs=1 --max-rules=24
```

For an apples-to-apples ablation, run the same build with
`--geometric-pairs=0` and a **different `--out` directory**.  The same corpus,
noise seed, train/held-out split, number of steps, and other hyperparameters
should be used. The geometry changes rule selection, so final rules need not
match. The prior linear-Ridge ZIP is also kept separately.

With `p` rules the regression has `p + p*(p-1)/2` feature columns: `p=24`
yields 300 columns; `p=48` yields 1,176 columns. For large `p`, dense
Gram construction and Cholesky solves may be expensive. Expect notably
higher memory and computation costs, especially in replacement trials.

Raw counts, deterministic in-alphabet noise before replacement, sequential
rewrites, CMS, width-4 beam, Add-versus-Replace selection, 15-point Pearson
moving averages (before `-ln(1-Pearson)`), and gnuplot are retained.

## Verification

The internal `--self-test` includes (in addition to pre-existing tests):

- expanded feature values and column count;
- Ridge block-add candidate Pearson against complete nonlinear re-fitting;
- grouped deletion loss against complete nonlinear re-fitting;
- OLS nonlinear add/remove fallback; and
- held-out feature ordering by predicting on the training sample itself.

The block-add and group-drop math was separately tested against NumPy on
synthetic data with `p = 1,2,3,5,8` and `lambda = 0.0001,0.01,0.1,1`.
**The runtime used to produce this artifact does not have Nim or gnuplot,
so Nim compilation/execution and graph generation have NOT been confirmed.**
