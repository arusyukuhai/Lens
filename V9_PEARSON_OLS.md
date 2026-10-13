# Lens v9: Pearson + ordinary least squares

## Exact change of objective

All active training and evaluation paths use Pearson *r* computed **separately per corrupted trajectory**, then averaged. The base fitting objective is ordinary unregularized least squares:

    y_i = beta_0 + sum_j beta_j * presence_j(x_i) + epsilon_i

`presence_j(x_i)` is exactly the original binary (0 or 1) presence flag. `y_i` is **the absolute fraction of corrupted positions** (not a rank). The intercept `beta_0` is **not** a rule and is not penalized. The default (`--balance-cases:false`) is **ordinary, unweighted OLS**. Optionally `--balance-cases:true` assigns equal total squared-error mass to each independent corruption path (weighted least squares). Neither uses ridge or other coefficient regularization.

Binary feature columns are often linearly dependent. v9 uses pivoted Cholesky on a normalized Gram matrix to select an independent subset. Coefficients of redundant features are set to zero rather than dividing by zero. The numerical tolerance is 1e-10 on the normalized residual Gram diagonal. The one-column append path uses Frisch-Waugh-Lovell and skips completely redundant additions. This strategy minimizes squared error on the independent feature span, but does **not** recover a unique coefficient vector when features are collinear.

The residual-based candidate screen now ranks correlation with the **numerical target residual** `true_noise_fraction - predicted_noise_fraction`, not the old rank residual. Validation, rotating Acceptance, Anchor and protected Held-out all use Pearson. Source-consensus checks also use per-source Pearson gain. The plotted metric is `-log2(1-Pearson)`; the logged raw values are Pearson. Spearman no longer takes part in training or selecting models.

## Expected behavior / limitations

- OLS optimizes squared error, not Pearson directly. With an unconstrained intercept, the *training pooled* OLS fit maximizes a related linear projection; **per-path averaged validation Pearson** is still a distinct criterion. Generalization is **not** guaranteed to improve.
- The prediction may lie outside [0,1] because it is an unbounded linear regression; if probability-like calibrated outputs are needed, measure MAE/RMSE separately rather than silently clipping for training.
- Ordinary OLS can be sensitive to rare indicator columns. The independent validation and source-consensus acceptance gates remain in place.
- `--ridge` and `--rank-scales` remain parsable for old saved options but are ignored by the v9 regression. `--balanced-rank-fit` is kept as an alias for `--balance-cases`.

## New run

    nim c -d:release --opt:speed cms.nim
    ./cms --corpus:github-code.txt --out:lens_run_v9 --iterations:100 --balance-cases:false
    nim c -r -d:release --opt:speed test_v9_pearson_ols.nim

Old checkpoints may be migrated: rules survive, but the former Spearman/rank-fit champion coefficients are re-estimated as OLS on the first training window. A new `training_v9.tsv` is created. Back up an old run directory before migrating in place; a fresh run allows a cleaner comparison.

Checkpoint format=4 stores the `intercept` and `objective=pearson_ols`; model format=2 does the same. The old `spearman_rank_ridge.nim` and old historic tests remain in the source bundle only as reference/archival material; execute `test_v9_pearson_ols.nim` and general functional tests for v9.
