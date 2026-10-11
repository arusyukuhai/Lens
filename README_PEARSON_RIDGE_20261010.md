# Lens / Replacer — Pearson + ordinary Ridge revision (2026-10-10)

This is based on `Lens_Spearman_CPU_branching_diffusion_20261010.zip` and keeps the
CPU Native evaluator, branching diffusion, HOF, GA, and **no inference** path.

## Objective

- `--readout-mode ridge-all` is the **new default**. Uses **all** rule features
  and trajectories, with no rank-graph regression and no top-K feature filtering.
- Let X = rule activation feature matrix, B = per-trajectory baseline, Y =
  cleanliness target. Fit `(w, b)` to the residual `Y - B` with an **ordinary
  unweighted Ridge with unpenalized intercept**:

  `min_{w,b} sum_i (Y_i - B_i - X_i @ w - b)^2 + lambda * ||w||_2^2`

- Per-trajectory score: `P_i = B_i + X_i @ w + b`.
- Fitness: arithmetic mean of `Pearson(P[case], Y[case])` across cases. Constant
  vectors contribute 0 rather than NaN. Higher fitness is better.
- The default `--ridge-lambda 4.0` remains unchanged. `--rank-max-features`
  only affects the explicit legacy `pairwise-all` mode.
- `--readout-mode pairwise-all` and `legacy-holdout` are optional ablations.
  Even under those modes the fitness is Pearson, not Spearman.

## Launch

```bash
pip install numpy scipy threadpoolctl pytest
python main.py --backend cpu-native --cpu-threads 8 \
  --readout-mode ridge-all --ridge-lambda 4.0 \
  --diffusion-mode dual --cases 2 --samples 100 \
  --population 450 --rules 1500 --hof-size 4096 --hof-eval 32
```

Requires the corpus file specified by `--local-corpus` (default
`github-code.txt`), as in prior revisions.

## Resume / compatibility

- Old Spearman `.npz` checkpoint structure remains readable. The population,
  embeddings, and HoF are retained, but **old fitness, best-ever and historical
  series are discarded** when switching objectives. Resumed generation numbering
  is retained.
- Checkpoints include the new `readout_bias` and objective/readout metadata;
  older archives without these fields load with bias=0.
- If a pre-existing `fitness_history.csv` has a different objective, it is moved
  aside to `.before_pearson_only.csv` (or a numbered equivalent).
- Saved best genome JSON now includes `readout_bias` in addition to rule weights.
- Inference GA remains removed from the training path.

## Important caveat

This implementation uses **in-sample Pearson**, consistent with the prior
in-sample Spearman fitness and the decision to avoid costly holdout validation.
With many more rules than trajectories, the fitted Ridge readout can overfit:
a Pearson fitness close to +1.0 does **not** establish out-of-sample language
modeling ability. Increasing regularization, reducing effective degrees of
freedom, or measuring generalization separately are relevant future experiments.

## Verification

Run `python main.py --self-test` and
`PYTHONPATH=. pytest -q`.
The legacy differential-evolution and inference tests are preserved in
`docs_archive/` / `historical_inference_tests/` but excluded from default
pytest discovery because their APIs were removed in the earlier Spearman-only edition.
