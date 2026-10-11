# Lens / Replacer — Directed Rule Learning + Pearson, with inference GA retained

2026-10-11 release, based on `Lens_Spearman_CPU_diffusion_redesign_20261010.zip`.

## What actually changed

**Outer rule learning: default `--learning-mode guided`.** Each outer generation:

1. Evaluate the current genome population using the existing CPU-native batch engine (MPS fallback remains available).
2. Fit a **standard, unregularized ordinary-least-squares linear regression** of exact Hamming cleanliness `1 - corrupted_bytes / source_bytes` on the existing per-rule execution features. **No pairwise rank margins and no Spearman** are used in the new default readout. Feature screening is performed using training data only, and defaults to at most 32 features (`--ols-max-features 32`; historical flag `--rank-max-features` remains an alias).
3. Keep an entire **source chunk** aside from OLS fitting, and calculate **Pearson correlation** on that source. The independent source chunk, not shuffled corruption rows from the same original, defines the genome's scalar fitness. Training-source per-case correlations are still recorded as diagnostics; the scalar fitness is computed from the excluded source(s) only.
4. For the current leader, mine literal, fixed-length correction rules from **only the OLS training source chunks**. Generate candidate substitutions using the observed difference between clean and corrupted bytes with 2–5-byte context windows.
5. Score candidate triggers with the covariance of their activation proxy against **the current regression residual**. Discount patterns that also appear in clean controls, and penalize longer rules. This is a **cheap proxy**: for sequential Replacer with hidden state the true improvement cannot be deduced from the proxy alone.
6. Replace low-utility rule slots in copy-on-write children with the best supervised candidates. The next outer population evaluation re-runs the **real sequential Replacer** and accepts/retains high-fitness genomes via existing elite/HOF/QD/Pareto selection. No crossover or random rule mutation is required in `guided` mode when supervised proposals exist.
7. Optionally inspect rule influence using **full counterfactual rule ablation** (`--ablation-every 10 --ablation-rules 2`), which re-runs Replacer and refits the readout after each deleted rule. This is an audit/diagnostic, not a formally proven causal effect on unseen text. Pairwise phi-coefficients among frequently firing rules are also reported for dependency diagnostics.

**Inner string reconstruction: default `--inference-mode ga`.** The existing `run_string_inference_ga()` is preserved, with its candidate population, elites, mutation, crossover, Pareto-ensemble scoring and CPU-native fused evaluation. The optional `--inference-mode direct` is **only an A/B experimental baseline**, not the default. Reconstruction accuracy is still computed after candidate selection using the withheld clean text.

**Outer/inner optimizations and features retained:** CPU-native batched evaluation, caching, persistent inference context, Hidden Memory, prior diffusion redesign, HOF and checkpoint support. The checkpoint contains `readout_mode`, `learning_mode` and `inference_mode`; when resuming an old pairwise model under Pearson the cached fitted scores are invalidated.

## Recommended usage

```sh
python -m pip install numpy scipy threadpoolctl
python main.py \
  --backend cpu-native --cpu-threads 8 --cpu-model-batch 32 --blas-threads 1 \
  --learning-mode guided --readout-mode linear-pearson \
  --inference-mode ga --diffusion-mode dual \
  --cases 3 --samples 100 --ols-max-features 32 \
  --population 96 --rules 1500 --hof-size 4096 --hof-eval 16
```

This example uses a smaller **active** population for quicker research loops; the HOF storage capacity remains 4096. Using at least 3 source chunks supplies two training chunks and one excluded source for Pearson; with `--cases 2`, mining is trained on one chunk and generalization may be unstable. `--cases 1` is supported for debugging only and falls back to within-chunk holdout, which should not be interpreted as independent-source generalization.

For pure genetic outer baseline, use `--learning-mode ga`. For a mix of directed offspring and historical genetic offspring, use `--learning-mode hybrid --guided-hybrid-share 0.15`. The inner restoration GA is independent of this setting.

For legacy rank comparisons, explicitly use `--readout-mode pairwise-all`. A checkpoint from the rank-based versions can be loaded under the new mode, but historical fitness values are **not numerically comparable**. The `--ridge-lambda` option is ignored by the new linear-pearson readout but remains available for the legacy options.

## Evaluation rigor and limitations

- Train/held-out source separation applies to **readout fitting and supervised candidate mining**. Structural model selection repeatedly observes the held-out Pearson, so it **is not a final unbiased test** of generalization. For a publication-quality score, evaluate a separate corpus that is never used in the outer loop, or use an additional untouched test split.
- Rule candidate scoring uses literal substring presence as a fast necessary-condition proxy. It is **not** a complete event trace of sequential rule execution, and proposed rules may turn out useless or harmful after a real run. These proposals are validated by the ordinary outer evaluation rather than trusted blindly.
- Complete counterfactual ablations are computationally expensive and run sparingly by default. The code does **not** yet provide fully incremental replay for arbitrary rule changes.
- Ordinary least squares uses the best 32 columns (by training-only univariate relevance), then solves unregularized OLS on those columns. This **feature-selection step is not ridge regularization**. Setting an excessively large feature limit relative to the training row count risks interpolation and unstable generalization.
- Pearson is sensitive to amplitude and linear calibration relative to Spearman, but *correlation alone* still does not measure absolute prediction calibration (an affine transformation can leave Pearson unchanged). For calibration assessment use independent MAE/RMSE on withheld examples.
- No claim is made that this version improves downstream language modeling performance, Mac training speed or reconstruction accuracy without empirical measurements on those exact tasks.

## Verification

```sh
python -m pytest -q
python main.py --self-test
```

Unit tests cover source-wise OLS holdout, the use of continuous 0..1 targets, training-only supervised rule mining, independent directed offspring, and a one-generation subprocess run using the **default inner GA**. Previous regression tests, including native CPU and diffusion redesign tests, also remain in the package.
