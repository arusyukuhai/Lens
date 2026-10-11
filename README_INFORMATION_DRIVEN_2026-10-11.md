# Lens – Information-Guided BIPS / Sparse Outer Search (2026-10-11)

The baseline is the **Relational BIPS** version: `lens_bips_relational.py` (paired repairs and downstream-consumer goals), Linear Regression + Pearson, Native CPU evaluation, historical inner string GA, and HOF/checkpoints all remain in place.

## Why this change

OLS rule coefficients and firing frequencies are **not** an attribution of rule usefulness. A downstream rule may require a producer with zero readout coefficient. Rules can also have independent firing vectors (phi near zero), while each is indispensable to explain the target cleanliness. Conversely, high phi does **not** imply a rule is useless: true multi-stage pipelines naturally coactivate. We **never optimize phi toward zero or one**.

## New diagnostic: refitted OLS drop-one attribution

For selected nonconstant, nonzero-readout columns, center training-only features and labels. Fit ordinary OLS with a free intercept. For each rule `i`, refit OLS without its feature column:

```
I_i = MSE_train(OLS(X without i)) - MSE_train(OLS(X))
```

The quantity is nonnegative up to rounding and measures **unique predictive contribution conditional on other selected features**, NOT a structural/causal loss. Correlated duplicate features receive near-zero unique contribution. Independent essential binary features all receive positive contribution. At each guided iteration, unused slots with low `I_i` are favored over important ones.

Train rows exclude the final, independently held-out **source original**. This heldout original is not used for importance-based replacement. The OLS computation remains continuous-target *Linear Regression*; no pairwise Ridge, Spearman, or φ regularization is reintroduced.

## Producer protection and program-level interventions

A supplementary **potential bridge risk** checks whether a rule emits literal tokens consumed by a later rule; backward propagation of downstream predictive importance conservatively protects indirect producers, including zero-weight rows. This is a static reachability proxy, **not causal proof**. It intentionally does not attempt to interpret every wildcard substitution/hidden operation. A zero-risk result is not proof of removability.

When `--ablation-every N` is enabled, candidates are now selected based on information importance and bridge importance (plus low-cost controls). The existing actual counterfactual removes a whole rule, **reruns Replacer**, refits the readout and measures the change in heldout Pearson. This much more expensive measurement includes changes to later rule firing, so it must be kept distinct from feature-column importance. It is logged as a *diagnostic*, not used to pick candidate replacement slots (avoids heldout leakage).

## φ coefficient map

Every `--information-every` generations:

* `<prefix>_g000010_importance.csv`: every row's train refit importance, potential bridge score, combined conservative deletion risk, firing rate.
* `<prefix>_g000010_phi.csv`: the top variable, relevant rules' **binary firing** Pearson/φ correlation matrix, on training originals only.
* `<prefix>_g000010_phi.png`: optional heatmap if matplotlib is installed. A φ value of `+1` or `-1` can mean redundancy or required sequencing/mutual exclusion. Interpret alongside program-level interventions, not as a penalty.

No dense 1500×1500 matrix is allocated by default; only `--information-top` (32) is visualized, and CSV attribution covers all rules.

## Larger samples, smaller outer optimizer

`--samples 1000` is supported with CPU Native: it now matters much more than `--corpus-chunks`, which only sets available corpus entries. A small outer population with much larger input sample count reduces the risk of selecting spurious correlations in tiny training batches. **Samples from the same original are correlated**; 3 originals × 1000 corruptions are NOT 3000 independent text examples. To test generalization, increase original diversity too, e.g. `--cases 4` or more, plus a fixed external audit corpus.

`--samples-by-length --samples-length-cap 1000` optionally sets trajectories per case to the **median effective source length**, capped at 1000. This is one shared size per original (not separate allocation by each case), to preserve Native/MPS batch capacity.

`--outer-search single-trial` is a population-free *outer search concept*, using **two evaluation slots only**: one current model and one candidate. It forces guided training, one elite, zero HOF/QD injections and zero random immigrants. Odd generations try individual-rule changes; even generations try multi-rule BIPS first, with guided fallback. It does **not** use structural GA crossover/mutation. The **inner reconstruction GA remains enabled**. The HOF archive and checkpoint format remain compatible; stored candidates can still be accessed, but a resumed run trims the live outer population to two. Compared with wider beam-search, one-trial is an **exploratory option, not guaranteed higher accuracy**.

## Suggested exploratory settings

```bash
python main.py \
  --backend cpu-native --cpu-threads 8 --blas-threads 1 \
  --learning-mode guided --outer-search single-trial \
  --cases 4 --samples 1000 --ols-max-features 64 \
  --bips-share 0.35 --bips-relational-share 0.5 \
  --information-selection --information-every 10 \
  --information-top 32 --information-output output/rule_info \
  --ablation-every 20 --ablation-rules 2 \
  --inference-mode ga --hof-size 4096 --hof-eval 8 \
  --generations 200
```

For wider comparisons replace `--outer-search single-trial` with `--outer-search population --population 8 --elites 2 --qd-inject 0 --hof-inject 0`.

**Performance warning:** 1000 trajectories × 4 originals × 1500 rules can cost dramatically more Native execution than 100 × 3; use small population / two-slot trial first and watch wall time and memory.

## Verification and limits

Tests include a 3-feature pairwise-phi-zero example with 3 strictly positive leave-one-feature-out attribution values, duplicated high-phi firing vectors with zero unique contribution, a zero-coefficient potential bridge, and whole-source exclusion. Large-sample CPU Native (3000 rows), live inner GA, single-trial mode, program ablation and readout all ran in smoke tests.

Claims NOT supported without new experiments: higher heldout Pearson, better inference restoration, calibration under distribution shift, causal validity of the static bridge proxy, or that φ=0 should universally be preferred. The structural intervention is limited to a few candidates because doing it for all 1500 rules would be expensive.
