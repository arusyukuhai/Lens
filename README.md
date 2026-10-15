# Lens: Dual-Grid Power Diffusion (Coarse → Fine → Periodic Evaluate)

**Starting point:** `Lens_CoarseFine_EvalPower_Periodic.zip`. The Coarse/Fine
ranking scheme, separate evaluation power, `--eval-every`, firing-density
features, Ridge/OLS, fingerprint filtering, and all rule rewriting are kept.

## New noise-count schedule

For a chunk of original length `L` and any finite decimal exponent `a >= 1`,
select the **union** of the two schedules, including both endpoints:

1. `floor(k^a)` for `k = 1, 2, ...` up to `L` (dense near zero), **plus**
2. `floor(k * L^(1-1/a))` for `k = 1, 2, ...` up to `L` (even coverage).

All results are capped to `L`, de-duplicated, and sorted. No level above `L`
is generated. The clean 0-noise row and fully noised `L` row are both present.
Both sequences use the same existing deterministic noise permutation: *only*
which intermediate snapshots get recorded has changed, not the noise applied.

`a = 1.0` still includes **all** `L + 1` noise levels, because both sequences
are `0,1,2,...,L`. As `a` grows, the number of levels is approximately
`O(L^(1/a))`, with up to about **twice** the earlier schedule's level count.

Examples for `L = 25`:

- `a=2`: first `0,1,4,9,16,25`; second `0,5,10,15,20,25`;
  union `0,1,4,5,9,10,15,16,20,25`.
- `a=3`: first `0,1,8,25`; second `0,8,17,25`;
  union `0,1,8,17,25`.

The union applies identically to `--coarse-power`, `--fine-power`, and
`--eval-power` because their row generation already uses one shared function.
`--eval-every=N` is unchanged: the Fine selection and detailed Train/Validation
still run at the first/last generations and every Nth generation.

## Usage

```bash
nim c -d:release --opt:speed -d:lensProfile -o:lens_train cms.nim
./lens_train --self-test
./lens_train --coarse-power=2.0 --fine-power=1.5 \
  --eval-power=2.0 --eval-every=5 --fine-top=12 \
  --delta-suffix=1 --steps=100 --out=lens_dual_grid
```

To run full diffusion on Fine use `--fine-power=1.0`. For a fair A/B against
the previous schedule, use separate output directories and fixed seed, corpus,
training/validation chunks, and initial rules. The new schedule changes
training sample selection, so it can change the learned rule trajectory.
The legacy `--noise-steps=N` path is unchanged and still requires all powers
set to `1.0` when it is used.

## Validation

`--self-test` includes exact short schedules, near-infinite exponents,
monotonicity, uniqueness, endpoints, and deterministic correspondence with
full diffusion for multiple fractional exponents. `verify_dual_grid.py`
independently checks the formulas with Python and scans for expected Nim
wiring. **Nim compilation is unavailable in the artifact runtime**, so
compilation, self-test execution, and wall-clock improvements have not been
verified here.

## Coarse/Fine/validation selection semantics

- Every generation: the Coarse grid is used to mine and score Add/Replace.
- On `--eval-every` checkpoints, the top `--fine-top` candidates per move type
  are reassessed with the Fine grid; the chosen move comes from its Fine score.
- On other generations, the selection stays on the Coarse score and there is
  no periodic Train/Validation diagnostic pass.
- At each checkpoint, `--eval-power` controls a separate diagnostic training
  and held-out grid (defaults to `--fine-power`). Fixed held-out uses the
  same deterministic corruption pattern, with no fitting on held-out rows.
- `--eval-every=1` invokes Fine/diagnostics each generation; `--eval-every=5`
  evaluates first, fifth, tenth, final generations.
- `rules.tsv` is saved after each generation; plots only update at checkpoints.
- The optimization affects observation density and thus search selections:
  changing it can change validation Pearson and learned rules. It does not
  alter Ridge/OLS mathematics, token noise values, the matching engine or
  firing-density `matches / originalLength` features.

### Preserved fast paths

Fingerprint8192, prefix state/vector reuse and optional `--delta-suffix=1`
continue working unchanged. `--noise-steps` remains a mutually exclusive
legacy row-sampling flag (when set, all powers must be 1.0).
