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
**New in this optimized revision:** Fine is used to rerank the Coarse shortlist
**EVERY generation**, independently of `--eval-every=N`. The latter controls
**only detailed train/held-out evaluation and plotted checkpoints** (always
first and last generation, plus every Nth generation).

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
- **Every generation**, Coarse produces candidate Add/Replace finalists; the
  top `--fine-top` candidates per move type are reranked on the Fine grid.
  Parent baselines and all move candidates are compared on **the same Fine
  rows**. Never mix Coarse and Fine Pearson to choose a rule.
- Between `--eval-every` checkpoints, the selection still uses Fine scores,
  but there is **no detailed Train/Validation diagnostic pass**.
- At each checkpoint, `--eval-power` controls a separate diagnostic training
  and held-out grid (defaults to `--fine-power`). Fixed held-out uses the
  same deterministic corruption pattern, with no fitting on held-out rows.
- `--eval-every=1` computes diagnostics each generation; `--eval-every=5`
  computes diagnostics at first, fifth, tenth, final generations. **Fine
  selection happens every generation in both cases.**
- `rules.tsv` is saved after each generation; plots only update at checkpoints.
- The optimization affects observation density and thus search selections:
  changing it can change validation Pearson and learned rules. It does not
  alter Ridge/OLS mathematics, token noise values, the matching engine or
  firing-density `matches / originalLength` features.

### Preserved fast paths

Fingerprint8192 and prefix state/vector reuse remain. **`--delta-suffix=1`
is now the default**, and adjacent-token-pair Fingerprint filtering is enabled
by default (`--pair-fingerprint=1`). Neither changes match counts. When
`--coarse-power == --fine-power`, the program shares identical Coarse and Fine
rows and prefix-state cache to avoid duplicate generation/re-encoding. `--noise-steps` remains a mutually exclusive
legacy row-sampling flag (when set, all powers must be 1.0).

## Optimization and knobs (this revision)

- `--pair-fingerprint=1` (default): an additional 8192-bit per-row bitmap for
  *adjacent literal pairs*, with no false-negative rejection. Helps only on
  patterns with neighboring fixed tokens; it does not help `A,-1,B` patterns.
  Set `--pair-fingerprint=0` to compare.
- `--delta-suffix=1` (default): reuse unchanged suffix features when swapping
  rules; `--delta-suffix=0` disables for a benchmark.
- `--eval-every=N` does **not** change Fine frequency. Keep `--fine-top` small
  enough that every-generation Fine rescoring remains affordable.
- `--coarse-power=2.0 --fine-power=1.5 --eval-power=2.0 --eval-every=5` is an
  illustrative configuration, not a measured optimum.

The Nim compiler was **not available** in the creation environment. Python
structural/formula tests ran, but runtime speedups and Nim compilation are
unverified. See `verify_pair_fingerprint.py` and `verify_eval_schedule.py`.
