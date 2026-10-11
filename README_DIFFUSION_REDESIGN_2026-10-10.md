# Lens Spearman: diffusion curriculum redesign (2026-10-10)

This release is built on top of `Lens_Spearman_CPU_batched_fast_20261010.zip`. It keeps the CPU-native outer evaluator, inference GA, crossover, HoF, checkpoint loading, hidden-memory implementation and Spearman scoring. It deliberately **changes the outer training corruption distribution and default rank-Ridge pair construction**; fitness values from old and new curricula are not directly comparable.

## Why change the old noise process?

The old `make_noise_trajectory()` walked a single random permutation of source positions and emitted `S` snapshots with `k = round(K*s/(S-1))`, where `K` is the largest number of corrupted bytes. This repeats exactly the same input every time two timesteps round to the same `k`. For example, a 20-byte source with `max_noise=0.35` admits at most 8 different Hamming-damage levels (`k=0..7`), so `--samples 100` used to contain only about 8 distinct strings. That wasted 92 of 100 evaluation slots.

Additionally, the old rank pair graph connected each row with its neighbor in sorted rank. Under many tied targets, almost all edges had **zero margin** and failed to teach the model which of two differently damaged strings should score higher.

## New outer corruption curriculum (`--diffusion-mode dual`, default)

`n` is the input length, `S = max(3, samples)`, and `K = max(1, min(n-1, round(max_noise*n)))` for `n >= 2`. `n=1` supports one corruptible byte. `n=0` is untrainable and normally filtered out by `--min-chunk`.

* If `S <= K+1`, no target noise level is repeated; emit one genuinely cumulative single corruption path, evenly spaced over `k=0..K`. This avoids unnecessary branches and extra cost for long strings.
* If `S > K+1`, emit **one clean control** and distribute the remaining `S-1` examples as evenly as possible among the positive Hamming-damage levels `k=1..K`.
* At every repeated severity, alternate two independently randomized channels: **scatter** (uniform permutation of damaged positions) and **burst** (one contiguous expanding interval of damaged positions). A/B counts differ by at most one per level. If there are still more requested examples, create more independent paths per channel rather than repeating any already emitted snapshot.
* Each path itself monotonically adds exactly one newly damaged position, and the proposed replacement is guaranteed different from the clean byte. Outputs are sorted by actual Hamming damage, not by path ID. As a result, neighboring output rows do not necessarily belong to the same path, but targets remain nonincreasing.
* Guard against accidental identical noisy strings at the same `k`, by adjusting a corrupted byte while retaining exact damage. Uniqueness is guaranteed while enough distinct corrupted strings exist in the finite byte-string state space; extremely short input with an exceptionally large sample budget can still exhaust that space.
* Every target is **measured exact Hamming cleanliness** `1-k/n`, never a nominal diffusion timestep. Independent realizations at the same damage have identical target labels by design.

**Example:** `n=20, --max-noise 0.35, --samples 100` now generates 1 clean string and 99 different noisy strings with noise severities 1..7; each severity includes scatter and burst alternatives. The old version generated only ~8 distinct strings.

**Important:** Both channels use byte substitution, but their *spatial pattern* differs; "two corruption types" does not introduce insertions/deletions or change the input length, which would make exact Hamming targets ambiguous.

## New target-aware rank pairs (`--rank-pair-mode severity`, default)

For each case, group examples by distinct cleanliness. For each adjacent pair of *distinct noise levels*, cyclically match examples, creating `max(group_A_size, group_B_size)` edges. Every example participates in meaningful cross-severity comparisons. The target margin remains normalized average-rank difference, preserving the pairwise Ridge approach and Spearman reporting metric.

The graph contains O(S) edges, not all O(S²) pairs. It avoids spending most of the readout fitting effort on zero-target-margin ties when a short string yields many independently corrupted copies.

Two independently selectable compatibility flags:

```sh
# New default training distribution and readout:
python main.py --backend cpu-native --diffusion-mode dual --rank-pair-mode severity

# Previous noise path and previous tie-edge graph, useful for ablations:
python main.py --backend cpu-native --diffusion-mode legacy --rank-pair-mode legacy

# Diffusion redesign alone (keep old rank pair graph):
python main.py --backend cpu-native --diffusion-mode dual --rank-pair-mode legacy
```

The checkpoint format stays compatible. Diffusion mode and rank-pair mode are newly recorded in checkpoint metadata, and a warning is printed if a resumed checkpoint explicitly records a different mode. Previous checkpoints did not record either field; choose legacy flags for an old-style experiment. A rolling evaluation set is always regenerated when resuming, as in the base release.

## Validation / tradeoffs

- `python -m pytest -q`: 120 passed, 4 skipped, 10 parameter subtests passed. GPU-specific tests skipped when their runtime was unavailable.
- `python main.py --self-test`: passed.
- CPU-native two-generation learning smoke test with rank fitting, native outer evaluation, inference GA and checkpoint enabled: passed.
- Exact Hamming corruption and target order checked for input lengths `1,2,3,4,5,9,20,96,100,1000` with 100 samples, including tests proving nonduplicate outputs and scatter/burst diversity.
- This is a **training-curriculum change, not a claimed speedup**. For short strings, generation cost increases mildly because more distinct inputs are actually evaluated rather than cheap duplicates. This higher cost is intentional: the previous cheap duplicated rows added no useful data. With `--samples <= K+1`, the original single monotone chain is preserved.
- Short sequences have intrinsically fewer possible Hamming noise severities. Two different 1-byte corruptions have the same target; no algorithm can manufacture extra ground-truth severity levels without changing the task. Spearman ties remain mathematically necessary.
- The statistical benefit (generalization and stability) should be checked against a fixed held-out corpus; successful unit tests do not establish improved model quality on unseen text.

## Recommended invocation

```sh
python -m pip install numpy scipy threadpoolctl
python main.py \
  --backend cpu-native --cpu-threads 8 --cpu-model-batch 32 --blas-threads 1 \
  --diffusion-mode dual --rank-pair-mode severity \
  --cases 2 --samples 100 --max-noise 0.35 \
  --population 450 --rules 1500 --hof-size 4096 --hof-eval 32
```

If you use `--corpus-chunks`, that changes loaded corpus volume, not `--cases * --samples` evaluated per generation. The diffusion schedule now uses the latter budget more effectively.
