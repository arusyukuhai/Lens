#!/usr/bin/env python3
"""Host-only microbenchmark for v59 all-sample pairwise readout."""
import time
import numpy as np
import main

rng = np.random.default_rng(3)
cases, samples, p = 3, 36, 1500
n = cases * samples
case_ids = np.repeat(np.arange(cases, dtype=np.int32), samples)
sample_ids = np.tile(np.arange(samples, dtype=np.int32), cases)
clean = np.tile(np.linspace(1.0, 0.65, samples), cases)
baseline = np.zeros(n, dtype=np.float64)
x = rng.poisson(0.08, size=(n, p)).astype(np.float32)
x[:, :32] += rng.integers(0, 2, size=(n, 32)).astype(np.float32)
train = (sample_ids % main.TRAIN_MOD) == 0
legacy_y = clean[train] - baseline[train]

main.fit_dual_ridge(x[train], legacy_y, 4.0)
main.fit_pairwise_rank_ridge_all(x, baseline, clean, case_ids, 4.0)
reps = 200

t = time.perf_counter()
for _ in range(reps):
    main.fit_dual_ridge(x[train], legacy_y, 4.0)
legacy = (time.perf_counter() - t) / reps

t = time.perf_counter()
for _ in range(reps):
    main.fit_pairwise_rank_ridge_all(x, baseline, clean, case_ids, 4.0)
rank = (time.perf_counter() - t) / reps

hi, lo, _ = main.case_adjacent_rank_pairs(clean, case_ids)
touched = len(set(hi.tolist()) | set(lo.tolist()))
print(f"legacy_exact_36: {legacy*1000:.3f} ms/genome")
print(f"v59_pairwise_all: {rank*1000:.3f} ms/genome")
print(f"ratio: {rank/legacy:.3f}x")
print(f"pairs: {len(hi)}, trajectories participating: {touched}/{n}")
