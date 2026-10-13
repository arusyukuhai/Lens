# Lens v7 — multiscale, balanced readout and gated refresh

## Main diagnostic

The v6 held-out curve reached approximately `-log2(1-rho)=4`, which corresponds to Spearman `rho=0.9375`. The issue is a plateau with dips and a poor correspondence between high search scores and a consistently rising protected held-out score—not an absence of any generalization.

## Core readout change

For a single noisy path with `m` snapshots sorted by corruption fraction, the former fit used only adjacent target ranks `(j, j+1)`. It minimized squared errors on those pair differences, while the final metric was Spearman across **all** snapshots. Many candidate features do not change between adjacent stages.

v7 uses distances `1,2,4,8` (configurable), leading to O(m log m) comparisons per trajectory rather than O(m²) all-pairs. For each pair `(j,j+d)`, the regression target is the normalized rank distance `d/(m-1)` (or the exact rank difference with ties). This reduces the disconnect between adjacent-stage fitting and full-sequence ranking.

Each corruption path has unit total weight in the regression objective: for each path of `E_c` edges, every edge is weighted `1/E_c`. An unbounded 4096-stage path can no longer dominate a 32-stage path just because it has more edges.

`fitPrimal` and cached one-column `fitAppend` use **the same edge set and weighting**, including scaled ridge diagonals. The default arguments of their low-level APIs remain legacy adjacent/unbalanced so existing direct consumers/tests retain their old behavior; the v7 **trainer** opts into `--balanced-rank-fit:true --rank-scales:4`.

## Safer ongoing adaptation

v5's frozen champion removed unintended weight drift on rejected iterations, but it also meant a useful existing rule set could never revise its readout. Every `--readout-refresh-every:4` iterations, when no new rule set was promoted, v7 proposes a half-step towards the newly trained readout using the SAME rule list. The proposal is accepted only if BOTH independent rotating Acceptance and Anchor improve over the champion. Rejected proposals leave the champion unchanged. `readoutRefreshed` is logged and appended as TSV column 13, without shifting legacy columns 1–12.

**Note:** multiple comparisons on rotating validation pools still risk selection overfitting. Neither Anchor nor Acceptance is truly unseen after many training iterations; protected Held-out is never fed into the algorithm. A once-only final evaluation on a separately held repository is still recommended.

## Cost / exploration

- Two independent corruption orders per source chunk by default, with at most 128 recorded stages per order, reducing per-chunk sample memory from O(L²) to O(L·min(L,128)). This is deliberate stage subsampling, not the full historical all-stages experiment.
- More diverse regression-only replay chunks, but only 32 stages per replay path; replay chunks remain absent from n-gram mining.
- `--mine-start-stride:2` samples n-gram **starting positions** while retaining wildcard-gap constraints. It changes candidate frequencies and may miss a useful pattern. To revert to exhaustive mining, use `--mine-start-stride:1`.
- Eight source chunks per rotating acceptance anchor, six per rotating Acceptance and four per search batch. An increased number of independent source chunks makes one lucky passage less decisive.

## Direct ablation experiments — **fresh runs with identical seed and corpus**

1. **Full new design:** `--rank-scales:4 --balanced-rank-fit:true --mine-start-stride:2 --readout-refresh-every:4`
2. **Legacy rank objective only:** `--rank-scales:1 --balanced-rank-fit:false --mine-start-stride:2 --readout-refresh-every:4`
3. **Exact mining:** `--mine-start-stride:1`, other flags like run 1.
4. **No readout refresh:** `--readout-refresh-every:0`, other flags like run 1.

Keep `--seed`, train partitions, and monitoring splits the same; do not choose a winner after repeatedly inspecting the protected held-out set. For a statistically meaningful comparison, evaluate winners once on a separate repository/source-level holdout that is never consulted during development.

## Compatibility

- Same rule representation and model JSON, checkpoint format 3 (extra optional config fields are merged, including nested MiningConfig.startStride).
- v6 checkpoint resumes without reordering source chunks; the champion's weights remain frozen until validated promotion or readout refresh.
- With old snapshots in a resumed run, the history still mixes old and new training policies; a fresh v7 run is recommended for scientific comparison.

## Validation commands

```
nim c -r -d:release --opt:speed test_v7_multiscale.nim
nim c -r -d:release --opt:speed test_fast.nim
nim c -r -d:release --opt:speed test_miner_fast.nim
nim c -r -d:release --opt:speed test_v6_optimization.nim
nim c -d:release --opt:speed cms.nim
```

**Status:** Static source checks and differential mathematical consistency checks can be performed in the current authoring environment, but Nim compiler / gnuplot execution and real corpus benchmarks are not available here. Do not treat the hypothesized accuracy/speed gains as experimentally validated.
