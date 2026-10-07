# v63 — steady-state diff3 DE / live progress / low-latency inference

v63 keeps the v61 search rule itself intact: population is fixed (default 450), every target `a` receives the categorical `b -> c` diff3 delta, and `a'` replaces `a` only when **both Spearman and inference recovery strictly improve on the same freshly rotated cases**.

## Live progress

A sweep no longer has to finish before useful diagnostics appear. While the 450 targets are processed, the trainer prints roughly every 5 seconds:

```text
sweep=    12 step=137/450 fit=+0.812345 inf=0.417 bestFit=+0.934210 bestInf=0.583 accept=9/137 (6.6%) speed=1.84 target/s ETA=2.8m
```

`training_saturation.png` is refreshed roughly every 30 seconds with an actual within-sweep point. The x axis is elapsed wall time. It plots current Spearman / MA / best-ever and current inference / MA / best-ever. Intermediate rows are also written to `fitness_history.csv` with `history_kind=progress`; full sweep rows use `history_kind=sweep`.

Full `training_fitness.png`, embedding and profiler plots are regenerated at sweep boundaries; intermediate updates regenerate only the saturation plot so plotting does not become a major bottleneck.

## Speed changes

The largest change is the inference allocator. The old generic budget often transformed the pairwise DE request into roughly `2 candidates × 24 sequential inner generations`. v63 preserves candidate diversity and reduces sequential rounds first; with the standard pair budget this is normally `16 candidates × 2 generations`. Equivalent trajectory work remains under the same cap, but GPU/CPU round-trip latency is much smaller.

Other hot-path changes:

- common-snapshot full-population audits are diagnostic only and now run every 10 sweeps instead of every sweep;
- donor `b,c` selection is O(1) rejection sampling instead of allocating a 449-element exclusion list per target;
- one live-root list is reused for the pair evaluator and inference evaluator;
- copy-on-write identical Rule objects immediately skip diff3;
- if `a == b`, diff3 returns `c` directly; if `a == c`, it is already applied;
- conflict resolution reuses the selected valid `a/b/c` Rule directly instead of cloning + sanitizing it;
- the old full-genome structural equality scan after every proposal is removed from the normal hot path.

On the included small CPU benchmark used while checking this revision, the previous v62 package took about 5.68 s wall time and v63 about 3.22 s for the same 20-genome/32-rule one-sweep command (startup included). The sweep itself dropped from about 0.58 s to about 0.27 s. This is only a smoke benchmark; the exact gain on a 450×1500 MPS run will depend on Metal dispatch and corpus length.

## CLI cleanup

The active CLI now exposes only settings used by the DE trainer. Legacy GA/NSGA-II/HoF/QD/linkage/plateau controls were removed, including `--elites`, all `--hof-*`, all `--plateau-*`, `--specialist-rate`, all `--qd-*`, all `--linkage-*`, `--operator-*`, `--tournament`, old outer `--crossover-rate`, `--diff-merge-share`, `--mutation-rate`, `--immigrant-rate`, and the old rolling-generation controls.

The requested DE behavior is fixed policy rather than hyperparameter sprawl:

```text
case refresh               3 per target
inference case refresh     3 per target
extra two-point crossover  0.30
extra structural mutation  0.35
simple Rule splice         0.20
splice max rows            8
common audit               every 10 sweeps
progress print             about every 5 s
progress saturation plot   about every 30 s
```

Inference search population/generation/elites/ensemble/budget internals and embedding mutation/crossover rates are likewise no longer exposed as CLI knobs. Rank-ridge remains fixed to the current `pairwise-all` readout.

Run `python main.py --help` for the remaining active options.
