# Lens — Exact All-Pairs Rank Surrogate (experimental)

This branch starts from `Lens_v8_Residual_Guided_Redesign.zip` and changes **readout training**, not the wildcard replacement semantics or protected-heldout evaluation. It should be treated as an ablation candidate, not a proven improvement.

## Motivation

Previous default `--rank-scales:4` fitted rank differences at distances 1,2,4,8. That can favor local progression and ignores many far-apart noise stages. The new default `--rank-scales:0` fits a **uniform all-pairs difference squared loss** within each trajectory, without O(rows^2) pair enumeration.

For a trajectory with `m` snapshots, feature `x_i` and normalized target rank `r_i=(rank_i-1)/(m-1)`:

    (1/m^2) sum_{i<j} ((x_i-x_j)-(r_i-r_j))^2
      = (1/m) sum_i ((x_i-mean(x))-(r_i-mean(r)))^2

The identity is exact for each feature-space prediction `x` (linear combination of rule firing indicators). Each independent trajectory has unit overall weight, regardless of its size.

This is **NOT** the exact Spearman objective (which ranks model predictions). It is a more global *linear least-squares surrogate*. For candidate selection, the measured metric remains actual Spearman on Search. Promotion still requires independent Acceptance and Anchor scores.

## Stable regularization

The old pairwise backend divided its ridge diagonal by its number of effective edges. With many edges, this made its relative penalty small. The **new all-pairs backend only** uses `lambda * diag(X'X)` so `--ridge:0.01` is a meaningful scale-free proportional shrinkage independent of the number of snapshots. The `--rank-scales:1..16` legacy backend retains the original behavior for reproducibility.

## Local readout tuning

`--readout-ridge-probes:3` (default) refits only the **final beam shortlist**, testing `0.25 * ridge`, `1.0 * ridge`, and `4.0 * ridge`. Each proposed readout is fitted on Train and ranked on Search. Neither Acceptance, Anchor nor Held-out is used for tuning. The acceptance/anchor promotion gates are unchanged.

For an ablation without tuning, set `--readout-ridge-probes:1`.

## Suggested fresh-run ablations

All cases must use the same seed, corpus, partition, and evaluation protocol; use different --out directories:

1. All-pairs + tuning: `--rank-scales:0 --readout-ridge-probes:3`
2. All-pairs without tuning: `--rank-scales:0 --readout-ridge-probes:1`
3. Historical multiscale: `--rank-scales:4 --readout-ridge-probes:1`

Because training objectives and ridge conventions differ, historic v8 checkpoint weights are not comparable to a fresh run. New runs are recommended. Old checkpoints remain readable and retain `rankScales` from their saved options unless explicitly overridden.

Use protected heldout solely for an end-of-run comparison, preferably over several **independent** corpus splits. Avoid repeatedly choosing hyperparameters using the same protected heldout.

## Tests

    nim c -r -d:release --opt:speed test_dense_rank.nim
    nim c -r -d:release --opt:speed test_v7_multiscale.nim
    nim c -r -d:release --opt:speed test_v8_architecture.nim

These compile commands have not been executed in the creation environment because Nim is absent there. An independent Python numerical test of the mathematical all-pairs equivalence *was* run successfully; that alone does not validate Nim syntax, end-to-end correctness, inference quality, or speed.

## Defaults and compatibility

The original `--mine-gap-samples:6`, `--screen-candidates:256`, `--screen-explore:48`, independent validation, frozen champion, dependency regrow options, and -log2 plot are unchanged. Only new-run `rankScales` default and the small end-of-beam readout tuning stage differ. The original mode is selectable via `--rank-scales:4 --readout-ridge-probes:1`.
