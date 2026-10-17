# Lens PVE — shared-prefix cache / optimization edition (2026-10-17)

This edition builds on `Lens_PVE_1500_Moving_Average.zip` without changing the GA's 1500-rule genotype, 32 local mutation operators, 8 two-parent crossovers, three-parent diff3, virus evolution, rotating data or moving-average output.

## What is reused

The rule engine applies rules **in sequence**, so the state before rule *j* depends on every rule before *j*. For genomes having an **identical starting run of genes**, this version stores checkpoints of the intermediate transformed byte sequences and the corresponding rule-firing feature columns. An offspring can start again at the nearest checkpoint preceding its first changed gene instead of applying all earlier rules.

Equality is checked using the actual pattern and mode (`geneEqual`), not hashes, so hash collisions cannot corrupt reuse. Checkpoints are per data cohort, not globally keyed by sequence ID:

- `trainCache`: erased when the rotating training batch advances; reused during `--rotate-every` stable generations.
- `fixedCache`: retained for the fixed-reference cohort.
- `validCache`: retained for held-out monitoring only; validation never influences selection.

Each is bounded by a least-recently-used entry count and a measured **approximate** byte budget. Cached checkpoints are immutable, and later rewrites use new output buffers. If the budget is exceeded, old traces are evicted; a very large new trace is not stored.

This is an **exact prefix cache**, not an unsafe cache of independent suffixes. A change to rule 50 may change the input to rule 1499, even if rules 51..1499 are byte-identical. Non-prefix segments are therefore *not* blindly skipped. The actual amount of speedup depends on the mutation/crossover pattern; genome-wide uniform crossover may share little prefix.

## Other hot-path optimizations

- **Impossible-marker pruning**: when earlier rule `k` has not fired anywhere in the cohort, marker `256+k` cannot be present. A later rule with that literal marker is skipped entirely instead of scanning each row. This is exact given the raw-byte 0..255 input convention and unique marker creation in the canonical RHS.
- **LOO-only model scoring**: population selection, fixed-reference archive comparisons and virus/control trials skip the per-feature regression coefficients and predictions. Full coefficients remain available for virus diagnosis and validation plots.
- **Virus-context reuse**: features produced by the parent fit are reused by `buildViralContext`, eliminating another identical replay on the same host and cohort.

## New configuration

```
--cache-entries=8   # max genome traces per cohort (LRU)
--cache-stride=64   # retain state after every 64 rules
--cache-mb=128      # shared approximate MiB budget (70% train / 20% fixed / 10% valid)
```

`--cache-mb=0` disables the prefix memoization (but leaves the independent impossible-marker and LOO-only optimizations enabled). Higher `--cache-entries` and lower `--cache-stride` may increase reuse but also RAM consumption.

`cache_stats.dat` reports cumulative training / reference / held-out hits, exact hits, avoided rule-by-row applications, skipped scans due to impossible intermediate markers, and retained byte estimates. Use `--bench-cache` to run a synthetic shared-prefix benchmark that performs both cached and uncached computation, asserts identical feature columns, and prints observed seconds/speedup. The benchmark deliberately changes rules near the end; this is not a guarantee of the same speedup for arbitrary populations.

## Compile & tests (Nim 2.2.4 target)

```bash
bash check_build.sh
nim c -d:release --opt:speed --mm:orc -o:lens_pve cms.nim
./lens_pve --self-test
./lens_pve --bench-cache
./lens_pve --corpus=github-code-dataset.txt --rules=1500 --cache-mb=128 --cache-stride=64
```

The embedded Nim `--self-test` checks identical cached vs uncached feature vectors, edits early/late in the genome, impossible marker pruning against an unoptimized reference implementation, matching Ridge LOO fitness, and cache clearing. In this delivery environment the Nim compiler is unavailable, so Nim compilation and actual timing were **not** verified here. Python checks cover design invariants and mathematics but cannot substitute for Nim execution on your Mac.
