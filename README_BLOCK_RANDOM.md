# Lens PVE: per-row interval memoization + random diffusion levels

This builds directly on the shared-prefix-cache and moving-average versions. The 1500-rule genome, 32 ordinary edits, host crossover suite, diff3, viral programs, rotating training chunks, fixed reference/held-out evaluation, and moving-average graphs remain intact.

## A more general exact cache

The original cache reused only the *common initial prefix* of a pair of genomes. The new **interval memo** caches each consecutive `--cache-stride` rule block **separately for each input row**:

```
key = hash(block_start, block_end, actual_rule_patterns_and_modes,
           actual_state_entering_this_block)
value = (actual_state_leaving_this_block, per-rule_raw_firing_counts)
```

Hashes only index candidate results: a potential hit verifies the complete rule block and the complete incoming state for equality, so hash collisions cannot result in incorrect feature values. The raw firing counts are re-normalized by the current row's source length when reused. The output buffer and stored input buffer are treated as immutable.

Importantly, **even when two genomes differ at rule 0**, their subsequent input states can coincide (e.g. the edit did not fire on some rows), and unchanged downstream blocks can be reused. Rows with different intermediate inputs are recomputed. No implicit assumption that an unchanged suffix of rules has an unchanged input.

The original prefix cache remains in place, so long identical starts can be skipped with minimal work. Both caches are restricted by separate budgets; prefix traces remain LRU and interval entries use bounded FIFO eviction. Train/held-out/reference have separate stores; the training cache is cleared on rotating-batch changes.

### Options (defaults)

- `--cache-entries=8` maximum prefix traces per cohort
- `--cache-stride=64` rules per checkpoint / memoized interval
- `--cache-mb=128` MiB approximate total prefix memory, distributed 70/20/10 among train/reference/held-out
- `--block-cache-mb=128` MiB approximate additional interval memory, also distributed 70/20/10
- `--block-cache-entries=8192` maximum interval records *per cohort*

`--cache-mb=0` disables only prefix caching; `--block-cache-mb=0` disables only interval memoization. Set both to zero to test uncached evaluation. `cache_stats.dat` now appends block-cache hit/miss counts, avoided rule-row applications, and cached block bytes. The existing columns retain their original positions. Interval memoization adds hashing and bookkeeping overhead when reuse is rare: its speed benefit is workload-dependent and must be measured.

### Benchmark

```bash
bash check_build.sh
./lens_pve_check --bench-block-cache
./lens_pve_check --bench-cache
```

These run equal-feature tests in a synthetic scenario; the block benchmark changes the first rule but chooses edits that do not fire, encouraging safe suffix reuse after a divergent prefix. **No real-world speedup can be promised without running the application and measuring the actual corpus, edits and rotations.**

## Random diffusion-level sampling (now the default)

The old sampling used approximately equally-spaced numbers of corrupted positions. Now `--diffusion-sampling=random` selects `--noise-steps-1` *distinct* integers uniformly without replacement from `1..chunk_length-1`, sorts them, then adds the intact (0) and fully corrupted (`chunk_length`) endpoints. Therefore the default `--noise-steps=16` produces **17 rows per sufficiently long chunk**, including 15 randomly sampled interior levels.

The underlying corruption *trajectory* remains a cumulative, per-chunk random permutation of positions. Noise fractions are actual `changed / original_length`, not a nominal evenly-spaced stage index. A global corpus chunk id and `--seed` determine its rows deterministically; changing the order of chunks in the training batch cannot change the sampling. This also ensures memoization and cross-genome comparison remain consistent. `--diffusion-sampling=uniform` reproduces the old equal-spacing scheme.

Use:

```bash
bash run.sh \
  --corpus=github-code-dataset.txt \
  --rules=1500 --population=12 --train=2 --valid=2 \
  --rotate-every=1 --noise-steps=16 \
  --diffusion-sampling=random \
  --cache-stride=64 --cache-mb=128 \
  --block-cache-mb=128 --block-cache-entries=8192
```

## Validation and limitations

The Python reference tests check 500 random level sets and 350 genomes with divergent prefixes, verifying all intermediate firing columns against an independent serial rewrite model. Existing mathematical/regression tests still pass. The Nim `--self-test` now exercises real `replace.nim` execution with both caches enabled/disabled, divergent edits, a simulated hash collision, memory bounds and deterministic random stages. **The Nim compiler is not available in the packaging environment; Nim 2.2.4 compilation and actual training/benchmark execution have NOT been verified.** Run `bash check_build.sh` on your Mac.
