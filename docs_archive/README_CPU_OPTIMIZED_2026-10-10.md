# Lens / Replacer — CPU-native inference optimization (2026-10-10)

This is a source-complete drop-in update to
`Lens_Spearman_CPU_native_fused_inference_20261010.zip`.
It retains the October 7 Spearman + inference-GA objectives, wildcard and
hidden-memory semantics, model/GA behavior, and checkpoint format. No Metal or
PyTorch is required for the `cpu-native` backend.

## Run

Requirements: Python 3.10+, NumPy, and a C++17 compiler (`clang++` on macOS,
`g++` on Linux); install optional scientific/plotting packages if using those
parts of the program. The C++ source is compiled on first use and cached in
the OS temporary directory, *not* shipped as a platform-specific binary.

```
python main.py --backend cpu-native --cpu-threads 8 \
  --local-corpus github-code.txt --population 450 --rules 1500 \
  --hof-size 4096 --generations 100000
```

On Apple Silicon, start by setting `--cpu-threads` near the number of
performance cores and then benchmark other values. MPS and Python reference
backends remain selectable with `--backend mps` and `--backend cpu`.

## Changes compared with the preceding fused-native ZIP

1. **No allocations for failed rule probes.** Wildcard captures now use an
   inline 16-element stack buffer; replacement output capacity is allocated
   only on the first actual match. This affects both Spearman trajectory
   evaluation and inference scoring.
2. **Adaptive inverted rule index.** For >=1024-rule models with diverse
   literal anchors (>=96 distinct 0..512 first literals), a 513-token postings
   index visits only rules whose necessary first literal is present in the
   current rewritten state. Rule order is retained. When rules have a dense or
   repetitive anchor vocabulary, it chooses the old cheap ordered scan
   instead. **Hidden rewrite phases are never skipped**, including when whole
   ranges of rules are inactive.
3. **Allocation-free adjacent-pair Bloom filter.** Replaces the per-state
   `unordered_set` of pairs with a fixed 4096-bit probabilistic membership
   filter. False positives can cause extra exact checks; false negatives are
   impossible. A pair can only be skipped if absent.
4. **Persistent inference worker threads.** The C++ inference context starts
   its workers once and dispatches all candidate batches to that pool until
   the context is closed. Each worker evaluates a separate (model, candidate)
   tuple and has its own hidden-memory state and feature counts.
5. **Hidden-memory bookkeeping.** Cache the number of tokens in each hidden
   bank instead of scanning each bank at every insertion. Skip unnecessary
   hidden-rewrite snapshots when no viable mapping exists.
6. **Exact duplicate-object folding.** Repeated references to the *same*
   genome object share one native model evaluation and scores are expanded
   back to the requested rows. Distinct genomes are *not* folded without an
   exact proof of equivalence.
7. **Bounded inference search score memo.** Inner GA's previously unlimited
   score dictionary is now an LRU of up to `max(16384, 3 * current_batch)`
   candidate strings. This avoids potentially large transient memory spikes
   in very long inner searches. A missing old score is recomputed, never
   approximated. `search_cache_evictions` is included in GA statistics.

## Verification

```
python -m pytest -q
python main.py --self-test
python benchmark_cpu_index.py --scenario sparse --models 128 --rules 1500 \
  --candidates 64 --bytes 128
```

Validation performed on a Linux CPU (not an Apple Silicon speed test):

- 99 tests passed; four MPS-specific tests skipped.
- Extra randomized CPU-reference comparisons cover 650-rule genomes with
  wildcard and hidden-memory instructions, indexed/dense literal distributions,
  custom embeddings, and multiple hidden-rewrite phase counts.
- A two-generation training smoke test finished with the inference GA enabled;
  `nativeCalls=4` was recorded per generation.
- A persistent inference context completed 200 consecutive candidate batches.
- Clang 17 and g++ successfully compiled the C++ source.

### Illustrative scoring-kernel timings

128 models × 1500 rules × 64 candidates, 128 bytes each, 8 CPU threads.
Times are medians of 7 warmed scoring batches; model packing/initialization is
*excluded*. Workloads are synthetic, and per-run CPU contention can change
these values. The preceding fused-native ZIP was run on the same host.

| Test data | Previous | Optimized | Approx. speedup |
|---|---:|---:|---:|
| Sparse literal anchors | 0.885 s | 0.738 s | 1.20× |
| Dense repeated anchors | 0.176 s | 0.185 s | 0.95× (slower) |
| Mixed anchors | 0.149 s | 0.136 s | 1.10× |

These benchmark runs were conducted before the final inline-capture change,
so actual speeds may differ. The predicted end-to-end gain is **not** 1.20×:
corpus preparation, crossover/mutation, readout training, Pareto sorting, and
CSV generation are outside this microbenchmark. Native context setup is also
excluded. Results must be measured on the target Mac.

## Correctness and trade-offs

All candidate matches and emitted strings still use the exact C++ interpreter
and all filtering rules are **necessary** conditions for a possible match.
No mutation probabilities, inference loss, or Spearman correlation were
changed. Checkpoints and `--backend cpu` reference remain available.

The persistent inference context is intended to be called sequentially from
one Python thread; do not call `.score()` concurrently on the *same* context
from multiple Python threads. Different contexts and independent trajectory
batches are safe. This repository does not currently have end-to-end Apple
Silicon benchmarks or a claim of universal acceleration.
