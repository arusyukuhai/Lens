> **2026-10-10 correction:** The original CPU-native release used a C++
> per-genome loop during inference GA. See
> `README_FUSED_NATIVE_INFERENCE_2026-10-10.md` for the newer fused native
> inference path, tests, benchmark and instrumentation.

# Lens / Replacer — Spearman CPU-native revision (2026-10-10)

This package modifies the **2026-10-07 Spearman + inference-GA revision**.  It
preserves the original implementation, Metal backend and Python reference CPU
backend. `--backend cpu-native` is the new default; use `--backend mps` to opt
back into Metal or `--backend cpu` to run the Python reference engine.

## Run

Requirements: Python 3.10+, NumPy, and a C++17 compiler.  Install optional
packages used by plotting, datasets, and tests as needed. PyTorch is *not*
required for CPU-native execution.

On macOS, install Xcode Command Line Tools (`xcode-select --install`) to get
`clang++`. On Linux, install `g++`. The first CPU-native evaluation compiles
`replacer_native_cpu.cpp` to a cached library in your operating system's temp
directory; subsequent runs reuse it. The ZIP contains *source*, not a binary
compiled for any particular architecture.

Example:

```sh
python main.py --backend cpu-native --cpu-threads 8 \
  --local-corpus github-code.txt --population 450 --rules 1500 \
  --cases 2 --samples 100 --hof-size 4096 --generations 100000
```

Change `--cpu-threads` to a value near your CPU's performance-core count.
The same number controls **both** outer Spearman evaluation and inner inference
GA. `--no-inference` bypasses the second objective entirely if required.
`--backend cpu` remains a useful correctness oracle.

## What was changed

* C++17 exact rewrite engine with `std::thread` trajectory parallelism; no
  OpenMP dependency is needed on macOS.
* Fast, **sound** 512-token bitset and adjacent literal-pair presence index.
  A rule is skipped only if its necessary literal(s) cannot occur in the
  current state. The index is lazily rebuilt after a visible rewrite.
* Implements all supported wildcard captures, rule output transforms, hidden
  opcode reads/pops/deletes, bounded hidden side-lists, persistent rewrite
  phases, cycle detection and per-rule firing counts.
* Inputs are embedded in native code. The batch interface avoids Python
  per-rule and per-input loops and skips final-state materialization in
  training and inference.
* The inference GA reuses bounded cached rule packs and packs each candidate
  batch once instead of once for each voting genome.
* Outer CPU-native evaluator now reuses valid unrotated-case fitness and
  deduplicates structurally identical genomes, including HoF entries. A case
  rotation invalidates the cached score exactly as in the MPS path.
* Readout fitting, Spearman calculation, Pareto selection, mutation, HoF and
  serialization are unchanged.

## Validation

```sh
python -m pytest -q
python main.py --self-test
```

`test_cpu_native.py` compares native and reference trajectories (features,
baseline, stats **and** final tokens), including hidden memory and all 24 hidden
opcodes. A two-generation CPU-native smoke training run with inference GA was
also completed. On the build/test host, all 51 applicable tests passed; four
MPS-specific tests were skipped. This is **not** a guarantee of bit-identical
behavior with MPS; the primary equivalence oracle is the Python CPU engine.

Illustrative measurements on the build/test host (not Apple Silicon):
256 rules / 12 inputs / 80 tokens: approximately 14x native versus Python;
1500 rules / 4 inputs / 256 tokens: approximately 34x. These compare only the
trajectory evaluator, not total evolution time, and do not predict the
CPU-versus-Metal crossover on your own machine.

## Notes and safeguards

* The C++ rule-pack LRU holds at most 96 genomes; no unbounded accumulation as
  HoF grows. The normal evolutionary code uses copy-on-write rules: replacing a
  rule object automatically invalidates the pack identity check. Do not mutate
  a cached Rule's `.pattern` or `.replacement` list **in place** from external
  extensions without bypassing or clearing the cache.
* This is an exact CPU reference-semantics backend. It does **not** change
  learning objectives or claim a faster generation than tuned MPS on all data.
* Existing checkpoints can be used with `--load`, provided the original
  checkpoint's semantics and `--hidden-rewrite-phases` are preserved.
* Always benchmark with actual 450-genome, 1500-rule production cases on your
  own CPU to choose between `cpu-native` and `mps`.
