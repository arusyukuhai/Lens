# Lens / Replacer — fused CPU-native inference fix (2026-10-10)

This revision fixes incomplete native inference integration in
`Lens_Spearman_CPU_native_20261010_full.zip`.

## What was actually wrong

The previous `cpu-native` backend **did** execute Replacer trajectories in C++,
but `evaluate_inference_candidates` still iterated across each genome in Python,
called the C++ evaluator separately for each genome, created and joined a thread
pool in every call, materialized a full feature matrix, and then applied each
readout with NumPy. When the inference GA has many models, that overhead can be
significant. The earlier statement that inference was already optimally
native was too broad.

## What is now native

* A *single* C++ call evaluates all `(model, candidate)` pairs with a shared
  native work queue. The GA's search ensemble and final remaining-model pass
  each get a cached, immutable C++ model context that persists over candidate
  rounds. Replacer rules, embeddings, and readout weights are parsed once per
  context, not once per candidate round.
* Readout-weighted inference scores are accumulated **inside the C++ evaluator**
  when a rule fires. A full model×candidate×rule feature tensor is not created.
  The exact same C++ `evaluate_one()` interpreter powers both outer Spearman
  evaluation and inference.
* All previously supported wildcard transforms, hidden-memory states, cycle
  detection, overflow score, and rewrite-phase scheduling are retained.
* `--backend cpu-native` with the existing `--cpu-threads N` works unchanged.
  It requires a C++17 compiler on the user's machine. It is not an MPS kernel.
* Each generation now prints `infNative=...s nativeCalls=N nativePack=...s`
  (only for the `cpu-native` backend) so native inference usage is visible.
  These values also appear in training-history entries as
  `inference_native_seconds`, `inference_native_calls` and
  `inference_native_context_seconds`.

**Not moved to C++:** proposal mutation/crossover, tie-aware rank voting,
Pareto-selection, and the outer evolutionary algorithm remain in Python.
These are distinct from the heavy Replacer candidate-scoring kernel. The
inference objective and mutation probabilities are unchanged.

## Run

```sh
python main.py --backend cpu-native --cpu-threads 8 \
  --local-corpus github-code.txt --population 450 --rules 1500 \
  --hof-size 4096 --generations 100000
```

No new dependencies beyond those for the earlier CPU-native release are needed.
The native shared library is automatically recompiled to a temporary cache when
the included `replacer_native_cpu.cpp` changes.

## Checks performed

* 81 tests passed; four MPS-only tests skipped on the non-Mac host.
* Randomized parity tests compare the fused C++ inference against Python CPU
  over 8 seeds and three hidden-memory phase counts, including mismatched or
  inactive readout weights.
* A smoke training run completed two full generations, three inference GA
  rounds each, with `nativeCalls=4` reported.
* An illustrative host benchmark with 64 models × 750 rules and 28 strings
  (96 bytes each) measured ~0.065s per batch on the **old** per-genome
  native path, versus ~0.041s on the **new** fused path, excluding the ~0.03s
  one-time context creation. These are local, non-Apple-Silicon values, **not**
  a claim about production end-to-end acceleration.

Re-run on the target Apple Silicon machine before concluding that CPU is faster
than MPS overall. `--backend cpu` remains the Python correctness reference.
