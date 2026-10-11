# Lens / Replacer — CPU batched outer evaluation + rank readout speed fix

This is an in-place compatible update of `Lens_Spearman_CPU_optimized_inference_20261010.zip`.
**The model, fitness definitions, crossover, mutation, hidden-memory semantics, archive and checkpoint format have NOT been intentionally changed.** Previous Python-CPU and MPS backends remain available.

## Root cause of disappointing chunk-count scaling

There are different knobs for **corpus size** and **per-generation compute**:

- `--corpus-chunks` limits how many chunks are loaded *into the corpus*. If you have significantly more than `--cases`, reducing this limit does **not** shrink the outer evaluation: it only reduces corpus variety and a portion of startup work. A larger corpus is randomly sampled.
- `--cases * --samples` controls the **outer evaluation trajectory count**. For example, `--cases 2 --samples 100` uses 200 trajectories; changing `--corpus-chunks 65536` to `--corpus-chunks 21845` leaves those 200 trajectories unchanged.
- `--inference-cases`, `--inference-population`, `--inference-generations`, `--inference-span`, `--inference-ensemble`, and the selected `--inference-budget-mode` control the inner inference GA. Reducing `--cases` or `--samples` may not reduce all inference work proportionally, especially under `--inference-budget-mode direct`.
- `--population`, `--hof-eval` (per generation; **NOT** `--hof-size`), `--rules` and input length change how much work the model does; breeding and checkpoints also have fixed costs.

Amdahl's law matters: when a substantial proportion of a generation is fixed, reducing just the trajectory count by 3x does not guarantee a 2x speedup.

## Optimization details

1. **Fused outer population evaluator** (C++17): a persistent worker pool evaluates all the combinations of 16–32+ genomes and outer trajectories in one call. Previously each genome launched/joined the C++ worker threads, re-built rule plans, and incurred one Python native call. The inner inference GA's fused evaluator is preserved and now re-used for outer trajectories as well.
2. **64 MiB temporary feature memory ceiling** per outer batch. `--cpu-model-batch` defaults to 32 and is automatically reduced for large trajectory/rule counts. This prevents a 450x200x1500 float32 feature array from being materialized all at once.
3. **Share fixed target-rank graph** across all genomes evaluated in one generation, rather than re-ranking the same cleanliness targets for every genome.
4. **Triangular Cholesky solver** using SciPy `cho_solve` when available. The previous code used generic `numpy.linalg.solve` twice *after* Cholesky. When SciPy is absent, the original NumPy implementation remains available.
5. **Avoid BLAS oversubscription**: `--blas-threads 1` is the new CPU-native default (implemented with `threadpoolctl` when installed). Override by passing `--blas-threads 0` to keep NumPy's original configuration, or pass 2–4 for comparative benchmarks.
6. **Phase instrumentation** prints `outerCPU`, `pack`, `readout`, `infGA`, `infNative`, `nativeCalls`, `eval/reuse/dedupe` on each generation; the CSV includes `cpu_native_kernel_seconds`, `cpu_native_setup_seconds`, and CPU readout seconds.

## Start

```bash
python -m pip install numpy scipy threadpoolctl
python main.py --backend cpu-native --cpu-threads 8 --cpu-model-batch 32 --blas-threads 1 \
  --local-corpus github-code.txt --cases 2 --samples 100 \
  --population 450 --rules 1500 --hof-size 4096 --hof-eval 32
```

For an **existing checkpoint**, use your normal `--load ...` and `--checkpoint ...` arguments. As always, keep `--hof-size 4096` if desired; tuning `--hof-eval` only controls how many archive entries are *re-evaluated per generation*, not how many are saved. See `python main.py --help` for details.

On macOS/Apple Silicon, start testing `--cpu-threads` with your performance-core count. Benchmark 4/6/8/10 etc., because more threads are not necessarily faster. `--cpu-model-batch 16` is also worth trying when memory pressure is visible.

If generation logs show `infGA` dominating, evaluate changes to `--inference-generations`, `--inference-population` or `--inference-span`, rather than `--corpus-chunks`. Always inspect **effective** `infPlan=...` rather than just requested `infReq=...`; in `auto` mode several knob values can map to the same capped plan.

## Verification performed on Linux x86-64

- Existing test suite: 99 passed, 4 MPS-specific skipped.
- Additional fused outer differential tests: 13 passed. Bitwise equality checked between single-genome C++ and fused batch C++ for multiple genome lengths, wildcard and hidden instructions, embeddings and 1/2/3/5 hidden phases. Also compared shared-readout pairs against uncached reference.
- Two-generation training run with Spearman + inference GA completed; matching fitness, inference accuracy, and generation statistics against prior release with the same RNG seed and configuration.
- 32 genomes x 1500 rules x 100 trajectories x 600 bytes: outer native dispatch+pack 0.141s -> 0.053s (2.66x for that phase).
- 96 genomes x 1500 rules x 200 trajectories x 650 bytes: outer native dispatch+pack 0.848s -> 0.271s (3.14x for that phase). The synthetic workload is not a prediction of full-training or Apple Silicon speed.
- Typical 200x1500-feature pairwise readout tests observed much poorer performance with eight BLAS threads than one, and the SciPy triangular solver removed redundant factorization work.

Run tests/benchmarks:

```bash
python -m pytest -q
python main.py --self-test
python benchmark_outer_native.py --models 96 --rules 1500 --samples 200 --bytes 650 --cpu-threads 8
```

**Caveats:** No full-size Apple Silicon benchmark has been run. Wall-clock speed depends on the real genome distribution, rule firing, hidden memory, thread scheduling and inference search. This update does not reduce the number of underlying logical evaluations or alter the intended training objective. Compiled native libraries are created on first use in the OS temp directory; no binaries are bundled.
