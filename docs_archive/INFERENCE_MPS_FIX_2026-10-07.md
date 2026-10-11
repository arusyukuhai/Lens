# Lens / Replacer inference-MPS performance repair — 2026-10-07

## Diagnosis

The apparent `mps≈6s` versus `seconds>20s` gap was mostly not an unknown Python stall.
The generation timer includes the inner inference GA, while the historical `mps_seconds`
and printed `mps=` value described only the outer training evaluation. During inference,
MPS diagnostic fields were intentionally saved/restored so the outer profiler was not
polluted. This made inference GPU work invisible in the `mps=` number.

The supplied `fitness_history.csv` confirms this strongly:

- Among 21 rows with `mps_seconds` in 5.5–7.0 s and `seconds>20 s`, median generation
  time was 22.50 s, median outer MPS time 6.61 s, median full outer evaluation 7.60 s,
  and median inference time 14.28 s. Inference ranged from 10.65 s to 42.01 s there.
- For rows with `mps_seconds` in 5–7 s, correlation between total `seconds` and
  `inference_seconds` is about 0.984.
- In the last 200 supplied history rows, the median residual
  `seconds - outer_eval - inference - search_model` is only about 0.039 s
  (90th percentile about 0.066 s).

So the dominant issue is the inference path itself plus misleading accounting.

## Why inference hyperparameters appeared to do nothing

`resolve_inference_budget()` uses `--inference-budget-ratio` as a hard auto-work cap.
With the common 200 outer samples, 450 genomes and 32-model inference ensemble, requests
such as 3x16x128 and 3x16x256 both collapse to the same effective plan 3x16x29.
Changing a requested value above that ceiling therefore cannot change runtime or search
shape. This was intentional bounded-work behavior, but the CLI did not make the clamping
obvious.

The repair adds:

- `infReq=<cases>x<population>x<generations>:<mode>` beside the actual `infPlan=...`.
- A first-generation warning when auto mode clamps the requested plan.
- `--inference-budget-mode direct`, which honors requested inference population and
  generations (subject only to the MPS sample-buffer width). Auto remains the safe default.

Use direct mode only when you deliberately want the shape knobs to be literal; it can
exceed the previous 0.85-equivalent-work safety ceiling.

## Performance fixes

### 1. Fast inference input upload

Previously every inner inference generation called `set_inputs()` and recomputed:

- byte frequencies,
- 256x256 pair counts,
- triple-frequency dictionaries,
- a CPU buffer for the entire outer sample allocation,
- and then uploaded the full allocation.

Those frequency tables are only used to choose structural rule anchors while packing a
model. The inference search reuses an immutable prepacked model/index context, so those
statistics cannot affect search-round correctness.

Inference candidate uploads now:

- leave the outer training anchor statistics untouched,
- allocate and upload only the active candidate prefix,
- preserve exact duplicate-input aliasing,
- and restore the outer raw input prefix without rescanning anchor statistics.

This also removes the old need to recompute full training frequency statistics before the
final inference pack: the training statistics were never overwritten in the first place.

### 2. Exact cross-round candidate score memoization

The inner GA preserves elites unchanged, but the old implementation evaluated the same
candidate strings again on every generation even though model structures/readout weights
are fixed throughout the inner search.

A generation-local `bytes(candidate) -> score vector` cache now evaluates only unique
previously unseen candidate strings. This is exact memoization, not a surrogate.

A CPU equivalence benchmark against the pre-fix implementation produced bit-identical
inference accuracies and diagnostics while reducing candidate evaluations from 396 to 313
and runtime from about 2.18 s to 1.66 s in that synthetic test (~24% faster).

### 3. Reuse the final ensemble evaluation

The final candidate pool is exactly the pool scored by the last search generation.
Previously the final 450-model objective pass redundantly reevaluated the 32 ensemble
models on those candidates. Their already-computed rows are now reused, and only the
remaining models are evaluated.

### 4. Remove redundant MPS clears

`stat_stage` and `profile_stage` are completely overwritten by each active Metal job.
Only `feature_stage`, whose counters are incremented, needs zeroing. The redundant clears
were removed.

## New timing fields

The CSV now records:

- `mps_outer_seconds` — historical outer evaluator wall time.
- `mps_inference_seconds` — inference evaluator calls only.
- `mps_total_seconds` — outer + inference evaluator wall time.
- `inference_mps_input_seconds` — candidate/raw input staging.
- `inference_mps_dispatch_seconds` — accumulated Metal dispatch/result synchronization.
- `inference_mps_reduce_seconds` — host readout prediction reduction.
- `inference_mps_context_seconds` — inference model/index packing.
- `inference_mps_restore_seconds` — restoring the outer raw input prefix.
- `inference_mps_calls` — actual MPS inference evaluation calls.
- `inference_search_cache_hits` / `inference_search_cache_entries`.
- requested inference shape and budget mode.

The stdout line now prints `mpsOuter=`, `infMPS=`, and `mpsTotal=` instead of implying
that outer MPS time is all MPS work.

## Validation

- `pytest`: **47 passed, 4 skipped**. The four skips are obsolete differential-evolution
  tests intentionally retained as skipped after the project returned to normal GA.
- Built-in `--self-test`: PASS.
- CPU inference equivalence against the pre-fix audited implementation: exact same
  per-genome inference accuracy, ensemble accuracy and oracle accuracy.
- CPU smoke evolution in `--inference-budget-mode direct`: PASS and CSV schema extension
  verified.
- Python compilation: PASS.

A real Apple MPS device is not available in this execution environment, so the revised
Metal orchestration cannot be benchmarked here on hardware. The changes to the actual
Metal rewrite semantics are zero: the kernel code's rewrite behavior is unchanged. The
MPS-side changes are host staging, redundant-clear removal, exact result reuse, and
instrumentation.

## Recommended first MPS run

Keep auto budget initially and inspect a few generations. The useful line now looks like:

`... seconds=... mpsOuter=... infMPS=... mpsTotal=... infMPS(input/dispatch/reduce/context)=... infCache=...`

If `infMPS dispatch` dominates, reducing effective inner generations is the direct knob.
If you want `--inference-generations` / `--inference-population` to be literal rather than
budget-clamped, add `--inference-budget-mode direct` and start conservatively.
