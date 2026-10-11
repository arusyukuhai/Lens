# Lens / Replacer — trace-guided rule synthesis (2026-10-10)

**UPDATE (2026-10-10):** Optional execution-provenance + phi dependency tracing is now enabled by default in `--optimizer trace` and can select earlier rules. See `README_RULE_DEPENDENCIES_20261010.md` for exact semantics, flags, and caveats.

This release adds a **new non-GA optimizer** that uses observed execution states and the teacher-forced target bytes to synthesize replacement rules. The previous GA remains available and is unchanged by default. HOF remains **4096** by default and is not discarded on resume.

## Starting from the existing checkpoint (recommended)

Execute from the extracted project directory, with the **same corpus** and `--cases`, `--case-rotate-every` used to create the original checkpoint:

```bash
python3 main.py \
  --optimizer trace --backend cpu \
  --local-corpus github-code.txt \
  --load lens_ar_checkpoint.pkl \
  --checkpoint lens_trace_checkpoint.pkl \
  --history-csv lens_trace_history.csv \
  --save latest_lens_trace.json \
  --hof-size 4096 --generations 400 --no-plot
```

`--generations` is the absolute total generation count. If the checkpoint is at generation 200, `--generations 400` runs from 200 to 399. For a completely new model, omit `--load` and provide `--rules 1500`; training then begins from a randomized starting *structure*, but all subsequent trace-mode improvements are supervised synthesized edits, **not** GA mutation or breeding.

To resume an optimizer-trace run, change only the load path:

```bash
python3 main.py --optimizer trace --backend cpu --local-corpus github-code.txt \
  --load lens_trace_checkpoint.pkl --checkpoint lens_trace_checkpoint.pkl \
  --history-csv lens_trace_history.csv --save latest_lens_trace.json \
  --hof-size 4096 --generations 600 --no-plot
```

Keep matching `--cases` and `--case-rotate-every` if you used nondefault values before. The rolling case index and RNG state are restored exactly, so using a different training corpus raises a deliberate error rather than silently losing reproducibility. Loading *pickle* from untrusted sources is unsafe; use your own checkpoint.

To restore the old GA mode, use `--optimizer ga` or omit `--optimizer`.

## What actually changed

1. **Bounded, exact reference traces.** On a handful of teacher-forced text prefixes, log the actual *pre-rule* recurrent state, target byte, and final baseline prediction for a selected late-stage rule. These are snapshots of true sequential execution, not invented input patterns.
2. **Rule synthesis from errors.** For incorrect predictions, construct: short literal suffix edits; anchor/capture/suffix wildcard edits; two-capture structural edits; and captured-byte transforms using the real permutation embedding (+1, -1, ×2, ÷2). These are generated from actual traces, not randomly modified genotypes.
3. **Counterfactual suffix replay.** Replace **one** late-stage rule and run only the rules following it on the *same saved pre-rule state*. Rank changes by `fixed_predictions − broken_predictions` over successes *and* errors. The original rule's effect is accounted for; the local state is never silently cropped.
4. **Complete native confirmation.** Only the best few locally promising rules are run through the usual full teacher-forced, **stateful** CPU evaluator on **all active samples**. Adopt the winner only if actual full-text accuracy strictly improves. This is important: a snapshot-local improvement is not sufficient.
5. **One incumbent, no GA.** The optimizer keeps one current model plus its HOF archive. It **never invokes crossover, mutation, tournament selection, or a 450-member fitness sweep** during the trace learning loop.
6. **Compatible checkpoints.** GA v4 checkpoints remain loadable. Newly saved trace checkpoints record a one-model live population, all archived HOF genomes (max 4096), history, RNG, and rolling case state. `--save` stores the most recent accepted *current* model, not a stale best from a different rotation of cases.

## Key settings

| Flag | Default | Description |
|---|---:|---|
| `--optimizer trace` | `ga` (backward compatible) | Choose supervised trace synthesis. |
| `--trace-tail` | 16 | Fallback/exploration: cycle across final 16 rules. Dependency mode can choose any rule. |
| `--trace-prefix` | 96 | Maximum number of prefix bytes per selected case used to collect exact execution traces. |
| `--trace-steps` | 16 | Evenly sampled prediction steps per selected case. |
| `--trace-texts` | 3 | Number of current training cases traced. |
| `--trace-context` | 6 | Maximum literal suffix context length used for new rule candidates. |
| `--trace-candidates` | 96 | Maximum candidate rules evaluated locally per generation. |
| `--trace-confirm` | 4 | Locally promising candidate models scored on **all** training bytes. |
| `--trace-state-cap` | 4096 | Limit for the *tracing/counterfactual budget*, not a model or rule truncation. |
| `--max-state-bytes` | 65536 | Existing native rollout budget; over-budget sample gets score 0. |
| `--hof-size` | **4096** | HOF capacity, unchanged. |

The trainer prints `TRACE`, `error=`, `row=`, `proposals=`, `local=`, `full=`, `accepted=`, `gain=`, `overflow=`, `seconds=`, and `hof=`. Each generation is appended to `--history-csv` with additional `trace_*` columns.

To cover more contexts in long source files, experiment with `--trace-prefix 256 --trace-steps 32`. This increases reference replay work but **not** the number of full native evaluations (at most `--trace-confirm` additional model candidates when local evidence suggests improvement).

## Tests / verification

```bash
python3 main.py --self-test
python3 -m pytest -q \
  test_trace_guided.py test_memory_regressions.py test_rolling_cases.py \
  test_autoregressive.py test_binary_operators.py test_corpus_split.py \
  test_genetic_operators.py test_progress.py test_dynamic_lengths.py
```

These nine relevant test modules pass (70 tests plus 6 embedded subtests). The original ZIP also contains older test suites written for incompatible, former **non-autoregressive/Nim** variants of `main.py`, so running indiscriminate `pytest` on every historical test does not pass even in the unmodified source. No attempt was made to retrofit those old tests to the new model.

Observed proof of concept: loaded the packaged gen=200 checkpoint with 1500 rules, traced 32 short text positions and verified locally-generated candidates; full native evaluation on two *short, synthetic* examples improved observed accuracy from 0.085526 to 0.138158. **This is not proof of long-corpus improvement or generalization.** The dependency update can now select arbitrary early rule positions, but still does not synthesize arbitrarily complex operators or perform mathematically exact reverse propagation. Teacher-forced full rollout confirms fitness on the active cases only, not on unseen data. Use independently held-out data for any generalization claim.
