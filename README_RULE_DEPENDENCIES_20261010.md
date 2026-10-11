# Lens / Replacer — execution-based rule dependency tracing (2026-10-10)

This update extends the **existing trace-guided optimizer** to select edit
locations anywhere in the rule list. GA remains separately available with
`--optimizer ga`, and the HOF capacity stays **4096** by default. Old trace/GA
checkpoints load without graph migration. The graph is temporary diagnostic
state, intentionally absent from checkpoints and HOF genomes.

## How the graph is assembled

While simulating a small number of teacher-forced text prefixes, the tracer
maintains a **last-writer rule index per state byte**. When rule `j` actually
changes a matched span containing a byte last rewritten by rule `i`, it records
an observed **directed dataflow edge** `i -> j`. Unchanged input spans keep
their provenance; output bytes of changed matches are assigned to `j`.
Teacher-forced target bytes and appended prediction placeholders have no rule
producer. Provenance persists between prediction steps.

This is an execution-derived *approximation*, not a complete causal graph:
last-writer provenance does not include indirect effects of deletions,
matching failure, branch/control dependencies, or nonlocal insertion effects.
A directed edge certifies observed token consumption by a rewriting rule, not
that deleting `i` would necessarily change the model's final prediction.
Final changes must still pass real full-text native rollout evaluation.

For each selected prediction step `t`, define binary `X_it = 1` if rule `i`
**changed the state** at any point during that step; a match with identical
replacement does not count. Two co-firing rules get the phi coefficient:

```
phi(i,j) = (N*n11 - ni*nj) / sqrt(ni*(N-ni)*nj*(N-nj))
```

where `N` is the number of sampled prediction steps; `ni` and `nj` are the
number of steps in which each rule changed state; `n11` is their co-firing
count. Undefined phi (one constant variable) reports `0.0`. A separate
`phi(X_i, error)` correlates effective firing with an incorrect next-byte
prediction. Phi is **an association**, not evidence of direction or causation.
Cross-step true dataflow can have negative phi.

## Selecting the rule to modify

1. Score the final prediction byte's last-writer on incorrect predictions.
2. Propagate decreasing credit backward through observed producer-to-rewriter
   edges up to three hops. Edge weights use observed frequency, with positive
   cofire phi only as a *small modifier*. Negative phi never invalidates a
   real cross-step edge.
3. Add a small bonus for effective firing positively associated with errors,
   with a minimum support threshold.
4. Rotate among the top-ranked rules. Every fifth generation, try a tail
   rule even if dependency evidence suggests other locations. If evidence is
   missing, fall back to the old tail rotation.
5. Synthesize proposals from the chosen rule's observed pre-rule error states,
   rank by exact one-step suffix counterfactual, and only accept with **native
   full teacher-forced rollout on all active training examples**.

This removes the former hard restriction to the last 16 rules for graph-led
iterations, without doing global GA crossover or mutation.

## Running from an existing trace checkpoint

```bash
python3 main.py \
  --optimizer trace --trace-dependencies --backend cpu \
  --local-corpus github-code.txt \
  --load lens_trace_checkpoint.pkl \
  --checkpoint lens_trace_checkpoint.pkl \
  --history-csv lens_trace_history.csv \
  --save latest_lens_trace.json \
  --hof-size 4096 --generations 400 --no-plot
```

Change `--load` to `lens_ar_checkpoint.pkl` when migrating directly from
an older GA checkpoint. Use the same training corpus and case rotation settings
as the original run. `--generations` denotes an **absolute** generation count.

## New flags

| Flag | Default | Purpose |
|---|---:|---|
| `--trace-dependencies` / `--no-trace-dependencies` | enabled | Dataflow / phi rule selection, or original tail-only behavior |
| `--trace-dep-every` | 8 | Recompute graph at least this often; edits and case rotation also invalidate |
| `--trace-dep-prefix` | 96 | Maximum prefix bytes traced per text |
| `--trace-dep-steps` | 24 | Evenly spaced sampled prediction steps per text |
| `--trace-dep-texts` | 3 | Text examples included in graph |
| `--trace-dep-topk` | 12 | Number of top rule locations to rotate through |
| `--trace-dep-min-support` | 2 | Minimum firing support for the phi(error) bonus |
| `--trace-dep-explore-every` | 5 | Tail edit every N steps (0 disables) |
| `--trace-dep-graph-prefix` | `trace_dependencies` | CSV prefix (empty disables export) |
| `--trace-dep-graph-every` | 20 | Rewrite CSVs every N generations (0 disables) |
| `--trace-state-cap` | 4096 | Shared budget for reference dependency trace and suffix replay; overflow aborts trace, not truncate model |

The graph emits **two bounded CSVs** (overwritten at each export):

* `<prefix>_edges.csv`: `source_rule`, `target_rule`, `dataflow_steps`, `phi_effective_fire`, `cofire_steps`, `source_fires`, `target_fires`, `observations`. Max 2048 edges sorted by support.
* `<prefix>_rules.csv`: `rule`, `effective_fire_steps`, `phi_fire_error`, `direct_prediction_errors`, `dependency_priority`, `observations`, `errors`.

Per-generation console and history also include `trace_dep_row_method`,
`trace_dep_score`, `trace_dep_samples`, `trace_dep_errors`, `trace_dep_edges`,
`trace_dep_active`, `trace_dep_overflows`, `trace_dep_phi_error`. There is no
persistent O(generations × 1500²) correlation matrix.

## Caveats and validation

Sampling, dense correlated activations, and the moving training case pool can
bias priorities. Phi based on only 24 steps per text is especially noisy for
rare rules; increase `--trace-dep-steps` or `--trace-dep-texts` to improve
coverage at additional CPU cost. Detection of dependencies does not prove that
changing a rule will improve ACC. Early-layer edits can have global effects:
**only the final full teacher-forced rollout decides acceptance**. An
independent held-out corpus is needed to measure generalization.

Verification:

```bash
python3 -m pytest -q test_rule_dependencies.py test_trace_guided.py \
  test_memory_regressions.py test_rolling_cases.py test_autoregressive.py \
  test_binary_operators.py test_corpus_split.py test_genetic_operators.py \
  test_progress.py test_dynamic_lengths.py
```

The shipped 200-generation / 1500-rule checkpoint was loaded successfully,
and a sample graph yielded directed dependencies without memory growth.
This is not a claim of a completed long-run scaling benchmark.
