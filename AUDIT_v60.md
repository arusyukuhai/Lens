# v60 checked revision — audit notes

This package was re-audited after the initial string-side inference/Pareto implementation. The checked revision keeps the same user-facing objective but fixes several correctness and performance issues.

## Important fixes

1. **Inner GA no longer evaluates all outer genomes on every search generation.**
   - Search generations are driven only by the configured Spearman voting ensemble.
   - After the string GA finishes, the full population is evaluated exactly once on the final candidate pool.
   - This is both cheaper and closer to the intended semantics: `inference_accuracy` now measures the recovery accuracy of the GA's final result rather than the best accidental intermediate candidate encountered at any earlier inner generation.

2. **MPS rule/index packing is reused across inner search generations.**
   - The genome set is immutable during string-side inference, so repacking and rebuilding the candidate index each inner generation was unnecessary overhead.
   - The search ensemble is packed once, reused for all inner generations, then the full population is packed once for final scoring.

3. **Global Rule Pool GC now keeps HoF genomes live during inference.**
   - The earlier inference call could trigger a pool compaction with only the current population as GC roots.
   - HoF-only rules could therefore be removed while archived genomes retained stale pool IDs.
   - Inference now passes the complete live population + Hall-of-Fame archive as GC roots.

4. **Pareto sorting itself was a host-side bottleneck and is now vectorized.**
   - The first implementation used nested Python loops with tiny NumPy comparisons for every genome pair.
   - At population=450 this measured about 0.79 s/generation on the audit machine.
   - A vectorized domination matrix produces the same ranks in about 0.016 s/generation in the same microbenchmark (~50× faster).

5. **NSGA-II crowding no longer gives arbitrary infinite crowding on a constant objective.**
   - Recovery accuracy is discrete and often ties, especially early in training or with `--no-inference`.
   - A zero-span objective is now ignored for crowding rather than assigning arbitrary boundary protection.
   - Exact boundary ties are handled symmetrically.

6. **Inner ensemble ranking uses average ranks for ties.**
   - Equal candidate scores are common before useful rules fire.
   - Previously, stable sort order silently broke ties and injected candidate-list order into evolution.
   - Tied candidates now contribute equal rank votes.

7. **The saturation graph now matches the requested three inference curves.**
   - gray: `inference raw`
   - dashed: current-generation `inference best`
   - solid: `inference MA(window)`
   - The cumulative `inference_best_ever` remains in CSV/history but is no longer substituted for the requested current best curve.

8. **Corpus loading bugs fixed.**
   - Long local chunks were accidentally dropped before the later crop branch could run.
   - The streaming GitHub loader referenced an undefined variable `part`.
   - Long chunks are now cropped as intended instead of discarded, and streaming uses the correct byte buffer variable.

9. **Small/unbounded evaluator edge cases fixed.**
   - Inference span no longer forces a 16-byte minimum beyond a smaller MPS raw stride.
   - MPS sample allocation now mirrors `RollingEvaluationSet`'s minimum sample/case coercions.
   - `--max-chunk 0` no longer collapses the MPS raw allocation when `--trajectory-len` is positive.

10. **Inference cases avoid the currently resident Spearman source chunks when possible.**
   - This further reduces accidental overlap between the rank-fitting batch and the second objective.

11. **Lexicase tie handling now respects the Pareto comparator.**
    - Case-specialist pressure is preserved, but if multiple candidates survive lexicase, the tie-break no longer ignores inference quality.

12. **Bundled validator scripts are portable again.**
    - `validate_recode_memo.py` and `validate_crossover_share.py` referenced an old absolute `/mnt/data/...v56.py` path and failed immediately outside the original scratch environment.
    - They now validate the packaged local `main.py` and both pass.


13. **Explicit elite preservation is now guaranteed in both genetic loops.**
    - The string-side GA exposes `--inference-elites` (default 3) and copies those candidates byte-for-byte into the next inner generation before any mutation/crossover.
    - Inner elitism is clamped to leave at least one offspring slot when the candidate population has more than one member; the old hard-coded 25% path could freeze a tiny population when every slot became elite.
    - The outer GA now builds survivor elites explicitly, pinning the Pareto knee plus the best Spearman and best inference genomes when enough `--elites` slots are available, then filling the remaining elite slots in Pareto order.
    - The elite survivors enter `next_pop` before QD/HoF injection and are never sent through mutation/crossover.

## Updated work budget

The checked inner GA uses an approximate full-population-equivalent work model:

```text
cases * candidate_population * (1 + search_generations * ensemble_size / population_size)
```

The `1` is the mandatory final full-population scoring pass. Search generations pay only for the voting ensemble.

With the default outer population 450, ensemble 64, 3 cases, 12 candidates, and 6 search generations:

```text
3 * 12 * (1 + 6*64/450) = 66.72 equivalent full-pop trajectories
```

Against a default 4×50 = 200-trajectory outer evaluation this is about **33% nominal extra rewrite work**, before taking into account that inference snippets are much shorter than normal trajectories. This leaves substantially more headroom below the requested ~2× wall-time ceiling than the first v60 implementation.

## Validation performed

- `python main.py --self-test` → PASS
- `python -m unittest -q test_improvements.py test_v58_search.py test_v59_rank_readout.py test_v60_inference.py` → 36 tests PASS
- CPU smoke A/B on the bundled checked code: inference-enabled run was about 1.23× the no-inference wall time for the small audit configuration used here.
- The mocked MPS orchestration regression test runs 12 generations and matches the CPU reference path.
- `validate_crossover_share.py` → PASS (1000 crossovers)
- `validate_recode_memo.py` → PASS (100k translated memo contracts + sentinels)

A real Apple MPS device is not available in this environment, so the Metal shader itself was not device-benchmarked here. The shader source was not changed by this audit; the changes are in host orchestration, budgeting, selection, and input handling.

## v60.1 — inferBest==0.000 root cause and fix

A reproducible zero-recovery failure was found after the elite-preservation revision.
This was not merely slow learning.

### Root cause: final argmax had a hidden no-op bias

The final per-genome inference result used `np.argmax(local, axis=1)`. Candidate 0 is
always the untouched noisy string. Replacer readouts are intentionally sparse and
quantized by rule-firing counts, so exact score ties between several candidate strings
are common. `np.argmax` always returns the first maximum, therefore a flat/tied genome
systematically selected candidate 0 and reported zero corrupted-byte recovery even when
a correct recovery was already present elsewhere in the candidate pool.

v60.1 evaluates the whole exact maximum-score tie set uniformly. This is the expected
accuracy of an unbiased uniform tie-break; the hidden clean target does not affect which
candidates enter the tie set. The ensemble result uses the same tie-safe rule.

A regression test explicitly constructs a flat-score model with a recoverable candidate.
The old behavior returns 0.0; the corrected behavior is strictly positive.

### Search sparsity was also unnecessarily severe

The old proposal distribution drew 72% of replacement bytes from the global byte
histogram and often mutated multiple corrupted positions at once. With roughly ten
corrupted bytes in a 192-byte snippet, 12 candidates and four generations, the clean
byte could simply fail to appear often enough for the second Pareto objective to become
a useful signal.

v60.1 therefore:

- builds a small byte-transition proposal model from the already-persisted n-gram pool;
- prefers context-conditioned replacement proposals without consulting clean targets;
- seeds the initial candidate population across corrupted loci round-robin;
- uses mostly one-locus point mutations so score changes are attributable;
- revisits corrupted loci systematically during offspring generation;
- raises the default inner search from 4 to 6 generations while retaining the existing
  work-cap calculation.

The transition proposal model is rebuilt from the checkpointed n-gram pool on resume,
so checkpoint runs do not silently use a different temporary proposal model.

### New diagnostics

Each generation now prints and stores:

- `infEns`: recovery of the ensemble-selected result;
- `infOracle`: best recovery present anywhere in the final candidate pool, diagnostic
  only and never used for search/selection;
- `infMove`: expected fraction of model/case selections that differ from the original
  noisy string.

Interpretation:

- `infOracle == 0`: proposal/search did not generate any correct byte yet.
- `infOracle > 0` but `inferBest == 0`: recoverable candidates exist but every model
  ranks them below its top set.
- `inferBest > 0`: at least one live genome's inference result recovers corrupted bytes.

### Reproduction

A small CPU smoke configuration reproduced the reported failure in the previous package:

```text
old gen=0 infer=0.000 inferBest=0.000
old gen=1 infer=0.000 inferBest=0.000
```

With v60.1 on the same small configuration:

```text
new gen=0 infer=0.167 inferBest=0.167 infEns=0.167 infOracle=0.333
new gen=1 infer=0.333 inferBest=0.333 infEns=0.333 infOracle=0.667
```

These numbers are a regression smoke test, not a claim about the long GitHub-Code run.

Validation after this fix:

- `python main.py --self-test` -> PASS
- 36 unit/regression tests -> PASS
- `validate_crossover_share.py` -> PASS (1000 crossovers)
- `validate_recode_memo.py` -> PASS (100k contracts + sentinels)

## v60.2 — log-uniform string mutation radius

Per request, the inner string GA's ordinary mutation radius is now sampled as `round(exp(uniform(0, log(k))))` for `k` mutable/corrupted loci, clamped to `[1,k]`. The existing focus scheduler only guarantees inclusion of its target and no longer reduces the radius to one locus. `force_changes` remains an intentional deterministic override for duplicate fallback. The mutable-position invariant is unchanged: no clean locus is eligible.
