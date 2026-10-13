# Lens v6 — Generalization-first / exact hot-loop optimizations

This version builds on **v5 ReferenceStyle Log2Plot**. It is an experimental scientific iteration: the effects on new-source Spearman are *hypotheses*, not measured improvements.

## Changes

1. **Seeded corpus permutation at source-chunk level** (new runs default `--shuffle-corpus:true`). The previous splitting was based on the file order: most of the training set came from the beginning of `github-code.txt`, validation/hold-out from the tail. If the corpus is sorted by repo/language, that creates distribution shift that swamps other improvements. The new permutation is reproducible, and partition boundaries remain disjoint. Duplicate texts across source chunks are *not* yet deduplicated; leakage is still possible for duplicate snippets.
2. **Stable candidate ranking across independent search chunks**: compute per-case Spearman, and rank by `mean Spearman - stabilityPenalty * standard deviation`. `--stability-penalty:0.35` default; `--stability-penalty:0` restores the old mean-only objective. **Acceptance remains based on raw Spearman on independent samples**, with both validation sets. It never uses the protected holdout for model selection.
3. **Shared corruption alphabet** (`--shared-noise-alphabet:true`): sample a vocabulary of source symbols exclusively from up to 256 training-pool chunks, then use it for Train, Search, Acceptance, Anchor and Held-out noise. Earlier versions derived a different alphabet for each data pool, so validation measured both novel text generalization *and* changing corruption distributions. `false` restores the old behavior and is automatically used for legacy checkpoint resume.
4. **Exact first-anchor inverted index in `batchPresence`**. Candidate can only match if its first literal occurs; test only candidates with observed first literals, while keeping full gap and overlap semantics. This preserves results, although the speedup depends on how selective the first anchors are.
5. **Bounded per-beam mining cache** (8 exact transformed clean corpora per lookahead): reuse mined candidates and compiled batch matcher when independent parent rule chains produce *identical transformed clean texts*. No approximate hashing, cache lives only inside a single addBeam call. `[mine cache] hit` will appear in log.
6. The previous **v5 frozen champion / distinct source-chunk acceptance and heldout**, rollback conditions, 4096-token limit and `-log2(1-Spearman)` gnuplot are unchanged.

## Migration / run

```bash
nim c -d:release --opt:speed cms.nim
nim c -r -d:release test_v6_optimization.nim
nim c -r -d:release test_fast.nim
nim c -r -d:release test_validation.nim
nim c -r -d:release test_generalization_v5.nim
./cms --corpus:github-code.txt --out:lens_run_v6 --iterations:100 \
  --beam:4 --lookahead:2 --holdout:8 --shuffle-corpus:true \
  --shared-noise-alphabet:true --stability-penalty:0.35
```

If the priority is to compare the **v5 scoring rule while keeping the new split**, use `--stability-penalty:0`. For a clean ablation of the split itself, also run `--shuffle-corpus:false`, on a different output path and with the same seed. **Do not compare random-split v6 heldout numbers directly to contiguous-split v5 heldout numbers**; they are different test sets. Rerun v6 variants with the same `seed` and split.

A v5 checkpoint resumes with its original file-order split and noise process (`shuffleCorpus=false`, `sharedNoiseAlphabet=false` set on migration). A v6 checkpoint saves the new permutation setting and disallows changing seed/corpus/4096-token cap in mid-run if shuffled. Resume will refit *exploratory* models but not the frozen champion weights.

## Known limitations

- Nim compiler and full corpus are not available in this execution environment: compilation, runtime benchmarks, and improvement in true held-out accuracy **have not been demonstrated**. All supplied tests must be run on your machine.
- Frequency mining still enumerates many wildcard-gap combinations. Large O(1e9) occurrence cases can remain expensive even after caching. `--max-candidates` only caps the expensive **scoring** stage, not the underlying exhaustive occurrence mining.
- Search utility is a heuristic, not a rigorous lower confidence bound. With small sample counts, noise rank correlations can be unstable. Keep the protected holdout blind to tuning.
- Exact duplicate or near-duplicate chunks may occur across pools; grouping by repo/file for splitting would be a useful future improvement if metadata becomes available.
