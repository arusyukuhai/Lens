# Dual-Grid Lens — optimized matching and Fine-per-generation selection

Based strictly on `Lens_DualGrid_CoarseFine_Power.zip`. Ridge regression,
CMS mining, token alphabet, deterministic diffusion, the dual power grids,
firing-density feature semantics, Coarse screening, and Fine candidate selection
remain intact. No new geometric mean features or OLS conversion.

1. **Adjacent-pair fingerprint.** Extend each row's existing 8192-bit token
   presence fingerprint by an additional 8192-bit ordered neighboring-token
   fingerprint. A pattern can be rejected only if a mandatory literal or
   literal-literal adjacent pair is absent. This is a lossless prefilter;
   hash collisions cause only false positives. Disable with
   `--pair-fingerprint=0` to benchmark. An overhead is added when building
   fingerprints; it is not guaranteed to help for wildcard-heavy rules.
2. **Delta-suffix enabled by default.** Existing tested delta-suffix suffix
   state reuse is on unless `--delta-suffix=0` is passed. This can avoid
   rewriting unchanged suffixes during exchange screening.
3. **Avoid duplicate suffix mining input at end of pipeline.** When the
   insertion position is the end, reuse the encoded baseline's final texts
   rather than re-running the identical rule pipeline to construct `stageTexts`.
4. **Fine selection on every generation.** Coarse mines and shortlists; Fine
   ranks the shortlisted candidates and parents for every step, including
   non-`--eval-every` checkpoints. `--eval-every` only limits expensive
   diagnostic Train/Validation and plot generation. When the Fine and Coarse
   exponents agree, their rows and prefix state cache are reused.

No known approximation has been introduced into the final fixed-grid
candidate evaluation; however, Coarse screening is still a heuristic as in the
base version. Lossless filtering correctness does not guarantee faster runtime.

## Verify

```bash
python3 verify_pair_fingerprint.py
python3 verify_eval_schedule.py
python3 verify_dual_grid.py
python3 verify_incremental_pearson.py
nim c -d:release --opt:speed -d:lensProfile -o:lens cms.nim
./lens --self-test
```

Run identical configurations with `--pair-fingerprint=0` and `1` separately;
compare `[profile] exact_match_seconds` and `[profile-step]` rather than
isolated rejection ratios. The supplied environment has **no Nim compiler**, so
binary test results and speed gains are not claimed.
