# Legacy implementation notes (v2-v4)

Historical behavior only. See README.md for the v5 behavior.

# Lens — source-disjoint training and dependency-aware regrowth

This ZIP builds on the 4096-token chunk-filtered, memory-guarded and gnuplot-fixed Lens implementation. `cms.nim` is the entry point. The new files `lens_validation.nim` and `lens_dependency.nim` are required.

## Build / run

```sh
nim c -d:release --opt:speed cms.nim
./cms --corpus:github-code.txt --out:lens_run_v2 --iterations:100
```

To migrate an **existing checkpoint**, overwrite all source modules with this bundle and rebuild. The program accepts old format-1 checkpoints and writes format-2 checkpoints that include the independently accepted champion. The corpus SHA-1 and tokenization must match. Old checkpoint scores are **not** interpreted as out-of-source validation scores; the newly disjoint validation begins upon resumption.

```sh
./cms --resume:lens_run/checkpoint.json --iterations:100
```

## What changed

**Disjoint source text pools.** A source chunk never appears in more than one pool by index: training (rotating window + optional rehearsal), search-validation (used to rank candidates), acceptance-validation (used to approve an updated champion), and fixed heldout (monitor only). Note: exact or near-duplicate *content* may exist across pools if the corpus includes duplicates; deduplication is not performed. Search/accept use independent corruption seeds. The champion and the beam are refitted on current training rows each iteration; no regression weights are fitted on the validation rows. A reused acceptance reservoir is not a permanently unbiased final test set: the fixed heldout also must not be used for iterative hyperparameter tuning.

**Dependency-Aware Regrow (not suffix-wide regrowth).** The v4 controller triggers on acceptance stagnation (see below), not every 4 iterations:

1. Shortlist up to 4 potential root positions, chosen via phi leaders and rotation. Rank these by FULL replay score with the position ablated. This is a heuristic to find root candidates, **not** a claim that ablation exactly measures causal importance.
2. For the best root (default 1), re-mine alternatives from its unchanged prefix; rewrite **only that root** initially, keeping its RHS marker allocation intact.
3. Replay the baseline chain and the revised chain on identical, trajectory-spaced training samples. Measure each later rule's **presence-mask difference** and **local rewrite effect-signature difference**, not distance from the edited root.
4. Try rebuilding the most affected downstream rule first, even if several unaffected rules lie between it and the root. Recompute influence after each wave because repairing one rule can change dependencies further downstream. Default at most 2 waves, 2 candidate states in the local frontier. **Rules unaffected on sampled input are left unchanged.**
5. Full-replay and refit the resulting state, then retain the candidate only if its search-validation score improves; independently evaluate it again on acceptance chunks before promoting the persistent champion.

This is a bounded, empirical dependency heuristic. Local rewrite signatures are not a proof of causality, and sampling may miss rare effects. It favors preserving unaffected rules and offers a much smaller search than discarding an entire suffix. A changed rule's RHS is kept fixed by default when choosing a replacement LHS, preserving downstream token naming; more complex RHS mutations are still explored by the ordinary beam's lookahead, not by surgical regrow.

## Common options

- `--window:3`: rotated train source chunks.
- `--replay-chunks:1`: additional train source chunk for rehearsal; 0 disables.
- `--search-pool:32 --search-chunks:1`: reserved searchable source pool size and sources evaluated per iteration.
- `--accept-pool:64 --accept-chunks:2`: separate acceptance pool size and sources evaluated per iteration.
- `--holdout:2`: fixed monitor sources; never participates in model promotion.
- `--beam:4 --lookahead:2`: state hard cap and new-rule lookahead.
- `--regrow-every:4`: minimum number of iterations between rollback attempts; 0 disables.
- `--regrow-patience:3`: minimum consecutive non-promotions on source-disjoint acceptance before rollback can start.
- `--regrow-max-loss:0.04`: ignore roots whose full-replay ablation loses more than 0.04 search Spearman on an initial plateau; after 2× patience the restriction relaxes.
- `--regrow-depth:2`: maximum impacted-rule waves after editing root (not contiguous suffix length).
- `--regrow-positions:1`: root positions to explore after ablation shortlist; 0 disables.
- `--regrow-ablate:4`: number of candidate root positions to ablate.
- `--regrow-beam:2`: local regrow frontier cap.
- `--repair-candidates:16`: per-position replacement shortlist cap; 0 is exhaustive.
- `--accept-margin:0.002 --rule-penalty:0.001`: acceptance merit = Spearman rho - penalty * rule count.
- `--max-chunk-tokens:4096`: ignore entire source chunks exceeding this; 0 disables.
- `--trajectory-limit-mib:1024`: guard minimum train+search+accept snapshot footprint; 0 disables.
- `--noise-steps:0`: all single-character stages; e.g. 64 explicitly subsamples steps to reduce memory.
- `--bytes:false`: Unicode-codepoint default; true means bytes.

The top-level selected model contains regression weights refitted on the current train window. A rule can be in the exploratory beam even if the independently accepted champion does not change.

## Results / plotting

`training_v2.tsv` records per-iteration train/search/acceptance/heldout correlations and acceptance/regrow events. It is a *new schema* and is intentionally **not** appended to old `training.tsv`. `training.gnuplot` plots raw Spearman rho in two panels: out-of-source monitor, and rotating-window training/search diagnostics. Only the *fixed heldout* time series compares the same source chunks from one point to the next (but its repeated inspection is still a tuning risk).

```sh
cd lens_run_v2
gnuplot training.gnuplot
```

`model.json`, `checkpoint.json`, `config.json` and `training.png` are also emitted. When available, gnuplot renders a staging PNG and keeps the previous image if render fails.

## Validation

```sh
nim c -r -d:release test_validation.nim
nim c -r -d:release test_lens.nim
nim c -r -d:release test_chunk_limit.nim
```

`test_validation.nim` includes checks for disjoint source indices, acceptance gating, rollback behavior, and a nonlocal dependency case: editing R1 changes R4's firing without changing R2/R3. A second case checks changed capture content despite identical firing masks. The clean text corpus is not included.

**Build limitation:** The ZIP was prepared in an environment without Nim and gnuplot executables. The source and tests are included, but compilation, plotting, and performance measurements have not been executed here. Please compile locally before relying on a long run.

## Regrow v3: balanced, nonlocal, delayed-pruning dependency exploration

This revision keeps the disjoint Train/Search/Acceptance/Held-out split and
checkpoint format 2. It refines **only the nonlocal regrow transaction**:

- `--regrow-roots:4`: retain up to four distinct candidate *root edits*
  before their downstream consequences have been repaired. The regular beam
  remains bounded by `--beam`; root alternatives are transient.
- `--regrow-targets:2`: inspect the two highest-drift *unvisited* rule indices
  per path and wave, without requiring adjacency. Each may spawn a separate
  branch. Previously only the single top target was attempted.
- `--regrow-beam:2`: keep the highest-scoring path **and, where available, a
  different path that actually performed downstream regrowth**, so a
  temporarily bad intermediate repair is not immediately discarded.
- `--regrow-samples:64`: evaluate empirical rule influence on samples spread
  evenly across independent training trajectories/chunks first, then through
  each trajectory's noise stages. Previously, long chunks dominated.
- The per-rule local rewrite fingerprint no longer records absolute match
  positions: inserting tokens *before an unrelated match* should not by itself
  be interpreted as evidence that the downstream rule's effect changed.
- `--regrow-mines:8`: no more than eight *distinct mining prefixes* are
  exhaustively enumerated in a single regrow transaction. Reused cached
  prefixes are free, and `0` disables this budget.
- A per-transaction cache avoids repeating identical influence measurements.

Importantly, this is still an **empirical heuristic**, not a proof of
causality. The local rewrite signature can conflate multi-match changes and
unrelated overlapping edits. A retained candidate always gets full chain
replay, regression refitting, search ranking, and separate acceptance checks.
A fixed held-out monitor is not used for selecting regrow paths.

Recommended starting parameters:

```sh
./cms --resume:lens_run/checkpoint.json --iterations:100 \
  --regrow-every:4 --regrow-roots:4 --regrow-targets:2 \
  --regrow-depth:2 --regrow-beam:2 --regrow-samples:64 --regrow-mines:8
```

With expensive corpus mining, use `--regrow-roots:2 --regrow-targets:1
--regrow-mines:4` before increasing the main beam. `--regrow-targets:1`
is closer to the former single-target strategy; `--regrow-roots:1` is strict
but can prematurely eliminate useful temporary root mutations.

The newly added tests in `test_validation.nim` cover case-balanced sampling,
relative-match-position invariance, and non-adjacent capture-sensitive drift.
They must be run on a machine with Nim available.


## Rollback v4: data-driven trigger and conservative root screening

The original v3 unconditional `(iteration+1) % regrowEvery == 0` could
waste many expensive mining/replay passes on a model that is already improving.
Worse, it repaired the highest **search-validation** exploratory candidate,
not necessarily the accepted champion.

Now the normal beam runs and is assessed on independent acceptance source
chunks **first**. Consecutive iterations with no promotion increase
`noPromotionStreak`, and a promotion resets it. Rollback starts only if:

1. At least 2 existing rules and regrow is enabled.
2. At least `--regrow-patience:3` consecutive iterations without champion promotion.
3. At least `--regrow-every:4` iterations since the last attempt (no previous attempt means no cooldown).
4. Full-replay ablation of a candidate root loses at most
   `--regrow-max-loss:0.04` in *search* Spearman. If the plateau lasts at least
   `2 * regrowPatience`, allow a broader search (still restricted by the
   root and mining budgets).

The **accepted champion**, not an unapproved search champion, is the
rollback root. A changed dependency path must improve the full-chain
search score to become a proposal, and independent acceptance still decides
whether it can replace the champion. Protected heldout results are never
read by the scheduler or model promotion. No hard reset/deletion of existing
rules takes place: changed roots retain their marker naming; affected
nonadjacent rules are selectively re-mined.

The small `rollback` controller state is persisted in format-2 checkpoints
as optional metadata. Older format-1 or format-2 checkpoints without this
metadata resume with a fresh counter. `--regrow-every:0` disables rollback.

Tests: `nim c -r -d:release test_rollback_policy.nim` plus the previous
`test_validation.nim` and `test_lens.nim`.

Caveat: rotating acceptance chunks are a **model selection sample**, not a
never-seen statistical test. A promotion streak is a practical heuristic,
not proof of a causal plateau; independent fixed heldout should remain only
a monitor and must not be used to choose the rollback schedule.


## v6 (2026-10-13)

- Optional seeded randomized split of eligible chunks, default true for fresh runs, false on old checkpoint migration.
- Shared train-derived corruption symbol pool, default true for fresh runs, false on old checkpoint migration.
- Candidate search ordering uses mean case Spearman minus `stabilityPenalty` times the across-case standard deviation (default 0.35). Raw Spearman and independent validation acceptance thresholds are unchanged.
- Exact candidate-first-literal index accelerates batch pattern presence evaluation; bounded per-lookahead mining and AC matcher cache reuses identical transformed clean text.
- **Not empirically benchmarked** (Nim compiler and original corpus unavailable here). See OPTIMIZATION_V6.md for run commands and ablations.
