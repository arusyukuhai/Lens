# Lens — multi-strategy guided rule discovery / diff3

## Changes relative to Lens_Pearson_DirectedRules_InferenceGA_20261011

The former default `--learning-mode guided` used only directly synthesized
literal repair rules. It now uses **six genuinely distinct source generators**:

| Name | Proposal geometry | Notes |
|------|-------------------|-------|
| `repair` | observed fixed-length correction, 2–5-byte local context | Original guided lane retained |
| `wildcard` | generalize an unmodified *flank* into wildcard 1 and copy its capture on RHS | Corrupted center is never replaced by a wildcard |
| `ngram` | attach observed 3–7-byte left/right context from source training examples | Context comes from the actual training corruption, not random unrelated n-grams |
| `diff3` | three-way patch merge: base literal repair → wildcardized donor, apply non-conflicting edit to widened contextual target | Uses `main.merge_sequence_delta`, the established `SequenceMatcher`-based git-style categorical differential operator; no fabricated numeric arithmetic over token IDs |
| `one_token` | single-token edit of a pre-existing genome rule, guided by bytes from training corruption / clean repair | Does not require the rule to have been directly generated from the current repair |
| `splice` | prefix/suffix splice of two locally grounded repair contexts | Keeps the corrected byte centered within the resulting rule |

The proxy ranker uses training-only cleanliness residuals from standard
unregularized **linear regression**, penalizes controls that already match the
rule trigger, and downweights rule length. Wildcard presence is approximated by
its longest contiguous literal anchor: it is **not** an exact firing count or
proof that a rule will help. The standard native sequential evaluator measures
each proposed child on the next outer generation, as before. The ordinary
clean-source held-out Pearson fitness and separate reconstruction metric remain.

No change to `--inference-mode ga`: **the inner reconstruction GA is retained**.
`--learning-mode ga` still runs the historical external GA and its three-parent
crossovers. `--learning-mode hybrid` mixes the new guided lanes with that GA.

## Suggested CLI

```
python -m pip install numpy scipy threadpoolctl
python main.py \
  --backend cpu-native --cpu-threads 8 --blas-threads 1 \
  --learning-mode guided --readout-mode linear-pearson \
  --inference-mode ga --diffusion-mode dual \
  --guided-strategies repair,wildcard,ngram,diff3,one_token,splice \
  --guided-candidates 48 --guided-source-samples 48 \
  --cases 3 --samples 100 --population 96 --rules 1500 \
  --hof-size 4096 --hof-eval 16
```

`--guided-strategies` accepts any nonempty comma-separated subset of the names,
for example `--guided-strategies repair,diff3,one_token`. To weight them, pass
`--guided-strategy-weights 0.25,0.5,0.25` in that exact selected strategy order.
If omitted, defaults are repair 29%, wildcard 16%, ngram 15%, diff3 17%,
one_token 14%, splice 9%, renormalized over enabled modes. This weights the
number of *attempted offspring*, not a guarantee of fitness contributions.

`--guided-candidates` now caps candidates **per strategy**, not combined. The
proxy miner takes an explicit bounded source sample and caps proxy evaluations
per lane; generating more candidate varieties still incurs additional CPU work.
Try `--guided-candidates 16` on resource-limited hardware first.

## Accounting

Every `fitness_history.csv` generation now carries
`guided_candidates_<strategy>`, `guided_children_<strategy>`,
`guided_prev_evaluated_<strategy>`, `guided_prev_improved_<strategy>`, and
`guided_prev_reward_<strategy>` for all six strategies. Rewards use overlapping
source-chunk scores between each child and the champion parent from which it
was created; they do **not** require an extra Replacer evaluation. The first
generation has no previous guided children to evaluate.

Standard logs also show the number of proposals and guided children per lane.
Unchanged champion elites and HOF preservation remain, and all children keep
copy-on-write rule-pack semantics (one modified row, untouched rules shared).

## Testing and caveats

- `python -m pytest -q` (130 tests pass, 4 MPS-dependent skipped, plus 10 subtests in this container).
- `python main.py --self-test` passes.
- A 3-generation CPU-native, inner-GA smoke run was completed, all six lanes
  produced children; this is a correctness check, **not** a long-run quality
  improvement or Apple Silicon speed claim.
- `diff3` uses **three related rule variants from one observed training error**
  (`base`, `donor`, `target`), not three randomly sampled whole genomes. The
  historical whole-genome three-parent diff merge remains in `--learning-mode
  ga/hybrid`.
- Early source-holdout protects the within-generation readout and rule mining;
  the outer algorithm still repeatedly selects structures using held-out
  Pearson, so publication-quality evaluation needs a truly untouched corpus.
- The scoring proxy is cheap and approximate; many synthetic candidates may
  be useless. Per-lane contribution counters make ablation studies possible.
