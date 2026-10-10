# Before-selected-rule / final-sweep random sampling (October 10, 2026)

This release keeps the single-genome / one-rule hill climber, the permutation
LUT, the nine binary `$1`–`$4` operators, local edits, wildcard grammars,
8 evaluation chunks, and replacement of one of those chunks every 16 trials.

## Observation sources

For each selected rule and each sampled prediction step:

- **Pattern (left side)**: sample solely from the state immediately **before**
  the selected rule is evaluated.
- **Replacement (right side)**: sample solely from the **final state after
  the entire ordered rule list has run**, before the prediction slot is
  teacher-forced using the true target. This includes all later rules even
  when the selected rule itself does not fire.
- No new bytes are sampled from the original corpus independently of these
  recorded states or from teacher-forced target bytes.
- `--ngram-trace-samples` and `--ngram-trace-bytes` control the number and
  maximum observed bytes of snapshots; they never truncate the running state.

## How randomness works

Frequency enumeration and Top-5 selection were removed. For a chosen grammar,
the sampler uniformly draws one valid `(snapshot, start_position)` pair among
all such pairs, then copies that occurrence's bytes exactly. This is uniform
**over occurrences/positions**, not over unique n-grams. Wildcard blocks still
use randomly selected gaps and recorded literals. Local mutations use the same
random sampling for inserted bytes and spans; Hamming-distance-1 character
substitution uses randomly probed observed windows, never a histogram.

For a literal-only output candidate, the entire resulting literal string must
occur in at least one final-sweep snapshot. Candidate input patterns
must match at least one pre-rule snapshot. Preserving unchanged old literals
is allowed; newly sampled bytes always use their correct side's observations.

There are no `--ngram-top-k` and `--ngram-budget` arguments anymore. The sampler
is inexpensive: no full histogram of the snapshot bytes is built.

## Test

```bash
python -m unittest discover -p 'test_*.py' -q
python main.py --self-test
```
