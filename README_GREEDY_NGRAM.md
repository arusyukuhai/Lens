# Lens: one-genome, n-gram-guided greedy training

This version **replaces the population-based training loop** with a single active
`Genome` that is mutated one `Rule` at a time. It retains the existing rule
syntax, native C++ evaluator, 256-element permutation LUT, and all nine binary
operators (`-112` through `-255`, referencing `$1`–`$4`). The older GA helper
functions remain in `main.py` for code/API compatibility, but `evolve()` **does
not call them**. There is no selection tournament, crossover, HOF, or immigrant
population in training.

## Start

```bash
python -m pip install numpy tqdm matplotlib
python main.py --local-corpus github-code.txt --backend cpu
```

The corpus is separated by literal `===SPLIT===` delimiters. Chunks exceeding
`--max-chunk` are **skipped**, not cropped. Compilation of the C++17 backend
happens locally on first use; if no compiler is available, the correct but much
slower Python reference is used.

Defaults:

- `--rules 1500`: initial rules, then **only one** rule mutated per trial.
- `--cases 8`: eight full chunks evaluated for both baseline and proposal.
- `--case-rotate-every 16`: exactly one of those eight cases replaced after
  every 16 proposals, in round-robin order.
- `--ngram-top-k 5`: candidates sampled uniformly from the five most frequent
  distinct motifs of the chosen structural shape (not one global Top 5 across
  incompatible pattern shapes).
- `--ngram-max-blocks 6`: up to six literal blocks in a pattern, separated by
  variable-length wildcard capture tokens. This limit is configurable.
- `--ngram-max-literal 12`: random per-block widths 1–12, or up to 24 for
  contiguous patterns; the classic 2–5-gram patterns are also sampled.
- `--ngram-trace-samples 6`: six time points sampled from each chunk.
- `--ngram-trace-bytes 384`: inspect the last 384 bytes of each traced recurrent
  state. This limits **observations**, never the running model's state.
- `--ngram-budget 100000`: max inspected motif windows; with default trace
  sizes this is above the number of windows and counts are exact.

Per trial, a selected rule's **actual input states immediately before it is
applied** and the **result states after the full rule sweep** are collected
while evaluating the current genome. The input-side motif is sampled from the
former; the output-side literal n-gram is sampled from the latter.

Four kinds of single-rule changes are possible:

1. Change just the pattern (`input`).
2. Change just the replacement (`output`).
3. Change both (`both`).
4. Extend the existing pattern by prepending or appending `wildcard + observed
   bigram` (`extend`).

Classic grammars include `2`, `3`, `4`, `5`, `2+W+2`, `3+W+3`, and
`2+W+2+W+2`. The generalized grammar can draw 1–6 literal blocks with random
positive widths and random observed separation lengths. In the encoded rule,
`W` is still the original `-1` **variable-length capture wildcard**, not a
new fixed-gap operator. Existing $n references and numerical/binary operations
remain usable. Replacement strings use actual observed output literals, with
occasional `$n` splice when valid. `repair_evolved_rule` still guards against
rampant self-duplication; output literals cannot simply grow without the
corresponding literal context.

**Acceptance:** candidate and active genome are scored on the identical current
batch of eight chunks. A candidate replaces the selected rule iff its
teacher-forced next-byte accuracy is *strictly greater*. Equal or worse scores
are rejected. At a case-rotation boundary the baseline is recalculated on the
new eight-chunk batch; scores from distinct windows are never directly compared.
Consequently the accuracy chart may drop on rotation even though no accepted
mutation made the incumbent worse on its own evaluation data.

The 256-byte LUT is preserved exactly in an ongoing greedy run. It is used by
numeric unary and binary operators as before. It is not mutated in this
single-rule-only training mode. To continue from an existing trained LUT,
load a saved v4 model:

```bash
python main.py --local-corpus github-code.txt --load best_lens_ar.json
```

A legacy v4 GA checkpoint can also be imported as a *starting genome*, but its
old population and fitness history are not kept. New greedy checkpoints are
reproducible and contain only the active genome, RNG, rolling-case positions,
trial number, and progress history:

```bash
python main.py --local-corpus github-code.txt --generations 200 --checkpoint greedy.pkl
python main.py --local-corpus github-code.txt --generations 400 --load greedy.pkl --checkpoint greedy.pkl
```

Note: the plotted **best observed** accuracy is over different changing case
windows and should not be interpreted as a fixed validation-set benchmark.

## Test

```bash
python -m unittest discover -p 'test_*.py' -q
```

The extra `test_ngram_greedy.py` checks complex generated grammars,
frequency ranking, native/reference state-trace equality, invariants, and
saved-model import. Existing native-vs-Python, binary operator, checkpoint
restart, and rotating-chunk tests are retained/updated.
