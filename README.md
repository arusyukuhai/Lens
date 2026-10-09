# Lens v14 — variable-length Replacer + refined GA / three-parent diff3

This build combines v13's improved mutation / crossover and v12's one-case-at-a-time rotation with a **variable-length rule and recurrent-state implementation**.

## Binary `$1`..`$4` operators (new in this patch)

See **[README_BINARY_OPS.md](README_BINARY_OPS.md)** for opcode layouts,
semantics, mutation support and runtime costs. Nine operations now occupy
`-112..-255` without changing the old unary opcodes or saved-model format.
Python and native C++ evaluate all nine; Metal selects native CPU fallback when
any binary operator is present.

## What changed

- **No hard 64-token rule limit for CPU/Python.** Patterns and replacements may exceed 64 tokens; the evolutionary operators can gradually grow them through insertion, context extension, duplication, and diff3 sequence merges. The original `MAX_RULE_TOKENS=64` restriction only remains as a compatibility limit **inside the legacy Metal fast path**.
- **Output is no longer forced to be nonexpanding.** Literal outputs can exceed the number of input literals, and a capture may be emitted repeatedly. Rule evaluation never silently truncates a replacement to its matched span.
- **Native C++ evaluator now supports dynamically growing recurrent-state buffers**, without the old `input_length+1` capacity assumption. Rule packing also uses variable row strides instead of the former fixed `rules × 65` layout. The reference Python evaluator uses identical semantics.
- **Genome validation and saved-model loading** accept longer / expanding rules. Existing v12/v13 autoregressive checkpoints and models remain loadable (checkpoint version unchanged). Evaluation results for unchanged nonexpanding rules are preserved.
- **Conservative evolution, expressive execution:** unrestricted expansion is valid for hand-written/saved programs, but the *random genetic operators* preferentially make small length increases in prediction-slot-conditioned rules and avoid creating repeated-capture exponential explosions accidentally. The ability to grow repeatedly over future generations is unlimited; this is not a hard rule-length cap.
- **MPS fallback:** the pre-existing GPU kernel is fixed-stride and nonexpanding. When rules are too long or can expand, `--backend mps` automatically evaluates that generation on native CPU rather than risking a GPU buffer overflow or incorrect score. On Mac, `--backend cpu` is the default, also reflecting measured CPU superiority on this workload.
- **Training log:** `ruleLen=<max-pattern-length>/<max-replacement-length>` shows the longest current rule in each generation.

## Preserved design

- 450 rules per genome by default, 450 population members, byte tokens 0..255 and reversible LUT within `sort/+1/-1/*2//2` and numeric binary operators.
- One ordered rule sweep per predicted byte; the resulting state is preserved, teacher-forced at the final prediction slot, then appended with a fresh slot.
- No fixed recurrent-state/context length, no probability distribution, and no automatic truncation of the internal state.
- `===SPLIT===` chunk delimiter; complete chunks of **1500 bytes or longer** are skipped (configurable with `--max-chunk`).
- `case_rotate_every` replaces only one evaluation case at each boundary; its RNG state survives checkpoints.
- tqdm evaluation/generation progress, caching, and refined mutation / two-parent and diff3 three-parent crossover from v13.

## Install and run

```bash
python -m pip install numpy tqdm
python main.py --backend cpu --local-corpus github-code.txt \
    --rules 450 --population 450 --cases 8 --case-rotate-every 3
```

For an existing Lens directory, replace **`main.py`, `native_cpu.py`, and `replacer_native.cpp` together**. The `gpu_replace_persistent.py` in this archive is the unchanged compatible Metal implementation. Keep all four in the same directory. Native CPU uses a locally available C++17 compiler and caches its compiled binary by a source hash.

## Verify

```bash
python -m unittest discover -q
python main.py --self-test
```

The additional `test_dynamic_lengths.py` covers 170+ token patterns, 220+ token replacements, repeated captures, native CPU/Python equality with expanding state, save/load, and MPS fallback.

## Resource note

"No artificial memory limit" does **not** imply physically infinite memory. A rule that duplicates a long capture can grow a state exponentially and exhaust RAM / take arbitrarily long; it will not be silently clipped. Such a manual rule is valid but should be used cautiously. When C++ allocations fail, a Python `MemoryError` is reported where possible. By design, the ordinary evolutionary operators discourage explosive rules without imposing a model-level state ceiling.
