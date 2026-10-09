# Replacer: `$1`–`$4` 二項演算 / binary capture operators

The existing unary opcodes `-1..-111` are unchanged. Nine new binary
operations are encoded in the **negative int16 range `-112..-255`**.
Only captures `$1`, `$2`, `$3`, `$4` may be referenced in binary operators;
existing unary operations still support up to `$16` as before.

## Encoding

```python
opcode = -(112 + 16 * kind + 4 * (left - 1) + (right - 1))
# left/right are 1-based capture indices in 1..4.
# main.binary_opcode(kind, left, right) implements this.
```

| kind | opcode range | name | Output |
|---:|---|---|---|
| 0 | -112..-127 | INTERSECT | Stable, distinct intersection in first capture's order |
| 1 | -128..-143 | DIFF | Stable, distinct elements of first capture absent from second |
| 2 | -144..-159 | CONV | Linear discrete convolution, output length `len(a)+len(b)-1` if neither empty |
| 3 | -160..-175 | ZIP_ADD | Pairwise `(LUT(a[i])+LUT(b[i])) mod 256`, truncates to shorter input |
| 4 | -176..-191 | ZIP_XOR | Pairwise `LUT(a[i]) xor LUT(b[i])`, truncates to shorter input |
| 5 | -192..-207 | EQUAL | `[1]` if entire byte sequences agree, else `[0]` (literal byte, not LUT-transformed) |
| 6 | -208..-223 | OVERLAP | Concatenate `a+b` while removing longest repeated suffix-of-a / prefix-of-b; O(len(a)+len(b)) KMP |
| 7 | -224..-239 | XCORR | Linear cross-correlation, lags `-(len(b)-1)..len(a)-1`, length `len(a)+len(b)-1` |
| 8 | -240..-255 | FILTER_IN | Keep elements from first capture found in second; **preserve duplicates and order** |

Each range packs all 16 ordered pairs (`$1..$4` × `$1..$4`). `INTERSECT`,
`DIFF` and `FILTER_IN` use raw token identity. `CONV`, `ZIP_ADD`, `ZIP_XOR`,
and `XCORR` perform arithmetic in the 256-token embedding coordinates and
map each result through the inverse LUT. Arithmetic is modulo 256 at each output
position. `EQUAL` and `OVERLAP` use raw byte sequences.

Examples (identity embedding):

- `$1 = [1,2]`, `$2 = [3,4]`: `CONV($1,$2) -> [3,10,8]`.
- `$1 = [3,1,3,2]`, `$2 = [3,2]`:
  `INTERSECT -> [3,2]`, `DIFF -> [1]`, `FILTER_IN -> [3,3,2]`.
- `$1 = [5,6,7]`, `$2 = [6,7,8]`: `OVERLAP -> [5,6,7,8]`.

A pattern must contain the requested number of `-1` wildcards; a binary
operator referring to `$4` is invalid in rules with only three captures.
`main.is_valid_rule()` enforces this; `repair_rule()` drops invalid opcodes.
The sequence wildcard matching semantics have not changed.

## Evolution and accelerators

- `make_rule()` can seed two-capture operators; `mutate_rule()` can create
  them as well, and crossover/diff3 transport their opcodes unchanged.
- `repair_evolved_rule()` conservatively keeps only the first binary operator
  per newly evolved rule to discourage uncontrolled sequence expansion.
  Manual rules and loaded checkpoints/models are not subject to that rule.
- All binary operations are implemented in **Python reference and C++17 native
  CPU** evaluators with equivalent semantics. C++ dynamically grows outputs.
- **Metal/MPS does not currently execute these new opcodes.** When an `mps`
  evaluation includes binary-operator rules, `main.py` selects native CPU
  instead, preventing GPU silent truncation or mis-decoding. Other MPS-eligible
  models use the original Metal implementation.
- The checkpoint/model schema version is unchanged (`4`); existing saved
  models remain loadable. Existing `-1..-111` operator semantics are unchanged.
- Convolution/cross-correlation cost `O(len(a)*len(b))`, which can be expensive
  on long captures. Output still obeys the model's unlimited-state principle,
  subject to physical RAM and time constraints.

## Run

Place `main.py`, `native_cpu.py`, `replacer_native.cpp`, and
`gpu_replace_persistent.py` together. Requires Python with numpy and optional
tqdm; native CPU accelerator compiles itself using a local C++17 compiler.

```bash
python -m pip install numpy tqdm
python -m unittest -v test_binary_operators.py
python main.py --self-test
python main.py --backend cpu --local-corpus example_corpus.txt \
    --population 12 --rules 12 --cases 2 --generations 2 --no-plot
```

The standard corpus is `github-code.txt`; the example is only for a smoke
run. Old checkpoints can be resumed in the existing working directory.
