# v64: hidden phrase memory

v64 adds a persistent two-list phrase memory that interacts with the ordinary Replacer sweep. The visible rewrite system remains ordered and non-expanding; hidden reads are dynamically clipped/skipped rather than allowing a rule to grow the visible state.

## Pattern tokens

- `-1`: ordinary wildcard capture.
- `-2`: wildcard capture plus hidden-memory write at the **top**.
- `-3`: wildcard capture plus hidden-memory write at the **bottom**.

Within one matched rule, the first two `-2` captures form a left/right pair inserted at the top. If only one exists, the right phrase is empty. The first two `-3` captures behave the same way at the bottom. Extra captures remain usable as ordinary captures but are not written to hidden memory.

The two hidden lists each hold at most 32 phrases. On the MPS path their phrase tokens are stored in a bounded packed arena; pressure evicts entries from the opposite end of the deque instead of overflowing evaluator scratch.

## Replacement hidden opcodes

Existing capture/transform opcodes remain `-1..-111`. v64 reserves `-128..-151` for 24 hidden-memory operations. Every opcode selects:

1. top or bottom,
2. first or second entry (`second` falls back to first if only one exists),
3. left or right list,
4. `read`, `pop` (read + remove), or `delete` (remove without output).

Use `encode_hidden_opcode(...)` / `decode_hidden_opcode(...)` rather than hard-coding numbers.

## End-of-sweep hidden rewrite

After the ordinary ordered rule sweep, current left/right entries are paired by position and interpreted as persistent `left -> right` literal rewrites. Empty sides are skipped. To preserve the engine's hard non-expanding state invariant, mappings with a longer right side are skipped. One pair receives one pass; N pairs receive up to `min(N, 8)` passes, stopping at a local fixed point. This phase never consumes the hidden lists.

## Loop detection

Cycle identity is now `(visible state, hidden left list, hidden right list)`, not visible state alone. Therefore a visible string may repeat while hidden memory continues to evolve without being falsely terminated as a cycle. The ordinary `ceil(2*sqrt(input_length))` round bound remains the ultimate safety bound.

## Evolution and migration

Mutation can now introduce `-2/-3` pattern captures and hidden opcodes. v64 checkpoints use version 2. Loading a version-1 checkpoint converts every legacy negative pattern token to `-1` first, because in those revisions `-1..-16` were all semantically identical wildcards; this prevents old `-2/-3` spellings from silently acquiring new side effects.

`test_v64_hidden_memory.py` covers all 24 opcodes, top/bottom capture behavior, pop/delete behavior, persistent hidden rewrites, non-expansion, and the hidden-aware cycle check.
