"""
GPU-oriented replacement engine compatible with the Replace semantics in
at_jev(20261001-230241).nim.

Backends: PyTorch CPU / CUDA / MPS (same function).

Design:
  1. Evaluate every possible start position in parallel on the selected device.
  2. Perform the inherently sequential greedy non-overlap selection on CPU
     using only compact match metadata (one device->host sync per Replace).
  3. Turn copy/replacement operations into variable-length output segments.
  4. Compute segment offsets with a prefix sum and emit most output positions
     in parallel on the GPU. sort($n) is emitted with torch.sort per segment.

This is intentionally a standalone correctness/prototyping implementation.
The batch API groups equal patterns, amortizes greedy-selection metadata syncs,
and emits successful jobs into one packed variable-length output. A native
CUDA/Metal greedy kernel can still remove the remaining host interval walk.
"""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch


EMBEDDING_CODE_COUNT = 512
MAX_WILDCARDS = 16
MAX_REPLACEMENT_OPS = 64
DEFAULT_MAX_OUTPUT = 1_048_576

# Emission segment kinds.
_COPY = 0
_LITERAL = 1
_CAPTURE = 2
_SORT = 3
_REVERSE = 4
_PLUS1 = 5
_MINUS1 = 6
_TIMES2 = 7
_DIV2 = 8


@dataclass(frozen=True)
class _CompiledPattern:
    parts: Tuple[Tuple[int, ...], ...]
    wildcard_count: int
    has_leading_literal: bool
    all_wildcard: bool
    empty_pattern: bool


@dataclass(frozen=True)
class _ReplacementOp:
    kind: int
    arg: int = 0
    value: int = 0


@dataclass
class ReplaceResult:
    # True when at least one pattern occurrence was found.
    matched: bool
    # True only when a valid output was produced. Overflow makes this False.
    applied: bool
    output_changed: bool
    overflowed: bool
    # On no-match/overflow this is an empty tensor, matching replaceSeqCompiled.
    data: torch.Tensor
    selected_matches: int = 0


@dataclass
class BatchReplaceResult:
    """Packed result of replace_batch_gpu.

    results[i].data is a zero-copy view into packed_data for applied jobs.
    output_offsets/output_lengths let callers keep the entire batch packed on
    the accelerator without constructing Python lists.
    """

    results: List[ReplaceResult]
    packed_data: torch.Tensor
    output_offsets: Tuple[int, ...]
    output_lengths: Tuple[int, ...]

    def to_lists(self) -> List[List[int]]:
        """Copy packed output to CPU once and split it by job."""
        host = self.packed_data.detach().cpu().tolist()
        out: List[List[int]] = []
        for r, off, length in zip(self.results, self.output_offsets, self.output_lengths):
            out.append(host[off:off + length] if r.applied else [])
        return out


def _pick_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _compile_pattern(pattern: Sequence[int]) -> _CompiledPattern:
    parts: List[Tuple[int, ...]] = []
    current: List[int] = []
    wc = 0
    for raw in pattern:
        x = int(raw)
        if x == -1:
            parts.append(tuple(current))
            current.clear()
            wc += 1
        elif x < -1:
            raise ValueError(f"invalid pattern token {x}: only -1 is a wildcard")
        else:
            current.append(x)
    parts.append(tuple(current))
    has_leading = len(parts) > 0 and len(parts[0]) > 0
    all_wildcard = wc > 0 and all(len(p) == 0 for p in parts)
    empty = wc == 0 and len(parts) == 1 and len(parts[0]) == 0
    return _CompiledPattern(tuple(parts), wc, has_leading, all_wildcard, empty)


def _compile_replacement(replacement: Sequence[int]) -> Tuple[_ReplacementOp, ...]:
    if len(replacement) > MAX_REPLACEMENT_OPS:
        raise ValueError(
            f"replacement has {len(replacement)} ops; GPU prototype supports the same "
            f"hot-path limit of {MAX_REPLACEMENT_OPS}"
        )
    ops: List[_ReplacementOp] = []
    for raw in replacement:
        t = int(raw)
        if t >= 0:
            ops.append(_ReplacementOp(_LITERAL, value=t))
        elif t >= -15:
            ops.append(_ReplacementOp(_CAPTURE, arg=-t - 1))
        elif t >= -31:
            ops.append(_ReplacementOp(_SORT, arg=-t - 16))
        elif t >= -47:
            ops.append(_ReplacementOp(_REVERSE, arg=-t - 32))
        elif t >= -111:
            offset = -t - 48
            bank = offset // 16
            arg = offset % 16
            kind = (_PLUS1, _MINUS1, _TIMES2, _DIV2)[bank]
            ops.append(_ReplacementOp(kind, arg=arg))
        else:
            raise ValueError(f"invalid replacement opcode: {t}")
    return tuple(ops)


def _literal_matches_at_all_starts(x: torch.Tensor, literal: Tuple[int, ...]) -> torch.Tensor:
    """Boolean vector m where m[i] means literal starts exactly at i."""
    n = int(x.numel())
    L = len(literal)
    if L == 0:
        return torch.ones(n, dtype=torch.bool, device=x.device)
    if n == 0 or L > n:
        return torch.zeros(n, dtype=torch.bool, device=x.device)
    lit = torch.tensor(literal, dtype=x.dtype, device=x.device)
    # unfold is a view on CPU/CUDA and is handled by MPS without host transfer.
    windows = x.unfold(0, L, 1)
    head = (windows == lit).all(dim=1)
    if L == 1:
        return head
    tail = torch.zeros(L - 1, dtype=torch.bool, device=x.device)
    return torch.cat((head, tail), dim=0)


def _next_occurrence_table(match: torch.Tensor) -> torch.Tensor:
    """next_pos[i] = smallest j >= i with match[j], or n if absent.

    Returns n+1 entries so querying position n is valid and yields n.
    """
    n = int(match.numel())
    if n == 0:
        return torch.zeros(1, dtype=torch.long, device=match.device)
    idx = torch.arange(n, dtype=torch.long, device=match.device)
    sentinel = torch.full((n,), n, dtype=torch.long, device=match.device)
    marked = torch.where(match, idx, sentinel)
    rev = torch.flip(marked, dims=(0,))
    rev_min = torch.cummin(rev, dim=0).values
    nxt = torch.flip(rev_min, dims=(0,))
    return torch.cat((nxt, torch.tensor([n], dtype=torch.long, device=match.device)))


def _parallel_match_table(
    x: torch.Tensor, pat: _CompiledPattern
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute independent match result for every start position on the device.

    This reproduces matchOccurrences semantics: a leading literal is anchored at
    start, and every interior literal uses the shortest occurrence at/after the
    current position. A true trailing wildcard captures the whole suffix.
    """
    n = int(x.numel())
    wc = pat.wildcard_count
    starts = torch.arange(n, dtype=torch.long, device=x.device)
    pos = starts.clone()
    valid = torch.ones(n, dtype=torch.bool, device=x.device)

    cap_starts = torch.zeros((wc, n), dtype=torch.long, device=x.device)
    cap_lens = torch.zeros((wc, n), dtype=torch.long, device=x.device)

    literal_matches: List[torch.Tensor] = []
    next_tables: List[torch.Tensor] = []
    for part in pat.parts:
        m = _literal_matches_at_all_starts(x, part)
        literal_matches.append(m)
        next_tables.append(_next_occurrence_table(m) if len(part) > 0 else torch.empty(0, dtype=torch.long, device=x.device))

    first = pat.parts[0]
    if len(first) > 0:
        valid &= literal_matches[0]
        pos = pos + len(first)

    for w in range(wc):
        part = pat.parts[w + 1]
        cap_start = pos.clone()
        L = len(part)
        if L == 0:
            in_bounds = (pos >= 0) & (pos <= n)
            valid &= in_bounds
            if w == wc - 1:
                cap_len = torch.clamp(n - pos, min=0)
                pos = torch.full_like(pos, n)
            else:
                cap_len = torch.zeros_like(pos)
        else:
            in_bounds = (pos >= 0) & (pos <= n)
            query = torch.clamp(pos, min=0, max=n)
            found = next_tables[w + 1][query]
            ok = in_bounds & (found >= pos) & (found < n) & (found + L <= n)
            cap_len = torch.where(ok, found - pos, torch.zeros_like(pos))
            valid &= ok
            pos = torch.where(ok, found + L, pos)

        cap_starts[w] = torch.where(valid, cap_start, torch.zeros_like(cap_start))
        cap_lens[w] = torch.where(valid, cap_len, torch.zeros_like(cap_len))

    return valid, pos, cap_starts, cap_lens


def _greedy_select(
    valid: torch.Tensor,
    finish: torch.Tensor,
    has_leading_literal: bool,
    n: int,
) -> List[int]:
    """Exact non-overlap policy of replaceSeqCompiledInto.

    The expensive matching was parallelized; this keeps only the dependent
    interval walk on CPU. It transfers successful starts + finishes once.
    """
    success_dev = torch.nonzero(valid, as_tuple=False).flatten()
    if int(success_dev.numel()) == 0:
        return []
    success = success_dev.detach().cpu().tolist()
    success_finish = finish[success_dev].detach().cpu().tolist()

    selected: List[int] = []
    if has_leading_literal:
        pos = 0
        k = 0
        while pos < n:
            k = bisect_left(success, pos, lo=k)
            if k >= len(success):
                break
            s = int(success[k])
            f = int(success_finish[k])
            if f <= s:
                raise RuntimeError(f"non-progressing match at {s} -> {f}")
            selected.append(s)
            pos = f
            k += 1
    else:
        finish_by_start = {int(s): int(f) for s, f in zip(success, success_finish)}
        pos = 0
        while pos < n:
            f = finish_by_start.get(pos)
            if f is None:
                break
            if f <= pos:
                raise RuntimeError(f"non-progressing match at {pos} -> {f}")
            selected.append(pos)
            pos = f
    return selected


def _build_segments(
    n: int,
    selected: List[int],
    selected_finish: List[int],
    cap_starts: List[List[int]],
    cap_lens: List[List[int]],
    wc: int,
    ops: Tuple[_ReplacementOp, ...],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Return parallel arrays: kind, source_start/value, length, literal_value."""
    kinds: List[int] = []
    srcs: List[int] = []
    lens: List[int] = []
    vals: List[int] = []

    def add(kind: int, src: int, length: int, value: int = 0) -> None:
        if length <= 0:
            return
        kinds.append(kind)
        srcs.append(src)
        lens.append(length)
        vals.append(value)

    prev = 0
    for mi, (s, f) in enumerate(zip(selected, selected_finish)):
        if prev < s:
            add(_COPY, prev, s - prev)

        for op in ops:
            if op.kind == _LITERAL:
                add(_LITERAL, 0, 1, op.value)
                continue
            if wc == 0:
                # Same as appendReplacementPlan: capture-derived instructions
                # vanish when there are no captures.
                continue
            ci = op.arg if op.arg < wc else 0
            cs = int(cap_starts[mi][ci])
            cl = int(cap_lens[mi][ci])
            add(op.kind, cs, cl)
        prev = f

    if prev < n:
        add(_COPY, prev, n - prev)
    return kinds, srcs, lens, vals


def _emit_segments(
    x: torch.Tensor,
    kinds: List[int],
    srcs: List[int],
    lens: List[int],
    vals: List[int],
) -> torch.Tensor:
    """Variable-length parallel emitter using prefix-sum offsets on device."""
    device = x.device
    if not lens:
        return torch.empty(0, dtype=x.dtype, device=device)

    lengths = torch.tensor(lens, dtype=torch.long, device=device)
    offsets = torch.cumsum(lengths, dim=0) - lengths
    total = int(sum(lens))

    seg_ids = torch.repeat_interleave(
        torch.arange(len(lens), dtype=torch.long, device=device), lengths
    )
    out_pos = torch.arange(total, dtype=torch.long, device=device)
    local = out_pos - offsets[seg_ids]

    kind_t = torch.tensor(kinds, dtype=torch.int16, device=device)
    src_t = torch.tensor(srcs, dtype=torch.long, device=device)
    len_t = lengths
    val_t = torch.tensor(vals, dtype=x.dtype, device=device)

    k = kind_t[seg_ids]
    base = src_t[seg_ids]
    seg_len = len_t[seg_ids]
    src_index = base + local

    out = torch.empty(total, dtype=x.dtype, device=device)

    direct_mask = (k == _COPY) | (k == _CAPTURE)
    out[direct_mask] = x[src_index[direct_mask]]

    literal_mask = k == _LITERAL
    out[literal_mask] = val_t[seg_ids[literal_mask]]

    reverse_mask = k == _REVERSE
    ridx = base[reverse_mask] + (seg_len[reverse_mask] - 1 - local[reverse_mask])
    out[reverse_mask] = x[ridx]

    # Arithmetic transforms are all element-wise and therefore highly GPU-friendly.
    for kind, mode in ((_PLUS1, 1), (_MINUS1, 2), (_TIMES2, 3), (_DIV2, 4)):
        mask = k == kind
        v = x[src_index[mask]]
        if mode == 1:
            y = torch.where(
                (v >= 0) & (v < EMBEDDING_CODE_COUNT),
                torch.remainder(v + 1, EMBEDDING_CODE_COUNT),
                v,
            )
        elif mode == 2:
            y = torch.where(
                (v >= 0) & (v < EMBEDDING_CODE_COUNT),
                torch.remainder(v + EMBEDDING_CODE_COUNT - 1, EMBEDDING_CODE_COUNT),
                v,
            )
        elif mode == 3:
            y = torch.where(
                (v >= 0) & (v < EMBEDDING_CODE_COUNT),
                torch.remainder(v * 2, EMBEDDING_CODE_COUNT),
                v,
            )
        else:
            y = torch.where(
                v >= EMBEDDING_CODE_COUNT,
                v,
                torch.div(v, 2, rounding_mode="floor"),
            )
        out[mask] = y

    # sort($n) is rarer and fundamentally segment-local. Keep it as one device
    # sort per segment; there is no host copy of capture data.
    running = 0
    for kind, src, length in zip(kinds, srcs, lens):
        if kind == _SORT:
            out[running : running + length] = torch.sort(x[src : src + length]).values
        running += length

    return out


def _literal_matches_at_all_starts_batch(
    x: torch.Tensor, lengths: torch.Tensor, literal: Tuple[int, ...]
) -> torch.Tensor:
    """Batched literal matcher for padded [B, N] inputs."""
    B, n = x.shape
    L = len(literal)
    pos = torch.arange(n, dtype=torch.long, device=x.device).unsqueeze(0)
    if L == 0:
        return pos < lengths.unsqueeze(1)
    if n == 0 or L > n:
        return torch.zeros((B, n), dtype=torch.bool, device=x.device)
    lit = torch.tensor(literal, dtype=x.dtype, device=x.device)
    windows = x.unfold(1, L, 1)
    head = (windows == lit).all(dim=2)
    valid_start = (
        torch.arange(n - L + 1, dtype=torch.long, device=x.device).unsqueeze(0) + L
        <= lengths.unsqueeze(1)
    )
    head &= valid_start
    if L == 1:
        return head
    return torch.cat(
        (head, torch.zeros((B, L - 1), dtype=torch.bool, device=x.device)), dim=1
    )


def _next_occurrence_table_batch(match: torch.Tensor) -> torch.Tensor:
    """Row-wise next occurrence table for a [B, N] boolean matrix."""
    B, n = match.shape
    if n == 0:
        return torch.zeros((B, 1), dtype=torch.long, device=match.device)
    idx = torch.arange(n, dtype=torch.long, device=match.device).unsqueeze(0).expand(B, -1)
    sentinel = torch.full((B, n), n, dtype=torch.long, device=match.device)
    marked = torch.where(match, idx, sentinel)
    nxt = torch.flip(
        torch.cummin(torch.flip(marked, dims=(1,)), dim=1).values, dims=(1,)
    )
    return torch.cat(
        (nxt, torch.full((B, 1), n, dtype=torch.long, device=match.device)), dim=1
    )


def _parallel_match_table_same_pattern_batch(
    x: torch.Tensor, lengths: torch.Tensor, pat: _CompiledPattern
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate one compiled pattern for all rows and all start positions."""
    B, n = x.shape
    wc = pat.wildcard_count
    starts = torch.arange(n, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, -1)
    pos = starts.clone()
    valid = starts < lengths.unsqueeze(1)

    cap_starts = torch.zeros((B, wc, n), dtype=torch.long, device=x.device)
    cap_lens = torch.zeros((B, wc, n), dtype=torch.long, device=x.device)

    literal_matches: List[torch.Tensor] = []
    next_tables: List[torch.Tensor] = []
    for part in pat.parts:
        m = _literal_matches_at_all_starts_batch(x, lengths, part)
        literal_matches.append(m)
        next_tables.append(
            _next_occurrence_table_batch(m)
            if len(part) > 0
            else torch.empty((B, 0), dtype=torch.long, device=x.device)
        )

    first = pat.parts[0]
    if len(first) > 0:
        valid &= literal_matches[0]
        pos = pos + len(first)

    row = torch.arange(B, dtype=torch.long, device=x.device).unsqueeze(1).expand(B, n)
    length_col = lengths.unsqueeze(1)
    for w in range(wc):
        part = pat.parts[w + 1]
        cap_start = pos.clone()
        L = len(part)
        if L == 0:
            valid &= (pos >= 0) & (pos <= length_col)
            if w == wc - 1:
                cap_len = torch.clamp(length_col - pos, min=0)
                pos = length_col.expand_as(pos)
            else:
                cap_len = torch.zeros_like(pos)
        else:
            in_bounds = (pos >= 0) & (pos <= length_col)
            query = torch.clamp(pos, min=0, max=n)
            found = next_tables[w + 1][row, query]
            ok = (
                in_bounds
                & (found >= pos)
                & (found < length_col)
                & (found + L <= length_col)
            )
            cap_len = torch.where(ok, found - pos, torch.zeros_like(pos))
            valid &= ok
            pos = torch.where(ok, found + L, pos)

        cap_starts[:, w, :] = torch.where(valid, cap_start, torch.zeros_like(cap_start))
        cap_lens[:, w, :] = torch.where(valid, cap_len, torch.zeros_like(cap_len))

    return valid, pos, cap_starts, cap_lens


def _greedy_select_host(
    valid_row: torch.Tensor,
    finish_row: torch.Tensor,
    has_leading_literal: bool,
    n: int,
) -> List[int]:
    """Dependent non-overlap interval walk on already-batched host metadata."""
    if n <= 0:
        return []
    valid = valid_row[:n]
    finish = finish_row[:n]
    selected: List[int] = []
    if has_leading_literal:
        success = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        pos = 0
        k = 0
        while pos < n:
            k = bisect_left(success, pos, lo=k)
            if k >= len(success):
                break
            s = int(success[k])
            f = int(finish[s])
            if f <= s:
                raise RuntimeError(f"non-progressing match at {s} -> {f}")
            selected.append(s)
            pos = f
            k += 1
    else:
        pos = 0
        while pos < n and bool(valid[pos]):
            f = int(finish[pos])
            if f <= pos:
                raise RuntimeError(f"non-progressing match at {pos} -> {f}")
            selected.append(pos)
            pos = f
    return selected


def _normalize_batch_arg(value: Sequence, batch_size: int, name: str) -> List[Sequence[int]]:
    """Allow one shared int sequence or one sequence per batch row."""
    if batch_size == 0:
        return []
    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            shared = value.detach().cpu().tolist()
            return [shared] * batch_size
        if value.ndim == 2 and int(value.shape[0]) == batch_size:
            return [row.detach().cpu().tolist() for row in value]
        raise ValueError(f"{name} tensor must have shape [L] or [B,L]")
    rows = list(value)
    if not rows:
        return [[]] * batch_size
    if isinstance(rows[0], int):
        return [list(map(int, rows))] * batch_size
    if len(rows) != batch_size:
        raise ValueError(f"{name} has {len(rows)} rows for batch size {batch_size}")
    return [list(map(int, row)) for row in rows]


def replace_batch_gpu(
    inputs: Sequence[Union[Sequence[int], torch.Tensor]],
    patterns: Sequence,
    replacements: Sequence,
    *,
    device: Optional[Union[str, torch.device]] = None,
    max_output: int = DEFAULT_MAX_OUTPUT,
    token_dtype: torch.dtype = torch.int64,
    metadata_chunk_size: int = 128,
    compute_changed: bool = True,
) -> BatchReplaceResult:
    """Batch independent Replace jobs on CUDA/MPS/CPU.

    patterns/replacements may be a single sequence shared by all inputs or one
    sequence per input. Equal patterns are grouped automatically, so matching is
    evaluated as [jobs, start_positions] GPU tensors. Match metadata is copied to
    the host only once per metadata chunk, then all successful variable-length
    outputs are emitted in one packed prefix-sum pass.

    metadata_chunk_size bounds the temporary [chunk, 34, max_input_len] int32
    metadata tensor. 128 is conservative for MPS; 256-1024 can be useful on
    larger CUDA devices.
    """
    dev = _pick_device(device)
    B = len(inputs)
    if metadata_chunk_size <= 0:
        raise ValueError("metadata_chunk_size must be positive")
    pats_raw = _normalize_batch_arg(patterns, B, "patterns")
    reps_raw = _normalize_batch_arg(replacements, B, "replacements")

    xs: List[torch.Tensor] = []
    lengths: List[int] = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            t = inp.to(device=dev, dtype=token_dtype).contiguous().flatten()
        else:
            t = torch.tensor(list(map(int, inp)), dtype=token_dtype, device=dev)
        xs.append(t)
        lengths.append(int(t.numel()))

    empty = torch.empty(0, dtype=token_dtype, device=dev)
    if B == 0:
        return BatchReplaceResult([], empty, tuple(), tuple())

    compiled_pats = [_compile_pattern(p) for p in pats_raw]
    compiled_reps = [_compile_replacement(r) for r in reps_raw]

    input_offsets: List[int] = []
    total_input = 0
    for n in lengths:
        input_offsets.append(total_input)
        total_input += n
    packed_input = torch.cat(xs, dim=0) if total_input else empty

    matched = [False] * B
    applied = [False] * B
    overflowed = [False] * B
    selected_counts = [0] * B
    output_offsets = [0] * B
    output_lengths = [0] * B

    all_kinds: List[int] = []
    all_srcs: List[int] = []
    all_lens: List[int] = []
    all_vals: List[int] = []
    packed_output_cursor = 0

    for chunk0 in range(0, B, metadata_chunk_size):
        chunk1 = min(B, chunk0 + metadata_chunk_size)
        ids = list(range(chunk0, chunk1))
        active_ids = [
            j
            for j in ids
            if lengths[j] > 0
            and not compiled_pats[j].empty_pattern
            and not compiled_pats[j].all_wildcard
            and compiled_pats[j].wildcard_count <= MAX_WILDCARDS
        ]
        if not active_ids:
            continue

        max_n = max(lengths[j] for j in active_ids)
        C = len(ids)
        # rows: valid, finish, 16 cap starts, 16 cap lengths.
        meta = torch.zeros((C, 34, max_n), dtype=torch.int32, device=dev)

        groups: Dict[Tuple[Tuple[int, ...], ...], List[int]] = {}
        for j in active_ids:
            groups.setdefault(compiled_pats[j].parts, []).append(j)

        for group_ids in groups.values():
            pat = compiled_pats[group_ids[0]]
            gmax = max(lengths[j] for j in group_ids)
            G = len(group_ids)
            xpad = torch.zeros((G, gmax), dtype=token_dtype, device=dev)
            glens = torch.tensor(
                [lengths[j] for j in group_ids], dtype=torch.long, device=dev
            )
            for gi, j in enumerate(group_ids):
                n = lengths[j]
                xpad[gi, :n] = xs[j]

            v, f, cs, cl = _parallel_match_table_same_pattern_batch(xpad, glens, pat)
            rows = torch.tensor(
                [j - chunk0 for j in group_ids], dtype=torch.long, device=dev
            )
            meta[rows, 0, :gmax] = v.to(torch.int32)
            meta[rows, 1, :gmax] = f.to(torch.int32)
            wc = pat.wildcard_count
            if wc:
                meta[rows, 2 : 2 + wc, :gmax] = cs.to(torch.int32)
                meta[rows, 18 : 18 + wc, :gmax] = cl.to(torch.int32)

        # One D->H sync for the whole chunk instead of one per Replace.
        host_meta = meta.cpu()

        for j in ids:
            pat = compiled_pats[j]
            n = lengths[j]
            if (
                n == 0
                or pat.empty_pattern
                or pat.all_wildcard
                or pat.wildcard_count > MAX_WILDCARDS
            ):
                continue

            row = host_meta[j - chunk0]
            selected = _greedy_select_host(
                row[0].bool(), row[1], pat.has_leading_literal, n
            )
            if not selected:
                continue

            matched[j] = True
            selected_counts[j] = len(selected)
            selected_finish = [int(row[1, s]) for s in selected]
            wc = pat.wildcard_count
            if wc:
                cap_starts = [
                    [int(row[2 + w, s]) for w in range(wc)] for s in selected
                ]
                cap_lens = [
                    [int(row[18 + w, s]) for w in range(wc)] for s in selected
                ]
            else:
                cap_starts = [[] for _ in selected]
                cap_lens = [[] for _ in selected]

            kinds, srcs, lens, vals = _build_segments(
                n,
                selected,
                selected_finish,
                cap_starts,
                cap_lens,
                wc,
                compiled_reps[j],
            )
            out_n = sum(lens)
            if out_n > max_output:
                overflowed[j] = True
                continue

            applied[j] = True
            output_offsets[j] = packed_output_cursor
            output_lengths[j] = out_n
            packed_output_cursor += out_n

            base = input_offsets[j]
            all_kinds.extend(kinds)
            all_srcs.extend(base + src for src in srcs)
            all_lens.extend(lens)
            all_vals.extend(vals)

    packed_out = (
        _emit_segments(packed_input, all_kinds, all_srcs, all_lens, all_vals)
        if all_lens
        else empty
    )

    changed = [False] * B
    if compute_changed and any(applied):
        changed_dev = torch.tensor(
            [applied[j] and output_lengths[j] != lengths[j] for j in range(B)],
            dtype=torch.bool,
            device=dev,
        )
        equal_len_jobs = [
            j
            for j in range(B)
            if applied[j] and output_lengths[j] == lengths[j] and lengths[j] > 0
        ]
        if equal_len_jobs:
            eq_lens_list = [lengths[j] for j in equal_len_jobs]
            eq_lens = torch.tensor(eq_lens_list, dtype=torch.long, device=dev)
            eq_jobs_t = torch.tensor(equal_len_jobs, dtype=torch.long, device=dev)
            job_ids = torch.repeat_interleave(eq_jobs_t, eq_lens)

            host_group_starts: List[int] = []
            c = 0
            for ln in eq_lens_list:
                host_group_starts.append(c)
                c += ln
            local = torch.arange(c, dtype=torch.long, device=dev) - torch.repeat_interleave(
                torch.tensor(host_group_starts, dtype=torch.long, device=dev), eq_lens
            )
            out_idx = torch.repeat_interleave(
                torch.tensor(
                    [output_offsets[j] for j in equal_len_jobs],
                    dtype=torch.long,
                    device=dev,
                ),
                eq_lens,
            ) + local
            in_idx = torch.repeat_interleave(
                torch.tensor(
                    [input_offsets[j] for j in equal_len_jobs],
                    dtype=torch.long,
                    device=dev,
                ),
                eq_lens,
            ) + local
            diff = (packed_out[out_idx] != packed_input[in_idx]).to(torch.int32)
            counts = torch.zeros(B, dtype=torch.int32, device=dev)
            counts.scatter_add_(0, job_ids, diff)
            changed_dev |= counts > 0
        # One final sync only if exact changed flags were requested.
        changed = changed_dev.cpu().tolist()

    results: List[ReplaceResult] = []
    for j in range(B):
        if applied[j]:
            off = output_offsets[j]
            ln = output_lengths[j]
            data = packed_out[off : off + ln]
        else:
            data = empty
        results.append(
            ReplaceResult(
                matched[j],
                applied[j],
                bool(changed[j]) if compute_changed else False,
                overflowed[j],
                data,
                selected_counts[j],
            )
        )

    return BatchReplaceResult(
        results,
        packed_out,
        tuple(output_offsets),
        tuple(output_lengths),
    )


def replace_batch_gpu_lists(
    inputs: Sequence[Sequence[int]],
    patterns: Sequence,
    replacements: Sequence,
    **kwargs,
) -> Tuple[List[bool], List[List[int]]]:
    """Convenience wrapper; copies the whole packed output to CPU only once."""
    r = replace_batch_gpu(inputs, patterns, replacements, **kwargs)
    return [x.applied for x in r.results], r.to_lists()

def replace_gpu(
    input_tokens: Union[Sequence[int], torch.Tensor],
    pattern: Sequence[int],
    replacement: Sequence[int],
    *,
    device: Optional[Union[str, torch.device]] = None,
    max_output: int = DEFAULT_MAX_OUTPUT,
    token_dtype: torch.dtype = torch.int64,
) -> ReplaceResult:
    """GPU-oriented equivalent of replaceSeqCompiled/replaceSeq.

    Pattern semantics:
      * only -1 in `pattern` is a wildcard, numbered by occurrence.
      * interior wildcards use the shortest following literal occurrence.
      * a trailing wildcard captures the entire remaining suffix.
      * all-wildcard / empty / >16-wildcard patterns do not fire.

    Replacement opcodes (same as the Nim code):
      >= 0       literal token
      -1..-15    $1..$15
      -16..-31   sort($1)..sort($16)
      -32..-47   reverse($1)..reverse($16)
      -48..-63   +1($1)..+1($16) modulo 512 for internal tokens
      -64..-79   -1(...)
      -80..-95   *2(...)
      -96..-111  floor_div_2(...)

    Returns an empty tensor on no match or overflow, mirroring replaceSeqCompiled.

    `token_dtype=torch.int64` preserves the broad integer behavior of Nim on
    CPU/CUDA. If a particular MPS/PyTorch build has limited int64 kernel support,
    use torch.int32; all normal model tokens (0..512) are exactly representable.
    """
    dev = _pick_device(device)
    if isinstance(input_tokens, torch.Tensor):
        x = input_tokens.to(device=dev, dtype=token_dtype).contiguous()
    else:
        x = torch.tensor(list(map(int, input_tokens)), dtype=token_dtype, device=dev)

    empty = torch.empty(0, dtype=x.dtype, device=dev)
    pat = _compile_pattern(pattern)
    if pat.empty_pattern or pat.all_wildcard or pat.wildcard_count > MAX_WILDCARDS:
        return ReplaceResult(False, False, False, False, empty, 0)
    if int(x.numel()) == 0:
        return ReplaceResult(False, False, False, False, empty, 0)

    ops = _compile_replacement(replacement)
    valid, finish, cap_start_all, cap_len_all = _parallel_match_table(x, pat)
    n = int(x.numel())
    selected = _greedy_select(valid, finish, pat.has_leading_literal, n)
    if not selected:
        return ReplaceResult(False, False, False, False, empty, 0)

    sel_dev = torch.tensor(selected, dtype=torch.long, device=dev)
    selected_finish = finish[sel_dev].detach().cpu().tolist()

    wc = pat.wildcard_count
    if wc > 0:
        # One compact transfer after greedy selection; shape <= matches x 16 x 2.
        cs = cap_start_all[:, sel_dev].T.detach().cpu().tolist()
        cl = cap_len_all[:, sel_dev].T.detach().cpu().tolist()
    else:
        cs = [[] for _ in selected]
        cl = [[] for _ in selected]

    kinds, srcs, lens, vals = _build_segments(
        n, selected, selected_finish, cs, cl, wc, ops
    )
    total = sum(lens)
    if total > max_output:
        return ReplaceResult(True, False, False, True, empty, len(selected))

    out = _emit_segments(x, kinds, srcs, lens, vals)
    changed = int(out.numel()) != n or not torch.equal(out, x)
    return ReplaceResult(True, True, bool(changed), False, out, len(selected))


def replace_gpu_list(
    input_tokens: Sequence[int],
    pattern: Sequence[int],
    replacement: Sequence[int],
    **kwargs,
) -> Tuple[bool, List[int]]:
    """Convenience wrapper with the old `(changed, data)`-like shape.

    The bool means a valid replacement was applied. `data` is [] on no match or
    overflow. This wrapper necessarily copies the final tensor back to CPU.
    """
    r = replace_gpu(input_tokens, pattern, replacement, **kwargs)
    if not r.applied:
        return False, []
    return True, r.data.detach().cpu().tolist()


if __name__ == "__main__":
    # Small compatibility smoke tests copied from the Nim Replace tests.
    def chk(inp, pat, rep, expected):
        ok, got = replace_gpu_list(inp, pat, rep, device="cpu")
        assert ok and got == expected, (inp, pat, rep, got, expected)

    chk([97, 120, 120, 98], [97, -1, 98], [-1], [120, 120])
    chk([122, 122, 97, 113], [-1, 97], [-1], [122, 122, 113])
    chk([97, 49, 98, 50, 99], [97, -1, 98, -1, 99], [-2, -1], [50, 49])
    chk([97, 49, 50], [97, -1], [-1], [49, 50])
    chk([49, 50, 97], [-1, -1, 97], [-1, 99, -2], [99, 49, 50])
    chk([97, 49, 98, 97, 50, 98], [97, -1, 98], [-1], [49, 50])

    arithmetic_input = [97, 2, 4, 8, 98]
    arithmetic_pattern = [97, -1, 98]
    chk(arithmetic_input, arithmetic_pattern, [-48], [3, 5, 9])
    chk(arithmetic_input, arithmetic_pattern, [-64], [1, 3, 7])
    chk(arithmetic_input, arithmetic_pattern, [-80], [4, 8, 16])
    chk(arithmetic_input, arithmetic_pattern, [-96], [1, 2, 4])
    chk(
        arithmetic_input,
        arithmetic_pattern,
        [-16, -32, -48, -64, -80, -96],
        [2, 4, 8, 8, 4, 2, 3, 5, 9, 1, 3, 7, 4, 8, 16, 1, 2, 4],
    )
    chk([97, -3, -2, -1, 98], arithmetic_pattern, [-96], [-2, -1, -1])
    chk([97, 97, 97], [97, 97], [98], [98, 97])

    # Exact output boundary / overflow behavior.
    edge = [0] + [1] * 1499
    ok, got = replace_gpu_list(edge, [0, -1], [-1, 0], device="cpu", max_output=1500)
    assert ok and len(got) == 1500 and got[0] == 1 and got[-1] == 0
    r = replace_gpu(edge, [0, -1], [-1, 0, 0], device="cpu", max_output=1500)
    assert r.matched and r.overflowed and not r.applied and r.data.numel() == 0

    # Batch API: shared rule over heterogeneous inputs.
    br = replace_batch_gpu(
        [[97, 120, 120, 98], [97, 49, 98, 97, 50, 98], [0, 0, 0]],
        [97, -1, 98],
        [-1],
        device="cpu",
        metadata_chunk_size=2,
    )
    assert br.to_lists() == [[120, 120], [49, 50], []]
    assert [r.applied for r in br.results] == [True, True, False]

    # Per-job rules, arithmetic, and overflow.
    br2 = replace_batch_gpu(
        [arithmetic_input, [97, 97, 97], edge],
        [arithmetic_pattern, [97, 97], [0, -1]],
        [[-48], [98], [-1, 0, 0]],
        device="cpu",
        max_output=1500,
    )
    br2_lists = br2.to_lists()
    assert br2_lists[0] == [3, 5, 9]
    assert br2_lists[1] == [98, 97]
    assert br2.results[2].matched and br2.results[2].overflowed and not br2.results[2].applied

    # Scalar/batch equivalence across representative semantics.
    corpus = [
        ([97, 120, 120, 98], [97, -1, 98], [-1]),
        ([122, 122, 97, 113], [-1, 97], [-1]),
        ([97, 49, 98, 50, 99], [97, -1, 98, -1, 99], [-2, -1]),
        ([97, 49, 50], [97, -1], [-1]),
        ([49, 50, 97], [-1, -1, 97], [-1, 99, -2]),
        (arithmetic_input, arithmetic_pattern, [-16, -32, -48, -64, -80, -96]),
    ]
    bb = replace_batch_gpu(
        [x[0] for x in corpus],
        [x[1] for x in corpus],
        [x[2] for x in corpus],
        device="cpu",
    )
    bb_lists = bb.to_lists()
    for i, (inp, pat0, rep0) in enumerate(corpus):
        scalar = replace_gpu(inp, pat0, rep0, device="cpu")
        assert bb.results[i].matched == scalar.matched
        assert bb.results[i].applied == scalar.applied
        assert bb.results[i].overflowed == scalar.overflowed
        assert bb_lists[i] == (scalar.data.cpu().tolist() if scalar.applied else [])

    print("PASS: scalar + batched GPU Replace compatibility smoke tests")

# ============================================================================
# V2 fast batch path
# - MPS shared-pattern/shared-replacement batches are fused into one Metal kernel.
# - No per-start metadata is copied back to the CPU.
# - match -> greedy non-overlap selection -> replacement emit happens in-kernel.
# - A second tiny Metal kernel packs per-job scratch rows into packed_data.
#
# The original PyTorch implementation above remains as a correctness fallback
# for CPU/CUDA, per-job rules, sort($capture), older PyTorch, and unusual dtypes.
# ============================================================================

_replace_batch_gpu_reference = replace_batch_gpu
_MPS_FUSED_LIB = None

_MPS_FUSED_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

inline bool write_token(
    device int* scratch,
    int scratch_base,
    int stride,
    thread int& out_pos,
    int value,
    const device int* input,
    int input_base,
    int input_len,
    thread bool& changed)
{
    if (out_pos >= stride) {
        return false;
    }
    scratch[scratch_base + out_pos] = value;
    if (out_pos >= input_len || input[input_base + out_pos] != value) {
        changed = true;
    }
    out_pos += 1;
    return true;
}

inline bool match_at(
    const device int* input,
    int input_base,
    int n,
    const device int* pattern,
    int pattern_len,
    int start,
    thread int* cap_start,
    thread int* cap_len,
    thread int& finish)
{
    int pos = start;
    int p = 0;
    int wc = 0;

    // Leading literal run is anchored at `start`.
    while (p < pattern_len && pattern[p] >= 0) {
        if (pos >= n || input[input_base + pos] != pattern[p]) {
            return false;
        }
        pos += 1;
        p += 1;
    }

    while (p < pattern_len) {
        // Every negative pattern token is a wildcard.
        if (pattern[p] >= 0 || wc >= 16) {
            return false;
        }
        p += 1;
        int ci = wc++;
        int cs = pos;

        int lit_begin = p;
        while (p < pattern_len && pattern[p] >= 0) {
            p += 1;
        }
        int lit_len = p - lit_begin;

        cap_start[ci] = cs;
        if (lit_len == 0) {
            if (p >= pattern_len) {
                // True trailing wildcard captures the suffix.
                cap_len[ci] = n - pos;
                pos = n;
            } else {
                // Consecutive interior wildcard: shortest capture is empty.
                cap_len[ci] = 0;
            }
            continue;
        }

        int found = -1;
        int last = n - lit_len;
        for (int cand = pos; cand <= last; ++cand) {
            bool ok = true;
            for (int k = 0; k < lit_len; ++k) {
                if (input[input_base + cand + k] != pattern[lit_begin + k]) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                found = cand;
                break;
            }
        }
        if (found < 0) {
            return false;
        }
        cap_len[ci] = found - pos;
        pos = found + lit_len;
    }

    finish = pos;
    return finish > start;
}

kernel void replace_fused_i32(
    const device int* input [[buffer(0)]],
    const device int* lengths [[buffer(1)]],
    const device int* pattern [[buffer(2)]],
    const device int* replacement [[buffer(3)]],
    device int* scratch [[buffer(4)]],
    device int* out_lengths [[buffer(5)]],
    device int* stats [[buffer(6)]],
    constant int& max_n [[buffer(7)]],
    constant int& pattern_len [[buffer(8)]],
    constant int& replacement_len [[buffer(9)]],
    constant int& wildcard_count [[buffer(10)]],
    constant int& has_leading_literal [[buffer(11)]],
    constant int& stride [[buffer(12)]],
    uint job_u [[thread_position_in_grid]])
{
    int job = int(job_u);
    int n = lengths[job];
    int input_base = job * max_n;
    int scratch_base = job * stride;

    int stat_base = job * 5;
    stats[stat_base + 0] = 0; // matched
    stats[stat_base + 1] = 0; // applied
    stats[stat_base + 2] = 0; // changed
    stats[stat_base + 3] = 0; // overflow
    stats[stat_base + 4] = 0; // selected matches
    out_lengths[job] = 0;

    if (n <= 0 || pattern_len <= 0 || stride <= 0) {
        return;
    }

    int out_pos = 0;
    int prev = 0;
    int scan_pos = 0;
    int selected = 0;
    bool changed = false;
    bool matched = false;

    int cap_start[16];
    int cap_len[16];

    while (scan_pos < n) {
        int s = -1;
        int finish = -1;

        if (has_leading_literal != 0) {
            // Find the first successful start >= scan_pos.
            int first_literal = pattern[0];
            for (int cand = scan_pos; cand < n; ++cand) {
                if (input[input_base + cand] != first_literal) {
                    continue;
                }
                int f = -1;
                if (match_at(input, input_base, n, pattern, pattern_len,
                             cand, cap_start, cap_len, f)) {
                    s = cand;
                    finish = f;
                    break;
                }
            }
            if (s < 0) {
                break;
            }
        } else {
            // Exact current-position policy used by the original implementation
            // when the pattern starts with a wildcard.
            int f = -1;
            if (!match_at(input, input_base, n, pattern, pattern_len,
                          scan_pos, cap_start, cap_len, f)) {
                break;
            }
            s = scan_pos;
            finish = f;
        }

        matched = true;
        selected += 1;

        // Copy untouched prefix between selected matches.
        for (int i = prev; i < s; ++i) {
            if (!write_token(scratch, scratch_base, stride, out_pos,
                             input[input_base + i], input, input_base, n, changed)) {
                stats[stat_base + 0] = 1;
                stats[stat_base + 3] = 1;
                stats[stat_base + 4] = selected;
                return;
            }
        }

        // Emit replacement program.
        for (int r = 0; r < replacement_len; ++r) {
            int t = replacement[r];
            if (t >= 0) {
                if (!write_token(scratch, scratch_base, stride, out_pos,
                                 t, input, input_base, n, changed)) {
                    stats[stat_base + 0] = 1;
                    stats[stat_base + 3] = 1;
                    stats[stat_base + 4] = selected;
                    return;
                }
                continue;
            }

            // With zero captures, capture-derived opcodes vanish.
            if (wildcard_count <= 0) {
                continue;
            }

            int kind = 0;
            int ci = 0;
            if (t >= -15) {
                kind = 1; // capture
                ci = -t - 1;
            } else if (t >= -31) {
                // sort($n) is intentionally excluded from the fused path.
                // Host dispatch falls back before this kernel is called.
                return;
            } else if (t >= -47) {
                kind = 2; // reverse
                ci = -t - 32;
            } else if (t >= -111) {
                int off = -t - 48;
                int bank = off / 16;
                ci = off - bank * 16;
                kind = 3 + bank; // +1, -1, *2, //2
            } else {
                return;
            }

            if (ci >= wildcard_count) {
                ci = 0;
            }
            int cs = cap_start[ci];
            int cl = cap_len[ci];

            for (int q = 0; q < cl; ++q) {
                int src_local = (kind == 2) ? (cs + cl - 1 - q) : (cs + q);
                int v = input[input_base + src_local];
                int y = v;
                if (kind == 3) {
                    if (v >= 0 && v < 512) y = (v + 1) & 511;
                } else if (kind == 4) {
                    if (v >= 0 && v < 512) y = (v + 511) & 511;
                } else if (kind == 5) {
                    if (v >= 0 && v < 512) y = (v * 2) & 511;
                } else if (kind == 6) {
                    if (v < 512) {
                        // floor(v / 2), including negative integers.
                        y = (v >= 0) ? (v / 2) : -(((-v) + 1) / 2);
                    }
                }

                if (!write_token(scratch, scratch_base, stride, out_pos,
                                 y, input, input_base, n, changed)) {
                    stats[stat_base + 0] = 1;
                    stats[stat_base + 3] = 1;
                    stats[stat_base + 4] = selected;
                    return;
                }
            }
        }

        prev = finish;
        scan_pos = finish;
    }

    if (!matched) {
        return;
    }

    // Copy untouched suffix.
    for (int i = prev; i < n; ++i) {
        if (!write_token(scratch, scratch_base, stride, out_pos,
                         input[input_base + i], input, input_base, n, changed)) {
            stats[stat_base + 0] = 1;
            stats[stat_base + 3] = 1;
            stats[stat_base + 4] = selected;
            return;
        }
    }

    if (out_pos != n) {
        changed = true;
    }
    stats[stat_base + 0] = 1;
    stats[stat_base + 1] = 1;
    stats[stat_base + 2] = changed ? 1 : 0;
    stats[stat_base + 3] = 0;
    stats[stat_base + 4] = selected;
    out_lengths[job] = out_pos;
}

kernel void pack_rows_i32(
    const device int* scratch [[buffer(0)]],
    const device int* out_lengths [[buffer(1)]],
    const device int* offsets [[buffer(2)]],
    device int* packed [[buffer(3)]],
    constant int& stride [[buffer(4)]],
    uint job_u [[thread_position_in_grid]])
{
    int job = int(job_u);
    int n = out_lengths[job];
    int src = job * stride;
    int dst = offsets[job];
    for (int i = 0; i < n; ++i) {
        packed[dst + i] = scratch[src + i];
    }
}
'''


def _get_mps_fused_lib():
    global _MPS_FUSED_LIB
    if _MPS_FUSED_LIB is None:
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("torch.mps.compile_shader is unavailable")
        _MPS_FUSED_LIB = torch.mps.compile_shader(_MPS_FUSED_SOURCE)
    return _MPS_FUSED_LIB


def _shared_int_sequence(value) -> Optional[List[int]]:
    """Return a shared 1-D integer sequence, or None for per-job rows."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            return None
        return [int(x) for x in value.detach().cpu().tolist()]
    rows = list(value)
    if not rows:
        return []
    if isinstance(rows[0], int):
        return [int(x) for x in rows]
    return None


def _mps_fused_supported(patterns, replacements, token_dtype) -> bool:
    if token_dtype != torch.int32:
        return False
    pat = _shared_int_sequence(patterns)
    rep = _shared_int_sequence(replacements)
    if pat is None or rep is None:
        return False
    cp = _compile_pattern(pat)
    if cp.empty_pattern or cp.wildcard_count > MAX_WILDCARDS:
        return False
    try:
        _compile_replacement(rep)
    except Exception:
        return False
    # sort($capture) needs a different cooperative kernel; use reference path.
    if any(-31 <= int(t) <= -16 for t in rep):
        return False
    return True


def _mps_fused_scratch_stride(max_n: int, pattern: Sequence[int], replacement: Sequence[int], max_output: int) -> int:
    """Safe per-job output upper bound, capped by max_output.

    Across non-overlapping matches, the total length contributed by any one
    capture-derived replacement op is <= input length. Literal replacement ops
    can occur at most one per selected match, and each match consumes at least
    the count of literal pattern tokens.
    """
    if max_n <= 0:
        return 1
    literal_pattern = max(1, sum(1 for x in pattern if int(x) >= 0))
    literal_rep = sum(1 for x in replacement if int(x) >= 0)
    cap_rep = sum(1 for x in replacement if int(x) < 0)
    max_matches = (max_n // literal_pattern) + 1
    bound = max_n + cap_rep * max_n + literal_rep * max_matches
    return max(1, min(int(max_output), int(bound)))


def _replace_batch_mps_fused(
    inputs,
    pattern: Sequence[int],
    replacement: Sequence[int],
    *,
    max_output: int,
    compute_changed: bool,
) -> BatchReplaceResult:
    B = len(inputs)
    dev = torch.device("mps")
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    if B == 0:
        return BatchReplaceResult([], empty, tuple(), tuple())

    # Build one padded CPU tensor, then perform one H->MPS transfer.
    lengths_host: List[int] = []
    host_rows: List[torch.Tensor] = []
    max_n = 0
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            row = inp.detach().to(device="cpu", dtype=torch.int32).contiguous().flatten()
        else:
            row = torch.tensor(list(map(int, inp)), dtype=torch.int32)
        host_rows.append(row)
        n = int(row.numel())
        lengths_host.append(n)
        max_n = max(max_n, n)

    if max_n == 0:
        results = [ReplaceResult(False, False, False, False, empty, 0) for _ in range(B)]
        return BatchReplaceResult(results, empty, tuple([0] * B), tuple([0] * B))

    host_input = torch.zeros((B, max_n), dtype=torch.int32)
    for j, row in enumerate(host_rows):
        if row.numel():
            host_input[j, : row.numel()] = row

    x = host_input.to(dev)
    lengths = torch.tensor(lengths_host, dtype=torch.int32, device=dev)
    pat_t = torch.tensor(list(map(int, pattern)), dtype=torch.int32, device=dev)
    rep_t = torch.tensor(list(map(int, replacement)), dtype=torch.int32, device=dev)

    cp = _compile_pattern(pattern)
    stride = _mps_fused_scratch_stride(max_n, pattern, replacement, max_output)

    # Avoid pathological scratch allocations; fall back rather than OOM.
    scratch_bytes = B * stride * 4
    if scratch_bytes > 512 * 1024 * 1024:
        raise MemoryError(f"fused MPS scratch would require {scratch_bytes / 2**20:.1f} MiB")

    scratch = torch.empty((B, stride), dtype=torch.int32, device=dev)
    out_lengths = torch.zeros(B, dtype=torch.int32, device=dev)
    stats = torch.zeros((B, 5), dtype=torch.int32, device=dev)

    lib = _get_mps_fused_lib()
    group = min(64, max(1, B))
    lib.replace_fused_i32(
        x,
        lengths,
        pat_t,
        rep_t,
        scratch,
        out_lengths,
        stats,
        int(max_n),
        int(len(pattern)),
        int(len(replacement)),
        int(cp.wildcard_count),
        int(cp.has_leading_literal),
        int(stride),
        threads=[B, 1, 1],
        group_size=[group, 1, 1],
    )

    # Prefix sum stays on MPS; only a scalar total is synchronized to size packed_data.
    prefix = torch.cumsum(out_lengths, dim=0, dtype=torch.int32)
    offsets = prefix - out_lengths
    total = int(prefix[-1].item()) if B else 0
    packed = torch.empty(total, dtype=torch.int32, device=dev)
    if total > 0:
        lib.pack_rows_i32(
            scratch,
            out_lengths,
            offsets,
            packed,
            int(stride),
            threads=[B, 1, 1],
            group_size=[group, 1, 1],
        )

    # Tiny result metadata only: O(B), not O(B * N * 34).  Concatenate it
    # first so there is only one final device->host copy/synchronization.
    meta_host = torch.cat(
        (stats, out_lengths[:, None], offsets[:, None]), dim=1
    ).cpu().tolist()

    results: List[ReplaceResult] = []
    lengths_out: List[int] = []
    offsets_host: List[int] = []
    for j in range(B):
        row = meta_host[j]
        matched = bool(row[0])
        applied = bool(row[1])
        changed = bool(row[2]) if compute_changed else False
        overflow = bool(row[3])
        selected = int(row[4])
        ln = int(row[5])
        off = int(row[6])
        lengths_out.append(ln)
        offsets_host.append(off)
        data = packed[off:off + ln] if applied else empty
        results.append(ReplaceResult(matched, applied, changed, overflow, data, selected))

    return BatchReplaceResult(
        results,
        packed,
        tuple(int(x) for x in offsets_host),
        tuple(int(x) for x in lengths_out),
    )


def replace_batch_gpu(
    inputs: Sequence[Union[Sequence[int], torch.Tensor]],
    patterns: Sequence,
    replacements: Sequence,
    *,
    device: Optional[Union[str, torch.device]] = None,
    max_output: int = DEFAULT_MAX_OUTPUT,
    token_dtype: torch.dtype = torch.int64,
    metadata_chunk_size: int = 128,
    compute_changed: bool = True,
) -> BatchReplaceResult:
    """V2 dispatcher.

    On MPS + int32 + shared rule, uses a fused Metal kernel that performs the
    whole match/greedy/emit pipeline on-device. Otherwise it falls back to the
    original batched PyTorch implementation above.
    """
    dev = _pick_device(device)
    if dev.type == "mps" and _mps_fused_supported(patterns, replacements, token_dtype):
        pat = _shared_int_sequence(patterns)
        rep = _shared_int_sequence(replacements)
        assert pat is not None and rep is not None
        try:
            return _replace_batch_mps_fused(
                inputs, pat, rep, max_output=max_output, compute_changed=compute_changed
            )
        except (RuntimeError, MemoryError) as e:
            # Keep compatibility with older PyTorch/MPS setups. Users can still
            # benchmark the reference path rather than losing functionality.
            # Set GPU_REPLACE_STRICT_MPS=1 if they want the error surfaced.
            import os
            if os.environ.get("GPU_REPLACE_STRICT_MPS") == "1":
                raise

    return _replace_batch_gpu_reference(
        inputs,
        patterns,
        replacements,
        device=dev,
        max_output=max_output,
        token_dtype=token_dtype,
        metadata_chunk_size=metadata_chunk_size,
        compute_changed=compute_changed,
    )

# Compact-metadata generic backend (CPU/CUDA and MPS fallback).
def _replace_batch_gpu_compact(
    inputs: Sequence[Union[Sequence[int], torch.Tensor]],
    patterns: Sequence,
    replacements: Sequence,
    *,
    device: Optional[Union[str, torch.device]] = None,
    max_output: int = DEFAULT_MAX_OUTPUT,
    token_dtype: torch.dtype = torch.int64,
    metadata_chunk_size: int = 128,
    compute_changed: bool = True,
) -> BatchReplaceResult:
    dev = _pick_device(device)
    B = len(inputs)
    if metadata_chunk_size <= 0:
        raise ValueError("metadata_chunk_size must be positive")
    pats_raw = _normalize_batch_arg(patterns, B, "patterns")
    reps_raw = _normalize_batch_arg(replacements, B, "replacements")

    xs: List[torch.Tensor] = []
    lengths: List[int] = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            t = inp.to(device=dev, dtype=token_dtype).contiguous().flatten()
        else:
            t = torch.tensor(list(map(int, inp)), dtype=token_dtype, device=dev)
        xs.append(t)
        lengths.append(int(t.numel()))

    empty = torch.empty(0, dtype=token_dtype, device=dev)
    if B == 0:
        return BatchReplaceResult([], empty, tuple(), tuple())

    compiled_pats = [_compile_pattern(p) for p in pats_raw]
    compiled_reps = [_compile_replacement(r) for r in reps_raw]

    input_offsets: List[int] = []
    total_input = 0
    for n in lengths:
        input_offsets.append(total_input)
        total_input += n
    packed_input = torch.cat(xs, dim=0) if total_input else empty

    matched = [False] * B
    applied = [False] * B
    overflowed = [False] * B
    selected_counts = [0] * B
    output_offsets = [0] * B
    output_lengths = [0] * B

    all_kinds: List[int] = []
    all_srcs: List[int] = []
    all_lens: List[int] = []
    all_vals: List[int] = []
    packed_output_cursor = 0

    groups: Dict[Tuple[Tuple[int, ...], ...], List[int]] = {}
    for j, pat in enumerate(compiled_pats):
        if (
            lengths[j] > 0
            and not pat.empty_pattern
            and not pat.all_wildcard
            and pat.wildcard_count <= MAX_WILDCARDS
        ):
            groups.setdefault(pat.parts, []).append(j)

    for group_ids_all in groups.values():
        pat = compiled_pats[group_ids_all[0]]
        wc = pat.wildcard_count
        for g0 in range(0, len(group_ids_all), metadata_chunk_size):
            group_ids = group_ids_all[g0 : g0 + metadata_chunk_size]
            G = len(group_ids)
            gmax = max(lengths[j] for j in group_ids)

            xpad = torch.zeros((G, gmax), dtype=token_dtype, device=dev)
            glens = torch.tensor([lengths[j] for j in group_ids], dtype=torch.long, device=dev)
            for gi, j in enumerate(group_ids):
                n = lengths[j]
                xpad[gi, :n] = xs[j]

            v, f, cs, cl = _parallel_match_table_same_pattern_batch(xpad, glens, pat)

            # Compact D->H transfer: only selection metadata, not all capture planes.
            sel_meta = torch.stack((v.to(torch.int32), f.to(torch.int32)), dim=1).cpu()

            selected_rows = []  # (j, gi, selected, selected_finish)
            gather_gi: List[int] = []
            gather_start: List[int] = []
            gather_ranges: List[Tuple[int, int]] = []

            for gi, j in enumerate(group_ids):
                n = lengths[j]
                row = sel_meta[gi]
                selected = _greedy_select_host(row[0].bool(), row[1], pat.has_leading_literal, n)
                if not selected:
                    continue
                matched[j] = True
                selected_counts[j] = len(selected)
                selected_finish = [int(row[1, s]) for s in selected]
                begin = len(gather_gi)
                if wc:
                    gather_gi.extend([gi] * len(selected))
                    gather_start.extend(selected)
                end = len(gather_gi)
                gather_ranges.append((begin, end))
                selected_rows.append((j, gi, selected, selected_finish))

            cap_start_host = None
            cap_len_host = None
            if wc and gather_gi:
                gi_t = torch.tensor(gather_gi, dtype=torch.long, device=dev)
                st_t = torch.tensor(gather_start, dtype=torch.long, device=dev)
                # [M, wc] -- only captures belonging to selected matches cross the bus.
                compact_caps = torch.cat(
                    (cs[gi_t, :, st_t].to(torch.int32), cl[gi_t, :, st_t].to(torch.int32)),
                    dim=1,
                ).cpu()
                cap_start_host = compact_caps[:, :wc]
                cap_len_host = compact_caps[:, wc:]

            for ri, (j, gi, selected, selected_finish) in enumerate(selected_rows):
                if wc:
                    begin, end = gather_ranges[ri]
                    assert cap_start_host is not None and cap_len_host is not None
                    cap_starts = cap_start_host[begin:end].tolist()
                    cap_lens = cap_len_host[begin:end].tolist()
                else:
                    cap_starts = [[] for _ in selected]
                    cap_lens = [[] for _ in selected]

                kinds, srcs, lens, vals = _build_segments(
                    lengths[j],
                    selected,
                    selected_finish,
                    cap_starts,
                    cap_lens,
                    wc,
                    compiled_reps[j],
                )
                out_n = sum(lens)
                if out_n > max_output:
                    overflowed[j] = True
                    continue

                applied[j] = True
                output_offsets[j] = packed_output_cursor
                output_lengths[j] = out_n
                packed_output_cursor += out_n

                base = input_offsets[j]
                all_kinds.extend(kinds)
                all_srcs.extend(base + src for src in srcs)
                all_lens.extend(lens)
                all_vals.extend(vals)

    packed_out = (
        _emit_segments(packed_input, all_kinds, all_srcs, all_lens, all_vals)
        if all_lens
        else empty
    )

    changed = [False] * B
    if compute_changed and any(applied):
        changed_dev = torch.tensor(
            [applied[j] and output_lengths[j] != lengths[j] for j in range(B)],
            dtype=torch.bool,
            device=dev,
        )
        equal_len_jobs = [
            j for j in range(B)
            if applied[j] and output_lengths[j] == lengths[j] and lengths[j] > 0
        ]
        if equal_len_jobs:
            eq_lens_list = [lengths[j] for j in equal_len_jobs]
            eq_lens = torch.tensor(eq_lens_list, dtype=torch.long, device=dev)
            eq_jobs_t = torch.tensor(equal_len_jobs, dtype=torch.long, device=dev)
            job_ids = torch.repeat_interleave(eq_jobs_t, eq_lens)

            starts: List[int] = []
            c = 0
            for ln in eq_lens_list:
                starts.append(c)
                c += ln
            local = torch.arange(c, dtype=torch.long, device=dev) - torch.repeat_interleave(
                torch.tensor(starts, dtype=torch.long, device=dev), eq_lens
            )
            out_idx = torch.repeat_interleave(
                torch.tensor([output_offsets[j] for j in equal_len_jobs], dtype=torch.long, device=dev),
                eq_lens,
            ) + local
            in_idx = torch.repeat_interleave(
                torch.tensor([input_offsets[j] for j in equal_len_jobs], dtype=torch.long, device=dev),
                eq_lens,
            ) + local
            diff = (packed_out[out_idx] != packed_input[in_idx]).to(torch.int32)
            counts = torch.zeros(B, dtype=torch.int32, device=dev)
            counts.scatter_add_(0, job_ids, diff)
            changed_dev |= counts > 0
        changed = changed_dev.cpu().tolist()

    results: List[ReplaceResult] = []
    for j in range(B):
        if applied[j]:
            off = output_offsets[j]
            ln = output_lengths[j]
            data = packed_out[off : off + ln]
        else:
            data = empty
        results.append(
            ReplaceResult(
                matched[j],
                applied[j],
                bool(changed[j]) if compute_changed else False,
                overflowed[j],
                data,
                selected_counts[j],
            )
        )

    return BatchReplaceResult(
        results,
        packed_out,
        tuple(output_offsets),
        tuple(output_lengths),
    )


# Final public dispatcher: fused MPS when possible, compact generic otherwise.
def replace_batch_gpu(
    inputs: Sequence[Union[Sequence[int], torch.Tensor]],
    patterns: Sequence,
    replacements: Sequence,
    *,
    device: Optional[Union[str, torch.device]] = None,
    max_output: int = DEFAULT_MAX_OUTPUT,
    token_dtype: torch.dtype = torch.int64,
    metadata_chunk_size: int = 128,
    compute_changed: bool = True,
) -> BatchReplaceResult:
    dev = _pick_device(device)
    if dev.type == "mps" and _mps_fused_supported(patterns, replacements, token_dtype):
        pat = _shared_int_sequence(patterns)
        rep = _shared_int_sequence(replacements)
        assert pat is not None and rep is not None
        try:
            return _replace_batch_mps_fused(
                inputs, pat, rep, max_output=max_output, compute_changed=compute_changed
            )
        except (RuntimeError, MemoryError):
            import os
            if os.environ.get("GPU_REPLACE_STRICT_MPS") == "1":
                raise

    return _replace_batch_gpu_compact(
        inputs,
        patterns,
        replacements,
        device=dev,
        max_output=max_output,
        token_dtype=token_dtype,
        metadata_chunk_size=metadata_chunk_size,
        compute_changed=compute_changed,
    )

# ============================================================================
# Persistent MPS state (V2-based)
#
# This keeps a padded batch resident on MPS and ping-pongs two fixed buffers.
# apply()/apply_program() perform no device->host synchronization.  The only
# host synchronization is when metadata()/to_lists() is explicitly requested.
#
# apply_program() is the important path for GA evaluation: an entire sequence
# of shared Replace rules is executed inside ONE Metal kernel launch, one
# thread per sequence, reusing the V2 serial-per-job strategy that benchmarked
# better than fine-grained candidate parallelism on short (~1500 token) rows.
# ============================================================================

from dataclasses import dataclass as _state_dataclass
from typing import Any as _Any

_MPS_STATE_LIB = None

# Reuse V2's Metal helpers (match_at, etc.) and append a program kernel.
_MPS_STATE_SOURCE = _MPS_FUSED_SOURCE + r'''

inline void state_copy_row(
    const device int* src,
    int src_base,
    device int* dst,
    int dst_base,
    int n)
{
    for (int i = 0; i < n; ++i) {
        dst[dst_base + i] = src[src_base + i];
    }
}

inline bool state_write_token(
    device int* dst,
    int dst_base,
    int capacity,
    int logical_max_output,
    thread int& out_pos,
    int value,
    const device int* src,
    int src_base,
    int input_len,
    int track_changed,
    thread bool& changed)
{
    if (out_pos >= capacity || out_pos >= logical_max_output) {
        return false;
    }
    dst[dst_base + out_pos] = value;
    if (track_changed != 0) {
        if (out_pos >= input_len || src[src_base + out_pos] != value) {
            changed = true;
        }
    }
    out_pos += 1;
    return true;
}

inline int state_apply_one_rule(
    const device int* src,
    device int* dst,
    int base,
    int n,
    int capacity,
    int logical_max_output,
    const device int* pattern,
    int pattern_len,
    const device int* replacement,
    int replacement_len,
    int wildcard_count,
    int has_leading_literal,
    int track_changed,
    thread int& matched_out,
    thread int& applied_out,
    thread int& changed_out,
    thread int& overflow_out,
    thread int& selected_out)
{
    matched_out = 0;
    applied_out = 0;
    changed_out = 0;
    overflow_out = 0;
    selected_out = 0;

    if (n <= 0 || pattern_len <= 0) {
        state_copy_row(src, base, dst, base, n);
        return n;
    }

    int out_pos = 0;
    int prev = 0;
    int scan_pos = 0;
    int selected = 0;
    bool changed = false;
    bool matched = false;

    int cap_start[16];
    int cap_len[16];

    while (scan_pos < n) {
        int s = -1;
        int finish = -1;

        if (has_leading_literal != 0) {
            int first_literal = pattern[0];
            for (int cand = scan_pos; cand < n; ++cand) {
                if (src[base + cand] != first_literal) {
                    continue;
                }
                int f = -1;
                if (match_at(src, base, n, pattern, pattern_len,
                             cand, cap_start, cap_len, f)) {
                    s = cand;
                    finish = f;
                    break;
                }
            }
            if (s < 0) {
                break;
            }
        } else {
            int f = -1;
            if (!match_at(src, base, n, pattern, pattern_len,
                          scan_pos, cap_start, cap_len, f)) {
                break;
            }
            s = scan_pos;
            finish = f;
        }

        matched = true;
        selected += 1;

        // Untouched prefix between selected matches.
        for (int i = prev; i < s; ++i) {
            if (!state_write_token(dst, base, capacity, logical_max_output,
                                   out_pos, src[base + i], src, base, n,
                                   track_changed, changed)) {
                state_copy_row(src, base, dst, base, n);
                matched_out = 1;
                overflow_out = 1;
                selected_out = selected;
                return n;
            }
        }

        // Replacement program.  sort($n) is excluded by Python validation.
        for (int r = 0; r < replacement_len; ++r) {
            int t = replacement[r];
            if (t >= 0) {
                if (!state_write_token(dst, base, capacity, logical_max_output,
                                       out_pos, t, src, base, n,
                                       track_changed, changed)) {
                    state_copy_row(src, base, dst, base, n);
                    matched_out = 1;
                    overflow_out = 1;
                    selected_out = selected;
                    return n;
                }
                continue;
            }

            if (wildcard_count <= 0) {
                continue;
            }

            int kind = 0;
            int ci = 0;
            if (t >= -15) {
                kind = 1; // capture
                ci = -t - 1;
            } else if (t >= -31) {
                // Unsupported in persistent fused path: passthrough safely.
                state_copy_row(src, base, dst, base, n);
                return n;
            } else if (t >= -47) {
                kind = 2; // reverse
                ci = -t - 32;
            } else if (t >= -111) {
                int off = -t - 48;
                int bank = off / 16;
                ci = off - bank * 16;
                kind = 3 + bank; // +1, -1, *2, //2
            } else {
                state_copy_row(src, base, dst, base, n);
                return n;
            }

            if (ci >= wildcard_count) {
                ci = 0;
            }
            int cs = cap_start[ci];
            int cl = cap_len[ci];

            for (int q = 0; q < cl; ++q) {
                int src_local = (kind == 2) ? (cs + cl - 1 - q) : (cs + q);
                int v = src[base + src_local];
                int y = v;
                if (kind == 3) {
                    if (v >= 0 && v < 512) y = (v + 1) & 511;
                } else if (kind == 4) {
                    if (v >= 0 && v < 512) y = (v + 511) & 511;
                } else if (kind == 5) {
                    if (v >= 0 && v < 512) y = (v * 2) & 511;
                } else if (kind == 6) {
                    if (v < 512) {
                        y = (v >= 0) ? (v / 2) : -(((-v) + 1) / 2);
                    }
                }

                if (!state_write_token(dst, base, capacity, logical_max_output,
                                       out_pos, y, src, base, n,
                                       track_changed, changed)) {
                    state_copy_row(src, base, dst, base, n);
                    matched_out = 1;
                    overflow_out = 1;
                    selected_out = selected;
                    return n;
                }
            }
        }

        prev = finish;
        scan_pos = finish;
    }

    // For state chaining, no-match means identity, not an empty sequence.
    if (!matched) {
        state_copy_row(src, base, dst, base, n);
        return n;
    }

    for (int i = prev; i < n; ++i) {
        if (!state_write_token(dst, base, capacity, logical_max_output,
                               out_pos, src[base + i], src, base, n,
                               track_changed, changed)) {
            state_copy_row(src, base, dst, base, n);
            matched_out = 1;
            overflow_out = 1;
            selected_out = selected;
            return n;
        }
    }

    if (track_changed != 0 && out_pos != n) {
        changed = true;
    }
    matched_out = 1;
    applied_out = 1;
    changed_out = changed ? 1 : 0;
    overflow_out = 0;
    selected_out = selected;
    return out_pos;
}

kernel void replace_program_state_i32(
    device int* buffer_a [[buffer(0)]],
    device int* buffer_b [[buffer(1)]],
    device int* lengths [[buffer(2)]],
    const device int* patterns [[buffer(3)]],
    const device int* pat_offsets [[buffer(4)]],
    const device int* pat_lengths [[buffer(5)]],
    const device int* replacements [[buffer(6)]],
    const device int* rep_offsets [[buffer(7)]],
    const device int* rep_lengths [[buffer(8)]],
    const device int* wildcard_counts [[buffer(9)]],
    const device int* leading_flags [[buffer(10)]],
    device int* program_stats [[buffer(11)]],
    constant int& capacity [[buffer(12)]],
    constant int& logical_max_output [[buffer(13)]],
    constant int& rule_count [[buffer(14)]],
    constant int& track_changed [[buffer(15)]],
    uint job_u [[thread_position_in_grid]])
{
    int job = int(job_u);
    int base = job * capacity;
    int n = lengths[job];
    bool current_is_a = true;

    int matched_rules = 0;
    int applied_rules = 0;
    int changed_rules = 0;
    int overflow_rules = 0;
    int selected_total = 0;

    for (int r = 0; r < rule_count; ++r) {
        device int* src_rw = current_is_a ? buffer_a : buffer_b;
        device int* dst = current_is_a ? buffer_b : buffer_a;
        const device int* src = src_rw;

        int m = 0;
        int a = 0;
        int c = 0;
        int o = 0;
        int s = 0;

        const device int* pat = patterns + pat_offsets[r];
        const device int* rep = replacements + rep_offsets[r];

        n = state_apply_one_rule(
            src, dst, base, n, capacity, logical_max_output,
            pat, pat_lengths[r], rep, rep_lengths[r],
            wildcard_counts[r], leading_flags[r], track_changed,
            m, a, c, o, s);

        matched_rules += m;
        applied_rules += a;
        changed_rules += c;
        overflow_rules += o;
        selected_total += s;

        // Every rule writes a complete next state (result or passthrough), so
        // parity is identical for every job and Python can swap buffers once.
        current_is_a = !current_is_a;
    }

    lengths[job] = n;
    int q = job * 5;
    program_stats[q + 0] = matched_rules;
    program_stats[q + 1] = applied_rules;
    program_stats[q + 2] = changed_rules;
    program_stats[q + 3] = overflow_rules;
    program_stats[q + 4] = selected_total;
}
'''


def _get_mps_state_lib():
    global _MPS_STATE_LIB
    if _MPS_STATE_LIB is None:
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("torch.mps.compile_shader is unavailable")
        _MPS_STATE_LIB = torch.mps.compile_shader(_MPS_STATE_SOURCE)
    return _MPS_STATE_LIB


@_state_dataclass
class MPSRuleProgram:
    """Device-resident packed rule program for GpuReplaceState."""

    key: tuple
    rule_count: int
    patterns: torch.Tensor
    pat_offsets: torch.Tensor
    pat_lengths: torch.Tensor
    replacements: torch.Tensor
    rep_offsets: torch.Tensor
    rep_lengths: torch.Tensor
    wildcard_counts: torch.Tensor
    leading_flags: torch.Tensor


class GpuReplaceState:
    """Persistent V2-style Replace state for Apple MPS.

    The batch remains padded and resident on the GPU.  Two [B, capacity] int32
    buffers ping-pong between rules.  No CPU synchronization occurs in apply()
    or apply_program().  Call to_lists()/metadata() only when host data is
    actually needed.

    Fast path constraints are the same as V2's fused MPS kernel:
      * one shared rule/program is applied to every row in the batch;
      * int32 tokens;
      * <= 16 wildcards;
      * sort($capture) is not supported in the fused persistent path.

    No-match and overflow preserve the previous row in state, matching the
    usual caller behavior of only committing Replace when applied=True.
    """

    def __init__(
        self,
        inputs: Sequence[Union[Sequence[int], torch.Tensor]],
        *,
        capacity: Optional[int] = None,
        max_output: Optional[int] = None,
        compute_changed: bool = False,
    ):
        if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
            raise RuntimeError("GpuReplaceState persistent fast path currently requires Apple MPS")

        self.device = torch.device("mps")
        self.batch_size = len(inputs)
        self.compute_changed = bool(compute_changed)

        host_rows: List[torch.Tensor] = []
        lengths_host: List[int] = []
        max_n = 0
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                row = inp.detach().to(device="cpu", dtype=torch.int32).contiguous().flatten()
            else:
                row = torch.tensor(list(map(int, inp)), dtype=torch.int32)
            host_rows.append(row)
            n = int(row.numel())
            lengths_host.append(n)
            max_n = max(max_n, n)

        if capacity is None:
            # For GA workloads around 1500 tokens, 2x headroom is a practical
            # default.  Pass an explicit capacity when a rule set can grow more.
            capacity = max(1, max_n * 2)
        capacity = int(capacity)
        if capacity < max_n:
            raise ValueError(f"capacity={capacity} is smaller than max input length {max_n}")
        self.capacity = capacity
        self.max_output = int(max_output if max_output is not None else capacity)
        if self.max_output <= 0 or self.max_output > self.capacity:
            raise ValueError("max_output must be in 1..capacity")

        host = torch.zeros((self.batch_size, self.capacity), dtype=torch.int32)
        for j, row in enumerate(host_rows):
            n = int(row.numel())
            if n:
                host[j, :n] = row

        # One initial H->MPS copy.  Everything thereafter stays resident.
        self._a = host.to(self.device)
        self._b = torch.empty_like(self._a)
        self._current_is_a = True
        self.lengths = torch.tensor(lengths_host, dtype=torch.int32, device=self.device)
        self.program_stats = torch.zeros((self.batch_size, 5), dtype=torch.int32, device=self.device)
        self._program_cache: Dict[tuple, MPSRuleProgram] = {}
        self.rules_applied = 0

    @property
    def current(self) -> torch.Tensor:
        """Current padded [B, capacity] device tensor; no synchronization."""
        return self._a if self._current_is_a else self._b

    @property
    def workspace(self) -> torch.Tensor:
        return self._b if self._current_is_a else self._a

    def _normalize_rules(self, rules) -> List[Tuple[List[int], List[int]]]:
        def is_one_int_sequence(v) -> bool:
            if isinstance(v, torch.Tensor):
                return v.ndim == 1
            if not isinstance(v, (list, tuple)):
                return False
            return len(v) == 0 or isinstance(v[0], int)

        # One rule may be passed directly as (pattern, replacement).
        if (
            isinstance(rules, tuple)
            and len(rules) == 2
            and is_one_int_sequence(rules[0])
            and is_one_int_sequence(rules[1])
        ):
            rules = [rules]

        out: List[Tuple[List[int], List[int]]] = []
        for pattern, replacement in rules:
            if isinstance(pattern, torch.Tensor):
                pattern = pattern.detach().cpu().tolist()
            if isinstance(replacement, torch.Tensor):
                replacement = replacement.detach().cpu().tolist()
            p = [int(x) for x in pattern]
            r = [int(x) for x in replacement]

            cp = _compile_pattern(p)
            if cp.empty_pattern:
                raise ValueError("empty pattern is not supported by the fused persistent path")
            # All-wildcard patterns are valid in the fused state kernel.
            # match_at() gives consecutive interior wildcards the shortest
            # capture (empty) and a trailing wildcard captures the remaining
            # suffix, matching the CPU reference semantics.
            if cp.wildcard_count > MAX_WILDCARDS:
                raise ValueError(f"pattern has {cp.wildcard_count} wildcards; max is {MAX_WILDCARDS}")
            _compile_replacement(r)  # validates opcode range + length
            if any(-31 <= int(t) <= -16 for t in r):
                raise ValueError("sort($capture) is not supported in persistent fused MPS mode")
            out.append((p, r))
        return out

    def prepare_program(self, rules) -> MPSRuleProgram:
        """Pack/cache rule tensors on MPS.  Reusing a program has no H->D copies."""
        normalized = self._normalize_rules(rules)
        key = tuple((tuple(p), tuple(r)) for p, r in normalized)
        cached = self._program_cache.get(key)
        if cached is not None:
            return cached
        if not normalized:
            # Zero-rule program is represented but never dispatched.
            z = torch.empty(0, dtype=torch.int32, device=self.device)
            program = MPSRuleProgram(key, 0, z, z, z, z, z, z, z, z)
            self._program_cache[key] = program
            return program

        pat_flat: List[int] = []
        pat_offsets: List[int] = []
        pat_lengths: List[int] = []
        rep_flat: List[int] = []
        rep_offsets: List[int] = []
        rep_lengths: List[int] = []
        wildcard_counts: List[int] = []
        leading_flags: List[int] = []

        for p, r in normalized:
            cp = _compile_pattern(p)
            pat_offsets.append(len(pat_flat))
            pat_lengths.append(len(p))
            pat_flat.extend(p)
            rep_offsets.append(len(rep_flat))
            rep_lengths.append(len(r))
            rep_flat.extend(r)
            wildcard_counts.append(cp.wildcard_count)
            leading_flags.append(1 if cp.has_leading_literal else 0)

        # Avoid zero-byte Metal buffers for an all-empty replacement program.
        rep_storage = rep_flat if rep_flat else [0]
        program = MPSRuleProgram(
            key=key,
            rule_count=len(normalized),
            patterns=torch.tensor(pat_flat, dtype=torch.int32, device=self.device),
            pat_offsets=torch.tensor(pat_offsets, dtype=torch.int32, device=self.device),
            pat_lengths=torch.tensor(pat_lengths, dtype=torch.int32, device=self.device),
            replacements=torch.tensor(rep_storage, dtype=torch.int32, device=self.device),
            rep_offsets=torch.tensor(rep_offsets, dtype=torch.int32, device=self.device),
            rep_lengths=torch.tensor(rep_lengths, dtype=torch.int32, device=self.device),
            wildcard_counts=torch.tensor(wildcard_counts, dtype=torch.int32, device=self.device),
            leading_flags=torch.tensor(leading_flags, dtype=torch.int32, device=self.device),
        )
        self._program_cache[key] = program
        return program

    def apply_program(self, rules_or_program) -> "GpuReplaceState":
        """Apply an entire shared rule sequence in one Metal kernel launch.

        This is the preferred GA path.  It performs no host readback and swaps
        the persistent buffers only according to the (global) rule-count parity.
        """
        if isinstance(rules_or_program, MPSRuleProgram):
            program = rules_or_program
        else:
            program = self.prepare_program(rules_or_program)
        if program.rule_count == 0 or self.batch_size == 0:
            return self

        lib = _get_mps_state_lib()
        group = min(64, max(1, self.batch_size))
        # The kernel assumes its first buffer is the current state.  Pass the
        # actual current/workspace tensors in that order; parity is handled below.
        cur = self.current
        work = self.workspace
        lib.replace_program_state_i32(
            cur,
            work,
            self.lengths,
            program.patterns,
            program.pat_offsets,
            program.pat_lengths,
            program.replacements,
            program.rep_offsets,
            program.rep_lengths,
            program.wildcard_counts,
            program.leading_flags,
            self.program_stats,
            int(self.capacity),
            int(self.max_output),
            int(program.rule_count),
            int(self.compute_changed),
            threads=[self.batch_size, 1, 1],
            group_size=[group, 1, 1],
        )

        # Same parity for every row because no-match/overflow is copied through.
        if program.rule_count & 1:
            self._current_is_a = not self._current_is_a
        self.rules_applied += program.rule_count
        return self

    def apply(self, pattern, replacement) -> "GpuReplaceState":
        """Apply one shared rule without CPU synchronization."""
        return self.apply_program([(pattern, replacement)])

    def metadata(self) -> dict:
        """Synchronize only small O(B) metadata for the most recent program."""
        if self.batch_size == 0:
            return {"lengths": [], "program_stats": []}
        # One combined device->host copy/sync.
        meta = torch.cat((self.lengths[:, None], self.program_stats), dim=1).cpu().tolist()
        return {
            "lengths": [int(r[0]) for r in meta],
            "program_stats": [tuple(int(v) for v in r[1:]) for r in meta],
        }

    def to_lists(self) -> List[List[int]]:
        """Final one-shot readback of lengths + padded state."""
        if self.batch_size == 0:
            return []
        # Concatenate on-device so this is a single final D->H transfer/sync.
        combined = torch.cat((self.lengths, self.current.reshape(-1))).cpu().tolist()
        lengths = [int(x) for x in combined[: self.batch_size]]
        flat = combined[self.batch_size :]
        out: List[List[int]] = []
        for j, n in enumerate(lengths):
            base = j * self.capacity
            out.append([int(x) for x in flat[base : base + n]])
        return out

    def clone_device(self) -> "GpuReplaceState":
        """Cheap-ish explicit device clone for branching experiments; no host sync."""
        obj = object.__new__(GpuReplaceState)
        obj.device = self.device
        obj.batch_size = self.batch_size
        obj.compute_changed = self.compute_changed
        obj.capacity = self.capacity
        obj.max_output = self.max_output
        obj._a = self.current.clone()
        obj._b = torch.empty_like(obj._a)
        obj._current_is_a = True
        obj.lengths = self.lengths.clone()
        obj.program_stats = self.program_stats.clone()
        obj._program_cache = self._program_cache  # immutable cached programs
        obj.rules_applied = self.rules_applied
        return obj


def persistent_mps_smoke_reference(
    inputs: Sequence[Sequence[int]],
    rules: Sequence[Tuple[Sequence[int], Sequence[int]]],
) -> List[List[int]]:
    """CPU reference for checking persistent-program semantics.

    Unlike replace_batch_gpu's per-call result contract, state semantics retain
    the old sequence on no-match/overflow because there is nothing to commit.
    """
    state = [list(map(int, row)) for row in inputs]
    for pat, rep in rules:
        batch = replace_batch_gpu(
            state,
            pat,
            rep,
            device="cpu",
            token_dtype=torch.int32,
            compute_changed=False,
        )
        next_state: List[List[int]] = []
        for old, rr in zip(state, batch.results):
            next_state.append(rr.data.cpu().tolist() if rr.applied else old)
        state = next_state
    return state
