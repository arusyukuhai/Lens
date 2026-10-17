"""Old revision's deterministic byte Replace, with number of firings.

The transformation is intentionally identical to discrete_replace.discrete_replace.
Count is an auxiliary measurement; an overflow->identity produces zero firings.
This code is not regex syntax: it preserves the old cursor rules exactly.
"""
from functools import lru_cache


@lru_cache(maxsize=65536)
def _compile(pattern: tuple[int, ...], replacement: tuple[int, ...]):
    if len(replacement) > 64:
        raise ValueError('replacement capacity exceeds 64')
    if any(not -15 <= t < 256 for t in pattern + replacement):
        raise ValueError('unsupported byte/opcode')
    pieces = [bytearray()]
    for tok in pattern:
        if tok < 0:
            pieces.append(bytearray())
        else:
            pieces[-1].append(tok)
    return tuple(bytes(s) for s in pieces), bytes(t for t in replacement if t >= 0)


def replace_with_count(x: bytes, pattern: tuple[int, ...], replacement: tuple[int, ...],
                       *, min_wildcard_len: int = 0, max_output: int = 1048576):
    if min_wildcard_len < 0 or max_output < 0:
        raise ValueError('negative limit')
    if not isinstance(x, bytes):
        x = bytes(x)
    pieces, literal_rhs = _compile(pattern, replacement)
    nstars = len(pieces) - 1
    if not pattern or nstars > 16 or (nstars and not any(pieces)):
        return x, 0
    if nstars == 0:
        count = x.count(pieces[0])
        if not count:
            return x, 0
        result = x.replace(pieces[0], literal_rhs)
        return (result, count) if len(result) <= max_output else (x, 0)

    size, first = len(x), pieces[0]
    trailers = pieces[1:]

    def match(start):
        end = start + len(first)
        caps = []
        for w, literal in enumerate(trailers):
            if not literal:
                stop = size if w == nstars - 1 else end
                caps.append(x[end:stop])
                end = stop
            else:
                found = x.find(literal, end + min_wildcard_len)
                if found < 0:
                    return None
                caps.append(x[end:found])
                end = found + len(literal)
        return end, caps

    out = bytearray()
    pos = copy_pos = 0
    fired = 0
    while pos < size:
        start = x.find(first, pos) if first else pos
        if start < 0:
            break
        matched = match(start)
        if matched is None:
            if not first:
                break
            next_start = x.find(first, start + 1)
            if next_start < 0:
                break
            matched = match(next_start)
            if matched is None:
                pos = next_start
                continue
            start = next_start
        end, caps = matched
        out.extend(x[copy_pos:start])
        for tok in replacement:
            if tok >= 0:
                out.append(tok)
            elif caps:
                ref = -tok - 1
                out.extend(caps[ref if ref < len(caps) else 0])
        if len(out) > max_output:
            return x, 0
        fired += 1
        pos = copy_pos = end
    if not fired:
        return x, 0
    out.extend(x[copy_pos:])
    return (bytes(out), fired) if len(out) <= max_output else (x, 0)


def serial_features(data: bytes, rules, *, min_wildcard_len=0, max_output=1048576):
    """Exactly one global rule in each stage; no per-occurrence resampling.

    Supports arbitrary expansion/deletion and same-stage non-overlapping matches.
    features[r] = actual replacement firings / ORIGINAL input byte count.
    """
    from array import array
    original_len = len(data)
    if not original_len:
        raise ValueError('empty text')
    features = array('f', [0.0]) * len(rules)
    for i, (a, b) in enumerate(rules):
        data, count = replace_with_count(data, a, b, min_wildcard_len=min_wildcard_len,
                                         max_output=max_output)
        features[i] = count / original_len
    return features, data
