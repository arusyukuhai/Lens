"""Dependency-free integer deployment of the supplied byte/capture semantics."""

def discrete_replace(x, a, b, *, min_wildcard_len=0, max_output=1_048_576):
    """Corrected byte/capture subset of replaceSeqCompiled (overflow -> identity).

    a: bytes or any negative wildcard. b: bytes or -1..-15 capture refs.
    No sort/reverse/inverse_sort. Explicitly reject replacements over 64 ops.
    """
    x, a, b = tuple(x), tuple(a), tuple(b)
    if min_wildcard_len < 0 or max_output < 0:
        raise ValueError('negative limit')
    if len(b) > 64:
        raise ValueError('replacement capacity exceeds 64')
    if any(not 0 <= c < 256 for c in x):
        raise ValueError('input must contain bytes')
    if any(c >= 256 for c in a) or any(not -15 <= c < 256 for c in b):
        raise ValueError('unsupported symbol/opcode')
    parts = [[]]
    for c in a:
        if c < 0:
            parts.append([])
        else:
            parts[-1].append(c)
    parts = tuple(tuple(p) for p in parts)
    nw = len(parts) - 1
    if not a or nw > 16 or (nw and not any(parts)):
        return x

    def find(p, start):
        for i in range(start, len(x) - len(p) + 1):
            if x[i:i + len(p)] == p:
                return i
        return -1

    def match(start):
        end = start + len(parts[0])
        caps = []
        for w, literal in enumerate(parts[1:]):
            if not literal:
                stop = len(x) if w == nw - 1 else end
                caps.append(x[end:stop])
                end = stop
            else:
                found = find(literal, end + min_wildcard_len)
                if found < 0:
                    return None
                caps.append(x[end:found])
                end = found + len(literal)
        return end, caps

    out, pos, copy_pos, changed = [], 0, 0, False
    while pos < len(x):
        start = find(parts[0], pos) if parts[0] else pos
        if start < 0:
            break
        matched = match(start)
        if matched is None:
            if not parts[0]:
                break
            next_start = find(parts[0], start + 1)
            if next_start < 0:
                break
            matched = match(next_start)
            if matched is None:
                # Search cursor may advance; copy_pos must NOT advance until
                # a successful replacement consumes the pending input prefix.
                pos = next_start
                continue
            start = next_start
        end, caps = matched
        out.extend(x[copy_pos:start])
        for c in b:
            if c >= 0:
                out.append(c)
            elif caps:
                r = -c - 1
                out.extend(caps[r if r < len(caps) else 0])
        if len(out) > max_output:
            return x
        changed = True
        pos = end
        copy_pos = end
    if not changed:
        return x
    out.extend(x[copy_pos:])
    return x if len(out) > max_output else tuple(out)

