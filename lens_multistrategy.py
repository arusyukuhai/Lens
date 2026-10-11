"""Multi-strategy *outer* Replacer rule proposals for direct Pearson learning.

All clean/corrupted training examples are drawn from non-held-out source chunks.
The proxy is deliberately only a candidate ranking heuristic: every child is
re-scored by the ordinary exact sequential Replacer in the next generation.
Inner string reconstruction GA is unaffected.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import random
from typing import Sequence
import numpy as np

MODES = ("repair", "wildcard", "ngram", "diff3", "one_token", "splice")
DEFAULT_WEIGHTS = (0.29, 0.16, 0.15, 0.17, 0.14, 0.09)


def parse_modes(text: str) -> tuple[str, ...]:
    modes = tuple(dict.fromkeys(x.strip().lower() for x in text.split(",") if x.strip()))
    if not modes:
        raise ValueError("--guided-strategies: select at least one strategy")
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError("unknown guided strategy: " + ", ".join(sorted(unknown)))
    return modes


def parse_weights(text: str, modes: Sequence[str]) -> dict[str, float]:
    """Explicit weights in the SAME order as --guided-strategies.

    Empty string uses canonical mode defaults, independent of CLI order.
    """
    if text.strip():
        vals = [float(t.strip()) for t in text.split(",")]
        if len(vals) != len(modes):
            raise ValueError("--guided-strategy-weights must match the number of strategies")
    else:
        table = dict(zip(MODES, DEFAULT_WEIGHTS))
        vals = [table[m] for m in modes]
    if any(not math.isfinite(v) or v < 0 for v in vals) or not any(v > 0 for v in vals):
        raise ValueError("guided weights must be nonnegative, finite, and not all zero")
    s = sum(vals)
    return {m: v / s for m, v in zip(modes, vals)}


def _training_sources(rolling, max_source: int) -> tuple[list[tuple[bytes, bytes]], list[bytes], np.ndarray, int]:
    """Use precisely the source split used by the OLS fit (last chunk held out)."""
    ntrain = len(rolling.slots) - 1 if len(rolling.slots) >= 2 else len(rolling.slots)
    pairs = []
    rows = []
    for case in rolling.slots[:ntrain]:
        clean = bytes(case.inputs[int(np.argmax(case.target))])
        for v, score in zip(case.inputs, case.target):
            damaged = bytes(v)
            rows.append(damaged)
            if float(score) < 1.0 - 1e-9 and len(damaged) == len(clean):
                pairs.append((clean, damaged))
    random.shuffle(pairs)
    return pairs[:max(1, int(max_source))], rows, np.array([len(x.inputs) for x in rolling.slots], int), ntrain


def _window(clean: bytes, damaged: bytes, at: int, left: int, right: int):
    start = max(0, at - left)
    stop = min(len(clean), at + right + 1)
    a = tuple(map(int, damaged[start:stop]))
    b = list(a)
    b[at - start] = int(clean[at])
    return a, tuple(b), at - start


def _anchored_proxy(pattern: tuple[int, ...], inputs: list[bytes]) -> np.ndarray:
    """Cheap literal-anchor *necessary* condition, not actual rule firing.

    Do not convert negative wildcard tokens to bytes; use the longest literal
    run for a bounded, cautious upper-bound proxy. This cannot validate a rule.
    """
    runs = []
    part = []
    for v in pattern:
        if 0 <= v < 256:
            part.append(v)
        else:
            if part:
                runs.append(bytes(part))
                part = []
    if part:
        runs.append(bytes(part))
    if not runs:
        return np.zeros(len(inputs), dtype=np.float64)
    anchor = max(runs, key=len)
    return np.fromiter((1.0 if anchor in x else 0.0 for x in inputs),
                       dtype=np.float64, count=len(inputs))


def _raw_to_rule(m, sampler, embedding, a, b):
    # Negative tokens in the pattern are wildcards, -1 in replacement is
    # wildcard #1 capture; positive bytes are recoded to genome token IDs.
    r = m.Rule([int(embedding[v]) if v >= 0 else int(v) for v in a],
               [int(embedding[v]) if v >= 0 else int(v) for v in b])
    m.sanitize_rule(r, sampler, embedding)
    if r.pattern == r.replacement or not m.is_rule_nonexpanding(r):
        return None
    return r


def _candidate_score(a, rows, clean_controls, pair_rows, residual):
    z = _anchored_proxy(a, rows)
    freq = z.sum()
    if freq < 1 or freq >= len(rows):
        return None
    z -= z.mean()
    var = float(z @ z)
    if var < 1e-9:
        return None
    correlation = float(z @ residual)
    gain = correlation ** 2 / (var + 1e-6)
    # Penalize triggers already common in clean text. Wildcards are scored
    # conservatively by their longest literal run, never by alleged exact hits.
    clean_hits = sum(bool(_anchored_proxy(a, [s])[0]) for s in clean_controls)
    noisy_hits = sum(bool(_anchored_proxy(a, [s])[0]) for s in pair_rows)
    contrast = (1.0 + math.log1p(noisy_hits)) / (1 + 3.0 * clean_hits)
    return float(gain * contrast / (1.0 + 0.03 * len(a)))


def mine_strategies(rolling, champion, feats, baseline, target, sampler, m,
                    limit=48, source_cap=48, modes=MODES,
                    ):
    """Return {mode: [(proxy_quality, Rule), ...]}.

    Strategy families:
      repair: supervised local literal correction
      wildcard: preserve a local context capture while repairing one byte
      ngram: extend literal context with observed training n-grams (2..7)
      diff3: git/SequenceMatcher-style 3-way edit merge: (base->donor)+target
      one_token: one-token structural perturbations of existing genome rules
      splice: recombine two independently mined repair windows
    """
    modes = tuple(modes)
    out = {mode: [] for mode in modes}
    pairs, rows, lengths, ntrain = _training_sources(rolling, source_cap)
    if not pairs or len(rows) < 3:
        return out
    train_count = int(lengths[:ntrain].sum())
    w = np.asarray(champion.readout_weights, dtype=np.float64)
    if len(w) != feats.shape[1]:
        w = np.zeros(feats.shape[1], dtype=np.float64)
    residual = np.asarray(target[:train_count], dtype=np.float64) - (
        np.asarray(baseline[:train_count], dtype=np.float64) +
        np.asarray(feats[:train_count], dtype=np.float64) @ w
    )
    residual = residual - float(np.mean(residual))
    raw: dict[str, Counter] = {mode: Counter() for mode in modes}
    clean_controls = [bytes(c.inputs[int(np.argmax(c.target))]) for c in rolling.slots[:ntrain]]
    pair_rows = [s for _, s in pairs]

    def put(mode, a, b):
        if mode in raw and a and b and a != b and 1 <= len(a) <= m.MAX_RULE_TOKENS and len(b) <= m.MAX_RULE_TOKENS:
            raw[mode][(tuple(a), tuple(b))] += 1

    for clean, damaged in pairs:
        differences = [i for i, (a, b) in enumerate(zip(clean, damaged)) if a != b]
        if not differences:
            continue
        for at in random.sample(differences, min(9, len(differences))):
            base_a, base_b, offset = _window(clean, damaged, at, 1, 1)
            if len(base_a) < 2:
                continue
            for l, r in ((1, 1), (2, 1), (1, 2), (0, 2), (2, 0)):
                a, b, _ = _window(clean, damaged, at, l, r)
                if len(a) >= 2:
                    put("repair", a, b)
            if "ngram" in raw:
                # Literal spans are observed training n-grams around the actual
                # error, not unrelated random n-grams grafted onto its cause.
                for l, r in ((3, 1), (1, 3), (2, 3), (3, 2), (3, 3)):
                    a, b, _ = _window(clean, damaged, at, l, r)
                    if 3 <= len(a) <= 7:
                        put("ngram", a, b)
            if "wildcard" in raw or "diff3" in raw:
                # Contextual wildcard copies the original byte(s); never mask
                # the corrupted locus itself, or a rule could match a clean row.
                for flank in (0, len(base_a) - 1):
                    if flank == offset:
                        continue
                    pa, pb = list(base_a), list(base_b)
                    pa[flank] = -1
                    pb[flank] = -1  # capture 1, not a literal replacement
                    put("wildcard", pa, pb)
                # True 3-way *sequence* merge. Both versions derive from the
                # same base event; donor generalizes one flank, while target
                # grows the opposite flank. Conflict checks are performed by
                # main.merge_sequence_delta, same as historical v60 diff3.
                if "diff3" in raw and offset > 0 and at + 2 < len(clean):
                    donor_a, donor_b = list(base_a), list(base_b)
                    donor_a[0] = -1
                    donor_b[0] = -1
                    target_a, target_b, _ = _window(clean, damaged, at, 1, 2)
                    p, pe = m.merge_sequence_delta(base_a, donor_a, target_a)
                    q, qe = m.merge_sequence_delta(base_b, donor_b, target_b)
                    if pe > 0 and qe > 0 and p != list(target_a):
                        put("diff3", p, q)
            if "splice" in raw:
                # Two distinct windows of the same repair evidence: perform a
                # 2-point prefix/suffix splice while preserving the changed
                # central byte in both pattern and replacement.
                a1, b1, o1 = _window(clean, damaged, at, 2, 2)
                a2, b2, o2 = _window(clean, damaged, at, 1, 3)
                if len(a1) >= 4 and len(a2) >= 4:
                    ca = a1[:o1] + a2[o2:]
                    cb = b1[:o1] + b2[o2:]
                    put("splice", ca, cb)
    # Coordinate-edit rule proposals are already in genome token IDs.
    # Collect these separately, never convert them through the embedding twice.
    coordinate_edits = []
    if "one_token" in modes and champion.rules:
        selected = list(np.argsort(np.abs(w))[-min(len(w), 12):]) if len(w) == len(champion.rules) else list(range(min(12, len(champion.rules))))
        for ix in selected:
            old = champion.rules[int(ix)]
            for _ in range(3):
                clean, damaged = random.choice(pairs)
                positions = [i for i,(a,b) in enumerate(zip(clean, damaged)) if a != b]
                if not positions:
                    continue
                at = random.choice(positions)
                for which, ch in (("pattern", damaged[at]), ("replacement", clean[at])):
                    oldseq = getattr(old, which)
                    if not oldseq:
                        continue
                    p, q = list(old.pattern), list(old.replacement)
                    update = p if which == "pattern" else q
                    update[random.randrange(len(update))] = int(champion.embedding[int(ch)])
                    if p != old.pattern or q != old.replacement:
                        coordinate_edits.append((p, q))
    # Limit expensive proxy scans: keep frequent training-aligned variants, but
    # do not entirely starve rare diff3 or wildcard proposals.
    per_mode_budget = max(int(limit) * 6, 96)
    for mode in modes:
        scored = []
        if mode == "one_token":
            inv = {int(e): int(i) for i, e in enumerate(champion.embedding)}
            for p, q in coordinate_edits[:per_mode_budget]:
                # Convert positive embedding coordinates back to actual bytes
                # ONLY for proxy presence testing; negative opcodes retained.
                raw_p = tuple(inv.get(int(x), 999) if x >= 0 else int(x) for x in p)
                score = _candidate_score(raw_p, rows, clean_controls, pair_rows, residual)
                if score is None:
                    score = 1e-12  # novel patterns still get a small exploration budget
                rr = m.Rule(p, q)
                m.sanitize_rule(rr, sampler, champion.embedding)
                if rr.pattern != rr.replacement and m.is_rule_nonexpanding(rr):
                    scored.append((float(score), rr))
        else:
            for (a, b), count in raw[mode].most_common(per_mode_budget):
                score = _candidate_score(a, rows, clean_controls, pair_rows, residual)
                if score is None:
                    continue
                rule = _raw_to_rule(m, sampler, champion.embedding, a, b)
                if rule is not None:
                    scored.append((float(score) * (1.0 + 0.025 * min(count, 5)), rule))
        seen = set()
        for score, rule in sorted(scored, key=lambda v: v[0], reverse=True):
            key = (tuple(rule.pattern), tuple(rule.replacement))
            if key in seen:
                continue
            seen.add(key)
            out[mode].append((score, rule))
            if len(out[mode]) >= max(1, int(limit)):
                break
    return out


def make_multi_offspring(m, champion, proposals, replace_ids, count, weights=None):
    """Weighted deficit schedule with dedup across all lanes, fixed population.

    Returns children and count per strategy; no extra fitness evaluations. The
    true parent baseline is captured for next-generation contribution reports.
    """
    lanes = {k: v for k, v in proposals.items() if v}
    counts = {k: 0 for k in proposals}
    if count <= 0 or not lanes or not replace_ids:
        return [], counts
    weight = {k: float((weights or {}).get(k, 1.0)) for k in lanes}
    weight = {k: v for k, v in weight.items() if v > 0}
    if not weight:
        return [], counts
    total = sum(weight.values())
    credit = {k: 0.0 for k in weight}
    cursors = {k: 0 for k in weight}
    output = []
    seen = set()
    trials = 0
    max_trials = max(256, 8 * int(count))
    while len(output) < count and trials < max_trials:
        trials += 1
        for name in credit:
            credit[name] += weight[name] / total
        mode = max(credit, key=lambda k: credit[k])
        credit[mode] -= 1.0
        prop = lanes[mode]
        idx = cursors[mode]
        cursors[mode] += 1
        # Vary the edited locus and the proposed rule independently.
        row_offset = idx % len(replace_ids)
        row = int(replace_ids[row_offset])
        # Cycle the proposal phase after each full rule-slot pass, so even a
        # one-proposal lane can eventually try all replacement coordinates.
        rule_idx = (row_offset + idx // len(replace_ids)) % len(prop)
        rule = prop[rule_idx][1]
        sig = (row, tuple(rule.pattern), tuple(rule.replacement))
        if sig in seen:
            continue
        seen.add(sig)
        old = champion.rules[row]
        if list(rule.pattern) == list(old.pattern) and list(rule.replacement) == list(old.replacement):
            continue
        child = m.clone_genome_shallow(champion)
        child.rules[row] = m.clone_rule(rule)
        m._mark_genome_pack_rows(child, (row,))
        m._clear_evaluation_state(child)
        m.mark_offspring_origin(child, champion, 'guided:' + mode, 1)
        output.append(child)
        counts[mode] += 1
        if len(seen) >= sum(len(v) for v in lanes.values()) * len(replace_ids):
            break
    return output, counts
