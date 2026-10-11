"""Bounded bidirectional *multi-rule* synthesis for Lens/Replacer.

A forward witness is a literal prefix observed in a training trajectory. A
backward constraint is an ordered suffix (and optionally a middle anchor) that
must be witnessed to create the final feature. We join them with private latent
markers. Proxies rank *whole chains*, never individual rules; the normal
sequential Replacer evaluator subsequently scores every resulting child.

This is intentionally a bounded BIPS prototype, not a general symbolic
preimage/CEGAR solver. Hidden-memory writes (-2/-3 pattern) are NEVER generated.
Existing checkpoint opcodes retain their historical meaning.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from typing import Sequence
import math
import numpy as np


@dataclass(frozen=True)
class PathProposal:
    score: float
    prefix: bytes
    suffix: bytes
    middle: bytes = b""
    bridges: tuple[bytes, ...] = ()
    origin_case: int = -1  # source original used to mine this path (-1: legacy)
    internal_anchors: tuple[tuple[int, ...], ...] = ()  # exact stage-token witnesses
    goal_kind: str = "marker"  # marker or an existing consumer rule
    consumer_index: int = -1
    consumer_pattern: tuple[int, ...] = ()

    @property
    def depth(self) -> int:
        return 2 + int(bool(self.middle)) + len(self.bridges)

    @property
    def anchors(self) -> tuple[bytes, ...]:
        return (self.prefix,) + ((self.middle,) if self.middle else ()) + self.bridges + (self.suffix,)

    @property
    def path_key(self):
        return (self.anchors,self.internal_anchors,self.origin_case,self.goal_kind,self.consumer_index)



def _ordered_occurrence(s: bytes, parts: tuple[bytes, ...]) -> bool:
    """Necessary condition for a forward/backward anchor connection."""
    start = 0
    for part in parts:
        where = s.find(part, start)
        if where < 0:
            return False
        start = where + len(part)
    return True


def available_latent_tokens(genome, amount: int = 3,
                            excluding_rule_rows: Sequence[int] = ()) -> tuple[int, ...]:
    """Fresh marker IDs cannot be input bytes nor existing literal constants.

    Scan *both* sides of every rule; this avoids accidental writer/reader
    collisions with other genome rules. Internal token IDs remain in the
    genome's existing 512-token coordinate system (no extra embedding pass).
    """
    occupied = set(map(int, genome.embedding))
    retired = set(map(int, excluding_rule_rows))
    for index, rule in enumerate(genome.rules):
        if index in retired:
            continue
        occupied.update(int(t) for t in rule.pattern if 0 <= int(t) < 512)
        occupied.update(int(t) for t in rule.replacement if 0 <= int(t) < 512)
    return tuple(t for t in range(511, -1, -1) if t not in occupied)[:amount]


def mine_paths(rolling, champion, features, baseline, target,
               *, max_proposals: int = 32, max_sources: int = 28,
               max_gap: int = 20, max_depth: int = 3,
               crossfit: bool = False) -> list[PathProposal]:
    """Find bridges on TRAINING sources only (final source is held out).

    The source selection uses current OLS residuals; no test chunk, held-out
    original, or held-out label enters mining. The source and target markers
    represent *joint* activation -- a single producer's proxy is not used as
    the optimization objective.
    """
    if max_depth not in (2, 3, 4):
        raise ValueError('BIPS depth must be 2, 3, or 4')
    if crossfit and len(rolling.slots) >= 3:
        # Two-stage leave-one-source-out proposal discovery.  The candidate's
        # literal anchors are mined and ranked using ONE clean source; CEGAR
        # may check them on a different source. This prevents construction of
        # special-case rules tailored using validation-chunk bytes/targets.
        # The population-level final original (rolling.slots[-1]) remains
        # completely unobserved during both proposal mining and refinement.
        from dataclasses import replace
        from types import SimpleNamespace
        ntrain = len(rolling.slots) - 1
        offsets = [0]
        for case in rolling.slots:
            offsets.append(offsets[-1] + len(case.inputs))
        quota = max(1, int(max_proposals)//ntrain)
        result = []
        for ci in range(ntrain):
            lo,hi = offsets[ci],offsets[ci+1]
            local = mine_paths(SimpleNamespace(slots=[rolling.slots[ci]]),
                               champion, features[lo:hi],baseline[lo:hi],
                               target[lo:hi],max_proposals=quota,
                               max_sources=max(1,int(max_sources)//ntrain),
                               max_gap=max_gap,max_depth=max_depth,
                               crossfit=False)
            result.extend(replace(p,origin_case=ci) for p in local)
        # Interleave rather than letting one source dominate. The native
        # verifier decides whether each path transfers to another source.
        lanes = [[p for p in result if p.origin_case == ci] for ci in range(ntrain)]
        output = []
        for i in range(max((len(lane) for lane in lanes),default=0)):
            for lane in lanes:
                if i < len(lane):
                    output.append(lane[i])
        return output[:max(0,int(max_proposals))]
    ntrain = len(rolling.slots) - 1 if len(rolling.slots) >= 2 else len(rolling.slots)
    if not ntrain or len(champion.rules) < max_depth:
        return []
    rows = [bytes(s) for case in rolling.slots[:ntrain] for s in case.inputs]
    if len(rows) < 4:
        return []
    train_count = len(rows)
    y = np.asarray(target, dtype=np.float64)[:train_count]
    w = np.asarray(champion.readout_weights, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)[:train_count]
    b = np.asarray(baseline, dtype=np.float64)[:train_count]
    if w.shape != (x.shape[1],):
        w = np.zeros(x.shape[1], dtype=np.float64)
    residual = y - (b + x @ w)
    residual -= float(residual.mean())
    if not np.all(np.isfinite(residual)) or float(residual @ residual) < 1e-14:
        return []

    evidence = []
    offset = 0
    for case in rolling.slots[:ntrain]:
        original = bytes(case.inputs[int(np.argmax(case.target))])
        for s in case.inputs:
            s = bytes(s)
            if len(s) == len(original):
                differences = [k for k, (a, d) in enumerate(zip(original, s)) if a != d]
                if differences:
                    evidence.append((abs(float(residual[offset])), offset, s, differences))
            offset += 1
    evidence.sort(key=lambda e: (-e[0], e[1]))
    evidence = evidence[:max(1, int(max_sources))]

    # Joint forward/backward anchor grammar: source, 0..2 bridges, sink.
    # Intermediate markers will preserve ALL skipped bytes with -1 captures.
    # The ordered requirement is crucial: independently-occurring n-grams
    # are NOT sufficient evidence for a feasible information path.
    keys = Counter()
    gaps = (0, 1, 2, 4, 8, 16, 24)
    for _, _, s, differences in evidence:
        for at in differences[:6]:
            for width in (1, 2, 3, 4):
                start = max(0, at - width + 1)
                left = s[start:at+1]
                if not left:
                    continue
                for gap in gaps:
                    if gap > max_gap:
                        continue
                    right_at = at + 1 + gap
                    for right_width in (1, 2):
                        right = s[right_at:right_at+right_width]
                        if len(right) != right_width:
                            continue
                        keys[(left, (), right)] += 1
                        if max_depth >= 3 and gap >= 1:
                            middle_at = at + 1 + gap // 2
                            middle = s[middle_at:middle_at+1]
                            if middle and middle_at+len(middle) <= right_at:
                                keys[(left, (middle,), right)] += 1
                        if max_depth >= 4 and gap >= 2:
                            a = at + 1 + (gap-1)//3
                            bpos = at + 1 + (2*gap)//3
                            if a < bpos < right_at:
                                bridge_a = s[a:a+1]
                                bridge_b = s[bpos:bpos+1]
                                if bridge_a and bridge_b:
                                    keys[(left, (bridge_a,bridge_b), right)] += 1
    if not keys:
        return []

    scored = []
    candidate_cap = max(64, 12*int(max_proposals))
    for depth in range(2, max_depth+1):
        family = [(k,v) for k,v in keys.items() if len(k[1])+2 == depth]
        family.sort(key=lambda kv: (-kv[1],kv[0]))
        for (left, middles, right), _support in family[:candidate_cap]:
            parts = (left,) + tuple(middles) + (right,)
            z = np.fromiter((_ordered_occurrence(s, parts) for s in rows),
                            dtype=np.float64, count=train_count)
            hits = float(z.sum())
            if hits < 1 or hits >= train_count:
                continue
            z -= hits/train_count
            var = float(z@z)
            align = float(z@residual)
            gain = align*align / max(var,1e-12)
            gain /= 1. + 0.06*sum(map(len,parts)) + 0.15*len(middles)
            if gain > 1e-14 and math.isfinite(gain):
                scored.append(PathProposal(gain,left,right,
                                           middles[0] if middles else b'',
                                           tuple(middles[1:])))
    scored.sort(key=lambda p: (-p.score,p.depth,p.anchors))
    count = max(0,int(max_proposals))
    if not count:
        return []
    # Fair depth quotas, then backfill if the source cannot support long paths.
    quota = {2:count}
    if max_depth >= 3 and count>=2:
        quota = {2:count-count//3,3:count//3}
    if max_depth >= 4 and count>=3:
        quota = {2:count-count//3-count//4, 3:count//3, 4:count//4}
    chosen = []
    used = set()
    for depth, cap in quota.items():
        for p in (p for p in scored if p.depth==depth):
            if cap<=0:
                break
            chosen.append(p);used.add(p);cap-=1
    if len(chosen)<count:
        chosen.extend(p for p in scored if p not in used and len(chosen)<count)
    chosen.sort(key=lambda p:-p.score)
    return chosen[:count]


def compile_path(mod, genome, proposal: PathProposal, markers: Sequence[int]):
    """Compile witness -> private marker -> backward requirement, in order.

    Ordinary -1 is used ONLY as a capture-and-preserve channel; -2/-3 hidden
    capture side effects and -128..-151 opcodes never appear in new rules.
    Distances are arbitrary: -1 bridges an intervening string of any length.
    """
    if len(markers) < proposal.depth or len(set(markers[:proposal.depth])) != proposal.depth:
        raise ValueError('not enough distinct latent marker tokens')
    e = genome.embedding
    def encoded(seq: bytes) -> list[int]:
        return [int(e[b]) for b in seq]
    anchors = ([list(map(int,a)) for a in proposal.internal_anchors]
               if proposal.internal_anchors else [encoded(a) for a in proposal.anchors])
    if len(anchors) != proposal.depth or any(not a for a in anchors):
        raise ValueError('invalid internal path anchors')
    p0 = anchors[0]
    if len(p0) < 1 or not proposal.suffix:
        raise ValueError('invalid path anchors')
    rules = [mod.Rule(p0, [int(markers[0])])]
    for index, right in enumerate(anchors[1:]):
        rules.append(mod.Rule([int(markers[index]), -1] + right,
                              [int(markers[index + 1]), -1]))
    if proposal.goal_kind == "consumer":
        pat = tuple(map(int, proposal.consumer_pattern))
        if not pat or any(t < 0 or t >= 512 for t in pat):
            raise ValueError("invalid consumer endpoint")
        rules[-1] = mod.Rule(rules[-1].pattern, list(pat) + [-1])
    elif proposal.goal_kind != "marker":
        raise ValueError("unsupported BIPS endpoint")
    if not all(mod.is_rule_nonexpanding(r) for r in rules):
        raise ValueError('BIPS generated an expanding rule')
    return rules


def _candidate_windows(genome, features, depth: int, limit: int = 6):
    """Replace low-utility ADJACENT windows so path ordering is guaranteed."""
    n = len(genome.rules)
    if n < depth:
        return []
    f = np.mean(np.asarray(features) != 0, axis=0)
    if len(f) != n:
        f = np.zeros(n)
    weights = np.asarray(genome.readout_weights, dtype=np.float64)
    if len(weights) != n:
        weights = np.zeros(n)
    utility = np.abs(weights) * (f + 1e-6)
    # Prefer later windows to avoid unnecessary transformations before the
    # producer, but allow the rest if all tail slots are important.
    windows = [(float(utility[i:i+depth].sum()) + 0.00001 * (n - i), i)
               for i in range(n-depth+1)]
    return [i for _, i in sorted(windows)[:max(1, int(limit))]]


def make_path_offspring(mod, champion, proposals, features, count, max_depth=3):
    """Whole-path simultaneous editing; no one-rule gate or individual fitness.

    Caller evaluates the complete child through the usual exact CPU/MPS path
    in the next generation, preserving Pareto and inner inference GA.
    """
    if count <= 0 or not proposals:
        return []
    windows_by_depth = {d: _candidate_windows(champion, features, d)
                        for d in (2, 3, 4) if d <= max_depth}
    result = []
    seen = set()
    # Interleave path lengths, then rotate target windows; otherwise six
    # windows of the highest-ranked 2-rule proposal would fill the whole
    # budget before any 3-rule path had a chance to be evaluated.
    lanes = {d: [p for p in proposals if p.depth == d]
             for d in (2,3,4) if d<=max_depth}
    ordered = []
    for i in range(max(map(len, lanes.values()), default=0)):
        for d in lanes:
            if i < len(lanes[d]):
                ordered.append(lanes[d][i])
    feasible = [p for p in ordered if p.depth <= max_depth]
    for window_rank in range(max(map(len, windows_by_depth.values()), default=0)):
        for proposal in feasible:
            windows = windows_by_depth.get(proposal.depth, ())
            if window_rank >= len(windows):
                continue
            at = windows[window_rank]
            # Reclaim private codes from rules that this child retires.  This
            # matters after long training runs where most latent codes have
            # been mentioned by at least one rule in the parent.
            markers = available_latent_tokens(
                champion, proposal.depth,
                excluding_rule_rows=range(at, at + proposal.depth))
            if len(markers) < proposal.depth:
                continue
            try:
                rules = compile_path(mod, champion, proposal, markers)
            except ValueError:
                continue
            key = (at, proposal.anchors)
            if key in seen:
                continue
            seen.add(key)
            child = mod.clone_genome_shallow(champion)
            for i, rule in enumerate(rules):
                child.rules[at + i] = rule
            mod._mark_genome_pack_rows(child, range(at, at + proposal.depth))
            mod._clear_evaluation_state(child)
            mod.mark_offspring_origin(child, champion, 'bips:d%d' % proposal.depth,
                                      proposal.depth)
            result.append(child)
            if len(result) >= count:
                return result
    return result
