"""Bounded execution-derived rule dependencies for Replacer.

Two DISTINCT signals:
  * directional token last-writer -> later rewriter (observed data flow);
  * phi coefficient from binary effective-rewrite events at sampled time steps.
Phi is not a causal measure. Provenance is a conservative *last-writer*
approximation, not a proof that a rule is necessary for a prediction.

All instrumentation is ephemeral, explicitly budgeted, and never serialized into
HOF/checkpoints. Sampling only records compact bitmasks rather than dense matrices.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(slots=True)
class DependencyGraph:
    observations: int
    errors: int
    fired_masks: list[int]
    error_mask: int
    producer_errors: Counter
    directed: Counter
    overflows: int

    def support(self, rule: int) -> int:
        return self.fired_masks[rule].bit_count()

    def cofires(self, a: int, b: int) -> int:
        return (self.fired_masks[a] & self.fired_masks[b]).bit_count()

    def phi(self, a: int, b: int) -> float:
        return phi_binary(self.fired_masks[a], self.fired_masks[b], self.observations)

    def phi_error(self, a: int) -> float:
        return phi_binary(self.fired_masks[a], self.error_mask, self.observations)


def phi_binary(a: int, b: int, n: int) -> float:
    """Pearson/phi coefficient of two binary vectors in lowest n bits.

    Returns zero when either variable is constant (undefined phi). Bits outside
    the declared sample window are ignored to avoid accidental contamination.
    """
    if n < 2:
        return 0.0
    mask = (1 << n) - 1
    a, b = a & mask, b & mask
    na, nb = a.bit_count(), b.bit_count()
    if not (0 < na < n and 0 < nb < n):
        return 0.0
    n11 = (a & b).bit_count()
    numerator = n * n11 - na * nb
    denom = math.sqrt(na * (n-na) * nb * (n-nb))
    return max(-1.0, min(1.0, numerator / denom))


def _rewritten_state_and_writers(lens, original, writers, updated, rule,
                                 rule_index, lut, inv):
    """Mirror replace_once match boundaries, attributing *changed matches*.

    An identical replacement keeps provenance; unchanged ranges retain their
    original last-writer. Changed output bytes are attributed to rule_index.
    Directed producer -> consumer edges are recorded only when some match
    actually changes bytes, not merely because its pattern matched.
    """
    pat = rule.pattern
    size = len(original)
    scan = prev = 0
    out = []
    provenance = []
    parents = set()
    while scan < size:
        found = None
        if pat[0] >= 0:
            for s in range(scan, size):
                if original[s] == pat[0]:
                    hit = lens._match_at(original, pat, s)
                    if hit is not None:
                        found = (s, *hit)
                        break
        else:
            hit = lens._match_at(original, pat, scan)
            if hit is not None:
                found = (scan, *hit)
        if found is None:
            break
        start, finish, captures = found
        out.extend(original[prev:start])
        provenance.extend(writers[prev:start])
        replacement = lens._emit_replacement(rule.replacement, captures, lut, inv)
        if replacement == original[start:finish]:
            provenance.extend(writers[start:finish])
        else:
            provenance.extend([rule_index] * len(replacement))
            parents.update(p for p in writers[start:finish] if p >= 0 and p != rule_index)
        out.extend(replacement)
        prev = scan = finish
    out.extend(original[prev:])
    provenance.extend(writers[prev:])
    if out != updated or len(provenance) != len(updated):
        raise AssertionError('Dependency instrumentation diverged from replace_once')
    return provenance, parents


def _sample_positions(prefix: int, maximum: int) -> set[int]:
    if prefix < 2 or maximum < 1:
        return set()
    stop = prefix - 1
    if maximum >= stop:
        return set(range(1, stop + 1))
    if maximum == 1:
        return {stop}
    return {1 + round(i * (stop-1)/(maximum-1)) for i in range(maximum)}


def collect_dependencies(lens, genome, examples, *, prefix=96,
                         steps_per_text=24, max_texts=3, state_cap=4096):
    """Instrument teacher-forced recurrent sweeps with bounded memory.

    Provenance is carried between timesteps; forced target bytes and fresh
    prediction slots have no rule writer. Only requested prediction steps are
    recorded in bitmasks. The full intermediate state is never persisted.
    """
    rules = genome.rules
    lut = genome.embedding
    inv = lens.inverse_lut(lut)
    anchors = lens._anchors(rules)
    firing = [0] * len(rules)
    bad_mask = 0
    errors = Counter()
    directional = Counter()
    samples = 0
    overflows = 0
    for ti, text in enumerate(examples[:max_texts]):
        limit = min(prefix, len(text))
        selected = _sample_positions(limit, steps_per_text)
        if not selected:
            continue
        state = [int(text[0]), lens.PREDICTION_SLOT]
        writers = [-1, -1]
        for step in range(1, limit):
            if len(state) > state_cap:
                overflows += 1
                break
            fired = set()
            edges = set()
            present = set(state)
            pairs = set(zip(state, state[1:]))
            failed = False
            for ri, (rule, (kind, anchor)) in enumerate(zip(rules, anchors)):
                if ((kind == 1 and anchor not in present) or
                    (kind == 2 and anchor not in pairs)):
                    continue
                updated = lens.replace_once(state, rule, lut, inv)
                if len(updated) > state_cap:
                    overflows += 1
                    failed = True
                    break
                if updated != state:
                    new_writers, upstream = _rewritten_state_and_writers(
                        lens, state, writers, updated, rule, ri, lut, inv)
                    writers = new_writers
                    state = updated
                    fired.add(ri)
                    edges.update((u, ri) for u in upstream)
                    present = set(state)
                    pairs = set(zip(state, state[1:]))
            if failed:
                break
            if not state:
                state = [lens.PREDICTION_SLOT]
                writers = [-1]
            pred = int(state[-1])
            is_wrong = pred != int(text[step])
            if step in selected:
                bit = 1 << samples
                for ri in fired:
                    firing[ri] |= bit
                for edge in edges:
                    directional[edge] += 1
                if is_wrong:
                    bad_mask |= bit
                    if writers[-1] >= 0:
                        errors[writers[-1]] += 1
                samples += 1
            # Teacher forcing resets the origin of the target byte. Future
            # steps cannot spuriously ascribe it to the prediction's producer.
            state[-1] = int(text[step])
            writers[-1] = -1
            state.append(lens.PREDICTION_SLOT)
            writers.append(-1)
    return DependencyGraph(samples, bad_mask.bit_count(), firing, bad_mask,
                           errors, directional, overflows)


def ranked_rules(graph: DependencyGraph, *, minimum_support=2):
    """Rank observed error producers and their upstream producers.

    Confidence from phi is used ONLY as a tie/priority modifier; a positive
    cofire by itself never creates a dataflow edge. Sparse edge supports are
    weighted and propagated at most three hops with decaying influence.
    """
    if not graph.observations or not graph.errors:
        return []
    scores = Counter({r: float(n) for r, n in graph.producer_errors.items()})
    inbound = {}
    totals = Counter()
    for (u, v), n in graph.directed.items():
        if n <= 0:
            continue
        inbound.setdefault(v, []).append((u, n))
        totals[v] += n
    wave = dict(scores)
    for _ in range(3):
        next_wave = Counter()
        for downstream, downstream_score in wave.items():
            for upstream, n in inbound.get(downstream, []):
                # Observed dataflow is stronger than correlation. Negative
                # phi must not eliminate real across-step provenance edges.
                phi = graph.phi(upstream, downstream)
                contribution = (0.60 * downstream_score * n / totals[downstream]
                                * (0.75 + 0.25 * max(0.0, phi)))
                next_wave[upstream] += contribution
        scores.update(next_wave)
        wave = next_wave
    for ri, mask in enumerate(graph.fired_masks):
        fired_count = mask.bit_count()
        if fired_count >= minimum_support:
            pe = graph.phi_error(ri)
            if pe > 0:
                scores[ri] += pe * math.sqrt(fired_count) * 0.2
    return sorted(((ri, score) for ri, score in scores.items() if score > 0),
                  key=lambda x: (-x[1], x[0]))


def choose_rule(graph, generation: int, total_rules: int, *, tail=16,
                topk=12, explore_every=5, minimum_support=2):
    """Alternate graph-directed edits and guaranteed tail exploration."""
    if total_rules < 1:
        raise ValueError('Cannot choose from empty rule network')
    fallback = total_rules - 1 - (generation % min(tail, total_rules))
    ranked = ranked_rules(graph, minimum_support=minimum_support)
    if not ranked or (explore_every > 0 and generation % explore_every == 0):
        return fallback, 'explore', 0.0
    selection = ranked[generation % min(topk, len(ranked))]
    return selection[0], 'dependency', selection[1]


def export_dependencies(graph, prefix, generation, *, max_edges=2048):
    """Two bounded CSV diagnostics; no per-token traces or genomes are saved."""
    path = Path(prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = sorted(graph.directed.items(), key=lambda kv: (-kv[1], kv[0]))
    with open(str(path) + '_edges.csv', 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['generation', 'source_rule', 'target_rule', 'dataflow_steps',
                    'phi_effective_fire', 'cofire_steps', 'source_fires',
                    'target_fires', 'observations'])
        for (a, b), n in edges[:max_edges]:
            w.writerow([generation, a, b, n, round(graph.phi(a,b),6),
                        graph.cofires(a,b), graph.support(a), graph.support(b),
                        graph.observations])
    ranked = dict(ranked_rules(graph))
    with open(str(path) + '_rules.csv', 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['generation', 'rule', 'effective_fire_steps',
                    'phi_fire_error', 'direct_prediction_errors',
                    'dependency_priority', 'observations', 'errors'])
        for i in range(len(graph.fired_masks)):
            w.writerow([generation, i, graph.support(i),
                        round(graph.phi_error(i), 6),
                        graph.producer_errors.get(i, 0),
                        round(ranked.get(i, 0), 6),
                        graph.observations, graph.errors])
