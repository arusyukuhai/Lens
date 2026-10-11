"""Trace-guided supervised rule synthesis for Lens / Replacer.

Single-incumbent optimization: synthesize rules from observed teacher-forced
errors, rank with exact counterfactual suffix replay, then verify by a native
whole-sequence rollout. This intentionally does NOT call GA breeding/mutation.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import statistics
import time

import rule_dependencies as dep


@dataclass(frozen=True, slots=True)
class Snapshot:
    before: bytes
    target: int
    baseline_prediction: int
    text_index: int
    step: int


def _sample_positions(prefix: int, maximum: int) -> set[int]:
    """1-based prediction steps; evenly cover short prefixes (no duplicates)."""
    if prefix < 2 or maximum < 1:
        return set()
    stop = prefix - 1
    if maximum >= stop:
        return set(range(1, stop + 1))
    if maximum == 1:
        return {stop}
    return {1 + round(i * (stop - 1) / (maximum - 1)) for i in range(maximum)}


def collect_snapshots(lens, genome, examples, rule_index, *, prefix=96,
                      steps_per_text=16, state_cap=4096, max_texts=3):
    """Record exact pre-rule recurrent states, using reference sweep semantics.

    Capture at *one* chosen rule only and no more than `steps_per_text` per text.
    Abort an oversized Python trace rather than allocating unbounded history.
    The full native scoring step makes final acceptance decisions.
    """
    lut = genome.embedding
    inv = lens.inverse_lut(lut)
    rules = genome.rules
    anchors = lens._anchors(rules)
    result = []
    aborted = 0
    for ti, text in enumerate(examples[:max_texts]):
        limit = min(prefix, len(text))
        selected = _sample_positions(limit, steps_per_text)
        if not selected:
            continue
        state = [int(text[0]), lens.PREDICTION_SLOT]
        for step in range(1, limit):
            if len(state) > state_cap:
                aborted += 1
                break
            before = None
            present = set(state)
            pairs = set(zip(state, state[1:]))
            for ri, (rule, (kind, anchor)) in enumerate(zip(rules, anchors)):
                if ri == rule_index and step in selected:
                    before = bytes(state)
                if kind == 1 and anchor not in present or kind == 2 and anchor not in pairs:
                    continue
                # Trace collection never runs an oversized state through Python.
                updated = lens.replace_once(state, rule, lut, inv)
                if len(updated) > state_cap:
                    aborted += 1
                    break
                if updated != state:
                    state = updated
                    present = set(state)
                    pairs = set(zip(state, state[1:]))
            else:
                if not state:
                    state = [lens.PREDICTION_SLOT]
                prediction = int(state[-1])
                if before is not None:
                    result.append(Snapshot(before, int(text[step]), prediction, ti, step))
                state[-1] = int(text[step])
                state.append(lens.PREDICTION_SLOT)
                continue
            # Some rule exceeded the tracing budget: do not claim its output
            # correct, and do not carry forward a truncated or invalid state.
            break
    return result, aborted


def propose_rules(lens, snapshots, embedding, *, context=6, max_candidates=96):
    """Construct replacements from observed mispredictions, never random byte edits.

    Patterns are exact suffixes, or literal anchors + capture + prediction slot.
    The latter retains a potentially long intermediate substring and may create
    context-generalized rules. Arithmetic capture transforms are proposed only
    if the observed one-byte capture actually maps to the desired target.
    """
    counts = collections.Counter()
    lut = embedding
    inverse = lens.inverse_lut(lut)
    for s in snapshots:
        if s.baseline_prediction == s.target or not s.before:
            continue
        row = list(s.before)
        goal = s.target
        maximum = min(max(1, context), len(row))
        for k in range(1, maximum + 1):
            suffix = tuple(row[-k:])
            # Replace the *predicted* byte while preserving matched context.
            counts[(suffix, suffix[:-1] + (goal,))] += 1
        if len(row) >= 5:
            # Two captured spans: a richer structural hypothesis than an n-gram.
            # The pair of captures is copied once, followed by the desired
            # prediction. No substring of the recurrent state is discarded.
            a, b, last = row[-5], row[-3], row[-1]
            counts[((a, -1, b, -1, last),
                    (a, -1, b, -2, goal))] += 1
        if len(row) >= 3:
            last = row[-1]
            for anchor_width in (1, 2):
                if len(row) < anchor_width + 2:
                    continue
                anchors = tuple(row[-anchor_width-2:-2])
                pat = anchors + (-1, last)
                rep = anchors + (-1, goal)
                counts[(pat, rep)] += 1
            # A capture transform is a genuinely symbolic hypothesis, not
            # memorization of the observed output. It may not generalize; the
            # exact local/full evaluator is responsible for rejecting it.
            previous = row[-2]
            for op in (-48, -64, -80, -96):
                value = lens._emit_replacement([op], [[previous]], lut, inverse)
                if value == [goal]:
                    anchor = row[-3]
                    counts[((anchor, -1, last), (anchor, -1, op))] += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0][0]), kv[0]))
    proposals = []
    for (p, q), support in ranked:
        rule = lens.Rule(list(p), list(q))
        if lens.is_valid_rule(rule):
            proposals.append((rule, support))
            if len(proposals) >= max_candidates:
                break
    return proposals



def bounded_sweep(lens, state, rules, lut, inv, anchors, capacity):
    """Reference-compatible rule sweep with a per-intermediate-state budget."""
    state = list(state)
    if len(state) > capacity:
        return None
    present = set(state)
    pairs = set(zip(state, state[1:]))
    for rule, (kind, anchor) in zip(rules, anchors):
        if kind == 1 and anchor not in present or kind == 2 and anchor not in pairs:
            continue
        new_state = lens.replace_once(state, rule, lut, inv)
        if len(new_state) > capacity:
            return None
        if new_state != state:
            state = new_state
            present = set(state)
            pairs = set(zip(state, state[1:]))
    return state

def local_rank(lens, genome, row_index, snapshots, proposals, *, max_full=4,
               state_cap=4096):
    """Exact single-step, suffix-only counterfactual; no approximate reward."""
    lut = genome.embedding
    inv = lens.inverse_lut(lut)
    after = genome.rules[row_index+1:]
    anchors = lens._anchors(after)
    baseline = sum(s.baseline_prediction == s.target for s in snapshots)
    candidates = []
    for rule, support in proposals:
        fixed = broken = affected = overflow = 0
        for sample in snapshots:
            state = list(sample.before)
            after_rule = lens.replace_once(state, rule, lut, inv)
            original = lens.replace_once(state, genome.rules[row_index], lut, inv)
            if after_rule == original:
                # Same output as the incumbent rule: later rules are identical.
                continue
            affected += 1
            if len(after_rule) > state_cap:
                overflow += 1
                continue
            # Guard *every intermediate state*, not only the final result.
            nxt = bounded_sweep(lens, after_rule, after, lut, inv, anchors,
                                state_cap)
            if nxt is None:
                overflow += 1
                continue
            new_pred = int(nxt[-1]) if nxt else lens.PREDICTION_SLOT
            before_ok = sample.baseline_prediction == sample.target
            after_ok = new_pred == sample.target
            fixed += int(not before_ok and after_ok)
            broken += int(before_ok and not after_ok)
        gain = fixed - broken
        if gain > 0 and overflow == 0:
            candidates.append((gain, fixed, -broken, support, -len(rule.pattern), rule))
    candidates.sort(key=lambda x: x[:5], reverse=True)
    return [(item[-1], {'local_gain': item[0], 'local_fixed': item[1],
                        'local_broken': -item[2], 'support': item[3]})
            for item in candidates[:max_full]], baseline


def improve_once(lens, incumbent, train, generation, args, *, selected_row=None):
    """One strictly data-guided rule edit, checked against complete real rollout."""
    count = len(incumbent.rules)
    tail = min(args.trace_tail, count)
    row_index = (count - 1 - (generation % tail)
                 if selected_row is None else selected_row)
    if not 0 <= row_index < count:
        raise ValueError('selected trace rule out of range')
    snaps, aborted = collect_snapshots(lens, incumbent, train, row_index,
                                       prefix=args.trace_prefix,
                                       steps_per_text=args.trace_steps,
                                       state_cap=args.trace_state_cap,
                                       max_texts=args.trace_texts)
    stats = {'trace_row': row_index, 'trace_samples': len(snaps),
             'trace_errors': sum(s.baseline_prediction != s.target for s in snaps),
             'trace_overflows': aborted, 'trace_proposals': 0,
             'trace_local': 0, 'trace_full': 0, 'trace_accepted': 0,
             'trace_gain': 0.0}
    if not snaps or stats['trace_errors'] == 0:
        return incumbent, stats
    # Pass the actual embedding to opcode proposals, without changing model.
    # propose_rules accepts the embedding explicitly (no global cache).
    proposals = propose_rules(lens, snaps, incumbent.embedding,
                              context=args.trace_context,
                              max_candidates=args.trace_candidates)
    stats['trace_proposals'] = len(proposals)
    shortlisted, baseline = local_rank(lens, incumbent, row_index, snaps, proposals,
                                        max_full=args.trace_confirm,
                                        state_cap=args.trace_state_cap)
    stats['trace_local'] = len(shortlisted)
    if not shortlisted:
        return incumbent, stats
    trial_genomes = []
    for rule, data in shortlisted:
        rules = incumbent.rules.copy()
        rules[row_index] = rule
        trial_genomes.append(lens.Genome(rules, incumbent.embedding.copy()))
    # Native CPU (or requested backend) evaluates ALL active, teacher-forced
    # case bytes; sampled/local fitness never grants acceptance.
    lens.score_population([incumbent] + trial_genomes, train, args.backend,
                          args.mps_batch, args.cpu_workers,
                          use_cache=not args.no_eval_cache)
    stats['trace_full'] = len(trial_genomes)
    best = max(trial_genomes, key=lambda g: g.fitness)
    if best.fitness > incumbent.fitness + 1e-12:
        stats['trace_accepted'] = 1
        stats['trace_gain'] = best.fitness - incumbent.fitness
        return best, stats
    return incumbent, stats


def evolve_trace(args, lens):
    """One-incumbent training loop; independent from legacy GA control flow."""
    if args.trace_tail < 1 or args.trace_prefix < 2 or args.trace_steps < 1:
        raise ValueError('trace-tail >= 1, trace-prefix >= 2, trace-steps >= 1 required')
    if args.trace_state_cap < 2 or args.trace_confirm < 1 or args.trace_texts < 1:
        raise ValueError('trace-state-cap >= 2, trace-confirm >= 1, trace-texts >= 1 required')
    if args.trace_candidates < 1 or args.trace_context < 1:
        raise ValueError('trace-candidates >= 1, trace-context >= 1 required')
    if getattr(args, 'trace_dependencies', False) and (
            args.trace_dep_every < 1 or args.trace_dep_topk < 1 or
            args.trace_dep_steps < 1 or args.trace_dep_texts < 1 or
            args.trace_dep_prefix < 2 or args.trace_dep_min_support < 1 or
            args.trace_dep_graph_every < 0 or args.trace_dep_explore_every < 0):
        raise ValueError('invalid --trace-dep-* settings')
    if args.max_state_bytes < 0 or args.hof_size < 0:
        raise ValueError('max-state-bytes and hof-size must be nonnegative')
    os.environ['LENS_MAX_STATE_BYTES'] = str(args.max_state_bytes)
    corpus = lens.load_corpus(args.local_corpus, args.min_chunk, args.corpus_chunks,
                              args.max_chunk)
    if args.load:
        first, population, best_ever, history, rng, archive, rotation_state = \
            lens.load_checkpoint(args.load, include_rotation_state=True)
        if not population:
            raise ValueError('loaded checkpoint has an empty population')
        # Trace checkpoints store the actual incumbent in population[0].
        # Legacy GA checkpoints instead bootstrap from best_ever.
        if len(population) == 1 and history and history[-1].get('optimizer') == 'trace':
            source = population[0]
        else:
            source = best_ever
        incumbent = lens.Genome(source.rules.copy(), source.embedding.copy())
        args.rules = len(incumbent.rules)
    else:
        first = 0
        rng = random.Random(args.seed)
        bigrams = lens._bigram_seeds(corpus)
        if not bigrams:
            raise ValueError('corpus requires at least one adjacent byte pair')
        incumbent = lens.new_genome(args.rules, rng, corpus, bigrams, not args.no_embedding)
        best_ever = None
        history, archive, rotation_state = [], [], None
    rolling = lens.RollingTrainingCases(corpus, args.cases, args.seed * 1000003,
                                        args.case_rotate_every, rotation_state)
    if rotation_state is None and first:
        rolling.for_generation(first-1)
    if len(archive) > args.hof_size:
        archive = sorted(archive, key=lambda g: g.fitness, reverse=True)[:args.hof_size]
    archive_keys = {lens.fingerprint(g) for g in archive}
    dependency_graph = None
    dependency_generation = -1
    dependency_rotation = -1
    if history and args.history_csv:
        # Union all old/trace columns once: avoid malformed CSV on GA->trace resumes.
        for row in history:
            for key in ('optimizer', 'trace_row', 'trace_samples', 'trace_errors',
                        'trace_overflows', 'trace_proposals', 'trace_local',
                        'trace_full', 'trace_accepted', 'trace_gain',
                        'trace_dep_row_method', 'trace_dep_score',
                        'trace_dep_samples', 'trace_dep_errors',
                        'trace_dep_edges', 'trace_dep_active',
                        'trace_dep_overflows', 'trace_dep_phi_error'):
                row.setdefault(key, '')
        lens.write_history(args.history_csv, history)
    elif args.history_csv and Path(args.history_csv).exists():
        # Fresh run in reused directory: don't append old training history.
        Path(args.history_csv).unlink()
    for gen in range(first, args.generations):
        t0 = time.perf_counter()
        train = rolling.for_generation(gen)
        lens.score_population([incumbent], train, args.backend, args.mps_batch,
                              args.cpu_workers, use_cache=not args.no_eval_cache)
        # The graph is a sampled *diagnostic*, not a learned parameter.
        # Cache only while genome and rotating training cases remain identical.
        dep_info = {'trace_dep_row_method': '', 'trace_dep_score': 0.0,
                    'trace_dep_samples': 0, 'trace_dep_errors': 0,
                    'trace_dep_edges': 0, 'trace_dep_active': 0,
                    'trace_dep_overflows': 0, 'trace_dep_phi_error': 0.0}
        chosen = None
        if getattr(args, 'trace_dependencies', False):
            if (dependency_graph is None or
                dependency_rotation != rolling.rotations_done or
                gen - dependency_generation >= args.trace_dep_every):
                dependency_graph = dep.collect_dependencies(
                    lens, incumbent, train,
                    prefix=args.trace_dep_prefix,
                    steps_per_text=args.trace_dep_steps,
                    max_texts=args.trace_dep_texts,
                    state_cap=args.trace_state_cap)
                dependency_generation = gen
                dependency_rotation = rolling.rotations_done
            chosen, method, dep_score = dep.choose_rule(
                dependency_graph, gen, len(incumbent.rules),
                tail=args.trace_tail, topk=args.trace_dep_topk,
                explore_every=args.trace_dep_explore_every,
                minimum_support=args.trace_dep_min_support)
            dep_info.update(trace_dep_row_method=method,
                trace_dep_score=dep_score,
                trace_dep_samples=dependency_graph.observations,
                trace_dep_errors=dependency_graph.errors,
                trace_dep_edges=len(dependency_graph.directed),
                trace_dep_active=sum(bool(x) for x in dependency_graph.fired_masks),
                trace_dep_overflows=dependency_graph.overflows,
                trace_dep_phi_error=dependency_graph.phi_error(chosen))
            if (args.trace_dep_graph_prefix and args.trace_dep_graph_every and
                gen % args.trace_dep_graph_every == 0):
                dep.export_dependencies(dependency_graph,
                                        args.trace_dep_graph_prefix, gen)
        incumbent, trace_info = improve_once(
            lens, incumbent, train, gen, args, selected_row=chosen)
        trace_info.update(dep_info)
        if trace_info['trace_accepted']:
            # Invalidated: even one changed rule can alter the entire graph.
            dependency_graph = None
        if best_ever is None or incumbent.fitness > best_ever.fitness:
            best_ever = lens.Genome(incumbent.rules.copy(), incumbent.embedding.copy(),
                                    incumbent.fitness, incumbent.case_scores.copy())
        # The JSON always reflects the latest *accepted incumbent*, even if a
        # previous rotation had a numerically higher (incomparable) accuracy.
        if gen == first or trace_info['trace_accepted']:
            lens.save_model(args.save, incumbent)
        sig = lens.fingerprint(incumbent)
        if args.hof_size > 0 and sig not in archive_keys:
            snapshot = lens.Genome(incumbent.rules.copy(), incumbent.embedding.copy(),
                                   incumbent.fitness, incumbent.case_scores.copy())
            archive.append(snapshot)
            archive_keys.add(sig)
            archive.sort(key=lambda g: g.fitness, reverse=True)
            if len(archive) > args.hof_size:
                removed = archive.pop()
                archive_keys.discard(lens.fingerprint(removed))
        elapsed = time.perf_counter() - t0
        total = sum(len(x) - 1 for x in train)
        row = {'generation': gen, 'best_accuracy': incumbent.fitness,
               'mean_accuracy': incumbent.fitness, 'median_accuracy': incumbent.fitness,
               'best_ever_accuracy': best_ever.fitness,
               'correct_bytes': round(incumbent.fitness * total),
               'evaluated_bytes': total, 'eval_seconds': elapsed,
               'rules': args.rules, 'tokens': 256, 'backend': args.backend,
               'case_rotation': rolling.rotations_done,
               'rotated_case': ((rolling.rotations_done-1) % args.cases if gen
                                and gen % args.case_rotate_every == 0 else -1),
               'optimizer': 'trace', **trace_info}
        history.append(row)
        if args.history_csv:
            lens.append_history_row(args.history_csv, row)
        print(f'gen={gen:6d} TRACE acc={incumbent.fitness:.6f} '
              f'gain={trace_info["trace_gain"]:+.6f} '
              f'error={trace_info["trace_errors"]}/{trace_info["trace_samples"]} '
              f'row={trace_info["trace_row"]} proposals={trace_info["trace_proposals"]} '
              f'local={trace_info["trace_local"]} full={trace_info["trace_full"]} '
              f'accepted={trace_info["trace_accepted"]} '
              f'overflow={trace_info["trace_overflows"]} '
              f'dep={trace_info["trace_dep_row_method"]} '
              f'edges={trace_info["trace_dep_edges"]} '
              f'seconds={elapsed:.2f} hof={len(archive)}/{args.hof_size}', flush=True)
        if not args.no_plot and (gen + 1) % max(1, args.plot_every) == 0:
            lens.plot_history(history, args.plot_prefix, args.plot_window)
        if args.checkpoint and args.checkpoint_every > 0 and \
                (gen + 1) % args.checkpoint_every == 0:
            lens.save_checkpoint(args.checkpoint, gen+1, [incumbent], best_ever,
                                 history, rng, archive, rolling)
    if best_ever is None:
        raise ValueError('No generations evaluated')
    if args.checkpoint:
        lens.save_checkpoint(args.checkpoint, args.generations, [incumbent],
                             best_ever, history, rng, archive, rolling)
    if not args.no_plot:
        lens.plot_history(history, args.plot_prefix, args.plot_window)
    return best_ever
