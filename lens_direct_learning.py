"""Residual-guided supervised rule mining, ablation diagnostics and optional non-GA baseline.

No access to evaluation clean strings before candidates are selected.  Rule mining
uses the *training* rolling corruption set only.  All rule edits are checked by
the ordinary Replacer evaluator on the next generation, not accepted on proxy.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import math
import random
import time
from typing import Sequence
import numpy as np


def _proxy_trace(pattern: tuple[int, ...], inputs: list[bytes]) -> np.ndarray:
    # A necessary-condition proxy for literal rule firing.  This deliberately
    # does not pretend to be the true sequential execution trace.
    word = bytes(pattern)
    return np.fromiter((int(word in s) for s in inputs), dtype=np.float64, count=len(inputs))


def mine_repair_rules(rolling, champion, feats: np.ndarray, baseline: np.ndarray,
                      target: np.ndarray, sampler, m, limit: int = 64,
                      source_cap: int = 48) -> list[tuple[float, object]]:
    """Build equal-length, literal corrective rules from *training* examples.

    The score combines least-squares residual covariance, observed clean/noisy
    contrast and a complexity penalty.  Not an exact expected change in loss:
    sequential rule interference must be measured by running Replacer.
    """
    source = []
    # Match OLS source separation.  Validation chunk is NEVER used to mine
    # corrective rules, compute residual covariances, or tune candidates.
    train_case_count = len(rolling.slots) - 1 if len(rolling.slots) >= 2 else len(rolling.slots)
    for ci, case in enumerate(rolling.slots[:train_case_count]):
        # The first row is always uncorrupted for dual diffusion.  In legacy
        # mode, prefer the maximum target rather than assuming ordering.
        clean_id = int(np.argmax(case.target))
        clean = case.inputs[clean_id]
        for row, y in zip(case.inputs, case.target):
            if y >= 1.0 - 1e-9 or len(row) != len(clean):
                continue
            source.append((clean, row))
        # We don't inspect held-out inference cases.
    candidates: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
    random.shuffle(source)
    for clean, damaged in source[:max(1, source_cap)]:
        differing = [i for i, (a, b) in enumerate(zip(clean, damaged)) if a != b]
        if not differing:
            continue
        for pos in random.sample(differing, min(12, len(differing))):
            for left, right in ((1, 1), (2, 1), (1, 2), (0, 2), (2, 0)):
                begin = max(0, pos - left)
                end = min(len(clean), pos + right + 1)
                a = tuple(map(int, damaged[begin:end]))
                b = list(a)
                b[pos - begin] = int(clean[pos])
                if len(a) < 2 or a == tuple(b):
                    continue
                candidates[(a, tuple(b))] += 1
    if not candidates:
        return []
    train_mask = np.concatenate([
        np.ones(len(c.inputs), dtype=bool) if ci < train_case_count else np.zeros(len(c.inputs), dtype=bool)
        for ci, c in enumerate(rolling.slots)
    ])
    raw_inputs = [bytes(x) for case in rolling.slots[:train_case_count] for x in case.inputs]
    residual = (target - (baseline + feats @ np.asarray(champion.readout_weights, dtype=np.float64)))[train_mask]
    # Properly centered residual: all candidate scores depend on evidence of
    # unexplained error, not the positive bias of the continuous target.
    residual -= np.mean(residual)
    residual_norm = float(np.dot(residual, residual) + 1e-9)
    scored = []
    for (a, b), count in candidates.items():
        z = _proxy_trace(a, raw_inputs)
        if not (1 < z.sum() < len(z) - 1):
            continue
        z -= np.mean(z)
        variance = float(z @ z)
        if variance < 1e-6:
            continue
        relevance = float(z @ residual)
        gain = (relevance ** 2) / (variance + 1e-6)
        # Supervised denoising contrast: wrong pattern should be much more
        # common in corrupted rows than in their clean sources.  Penalize
        # false positives on uncorrupted text (explicit negative examples).
        trigger = bytes(a)
        clean_hits = sum(trigger in bytes(case.inputs[int(np.argmax(case.target))])
                         for case in rolling.slots[:train_case_count])
        noisy_hits = sum(trigger in bytes(damaged) for _clean, damaged in source)
        false_positive_penalty = 1.0 + 3.0 * clean_hits
        contrast = (1.0 + math.log1p(noisy_hits)) / false_positive_penalty
        complexity = 1.0 + 0.025 * len(a)
        quality = (gain / complexity) * contrast * (1.0 + min(4, count) * 0.03)
        scored.append((quality, a, b))
    scored.sort(reverse=True)
    result = []
    for gain, a, b in scored[:max(1, limit)]:
        # Map raw bytes into the genome's token coordinates.
        pattern = [int(champion.embedding[int(i)]) for i in a]
        replacement = [int(champion.embedding[int(i)]) for i in b]
        r = m.Rule(pattern, replacement)
        # Enforce the same no-growth invariant as legacy mutation.
        m.sanitize_rule(r, sampler, champion.embedding)
        if r.pattern != r.replacement and m.is_rule_nonexpanding(r):
            result.append((float(gain), r))
    return result


def phi_dependencies(feats: np.ndarray, max_rules: int = 32):
    """Return high absolute phi coefficients among frequently firing rule IDs."""
    if feats.ndim != 2 or feats.shape[0] < 3:
        return []
    firing = np.asarray(feats != 0, dtype=np.float64)
    freq = firing.mean(axis=0)
    elig = np.flatnonzero((freq > 0.01) & (freq < 0.99))
    if elig.size < 2:
        return []
    if elig.size > max_rules:
        elig = elig[np.argsort(-freq[elig])[:max_rules]]
    x = firing[:, elig] - freq[elig][None, :]
    std = np.sqrt((x * x).sum(axis=0))
    corr = (x.T @ x) / np.maximum(1e-12, std[:, None] * std[None, :])
    pairs = []
    for i in range(len(elig)):
        for j in range(i + 1, len(elig)):
            pairs.append((float(corr[i, j]), int(elig[i]), int(elig[j])))
    return sorted(pairs, key=lambda v: -abs(v[0]))[:16]


def choose_replacement_ids(champion, feats, max_count: int) -> list[int]:
    weights = np.asarray(champion.readout_weights, dtype=np.float64)
    frequency = np.mean(np.asarray(feats != 0, dtype=np.float64), axis=0)
    # Prefer genuinely unused slots before retiring low-impact active rules.
    utility = (np.abs(weights) * (frequency + 1e-6))
    order = np.argsort(utility, kind='stable')
    return [int(i) for i in order[:max(1, max_count)]]


def make_directed_offspring(m, champion, proposals, replace_ids, count: int):
    """Coordinate-descent / beam candidates, never crossover or random mutation.

    A single replace is a single coordinate trial; actual fitness is computed
    by the standard outer native evaluation in the following generation.
    """
    result = []
    if not proposals or not replace_ids:
        return result
    for i in range(count):
        candidate = m.clone_genome_shallow(champion)
        row = int(replace_ids[i % len(replace_ids)])
        rule = proposals[(i // len(replace_ids)) % len(proposals)][1]
        # Fresh Rule object avoids mutating an archival parent's cached pack.
        candidate.rules[row] = m.clone_rule(rule)
        m._mark_genome_pack_rows(candidate, (row,))
        m._clear_evaluation_state(candidate)
        candidate._origin_operator = 'residual-guided'
        result.append(candidate)
    return result


def counterfactual_ablation(m, champion, inputs, target, case_ids, sample_ids,
                             args, feats, limit=3, priority_ids=None):
    """Delete one structural rule and fit a new readout before scoring.

    Full trajectories are replayed (correctly accounting for hidden memory and
    downstream effects); not a misleading feature-column-only ablation.
    """
    if limit <= 0 or len(champion.rules) < 2:
        return []
    weights = np.abs(np.asarray(champion.readout_weights, dtype=np.float64))
    frequency = np.mean(feats != 0, axis=0)
    priority = weights * (frequency + 1e-7)
    indices = (list(map(int, priority_ids))[:min(limit,len(priority))]
               if priority_ids is not None
               else np.argsort(-priority)[:min(limit,len(priority))])
    out = []
    for ri in indices:
        candidate = m.clone_genome_shallow(champion)
        candidate.rules.pop(int(ri))
        m._invalidate_genome_pack(candidate)
        m._clear_evaluation_state(candidate)
        m.evaluate_genome(candidate, inputs, target, case_ids, sample_ids,
                          args.backend, args.ridge_lambda, args.max_output,
                          args.readout_mode, args.rank_max_features,
                          args.hidden_rewrite_phases, args.rank_pair_mode)
        out.append((int(ri), float(candidate.fitness),
                    float(champion.fitness - candidate.fitness)))
    return out


def _candidate_bytes(seq, pos, sampler, width):
    left = int(seq[pos - 1]) if pos else None
    right = int(seq[pos + 1]) if pos + 1 < len(seq) else None
    prior = sampler._context_pairs
    if left is not None and right is not None:
        scores = prior[left, :] * prior[:, right]
    elif left is not None:
        scores = prior[left, :]
    elif right is not None:
        scores = prior[:, right]
    else:
        scores = sampler.byte_prob
    k = min(256, max(1, int(width)))
    ids = np.argsort(-scores, kind='stable')[:k]
    return [int(v) for v in ids if int(v) != int(seq[pos])]


def run_direct_inference(m, genomes, inference_set, sampler, backend, max_output,
                         hidden_rewrite_phases=2, mps_evaluator=None,
                         restore_inputs=None, case_count=None,
                         choices_per_site=4, ensemble_size=6, max_loci=6):
    """OPTIONAL coordinate-wise non-genetic denoising baseline, not the default.

    Proposals are formed entirely from the public corpus bigram prior.  Search
    uses an ensemble of fitted scoring functions; the clean target is examined
    *after* selecting the candidate.  All genomes get an unbiased held-apart
    reconstruction accuracy, rather than fitting to known inference targets.
    """
    t0 = time.perf_counter()
    cases = inference_set.slots[:max(1, int(case_count or len(inference_set.slots)))]
    top_ids = np.argsort([-g.fitness for g in genomes], kind='stable')[:max(1, min(len(genomes), int(ensemble_size)))]
    search_models = [genomes[int(i)] for i in top_ids]
    found = []
    total_work = 0
    for case in cases:
        current = list(case.noisy)
        # Locus updates are deterministic; beam width=1, no mutation/selection.
        for pos in list(case.mutable_positions)[:max_loci]:
            options = [current]
            for ch in _candidate_bytes(current, pos, sampler, choices_per_site):
                row = current.copy()
                row[pos] = ch
                options.append(row)
            scores = m.evaluate_inference_candidates(
                search_models, options, backend, max_output,
                mps_evaluator=mps_evaluator, hidden_rewrite_phases=hidden_rewrite_phases,
            )
            # Different OLS genomes have different output scales; vote by rank.
            normalized = m._row_rank01(scores)
            aggregate = np.mean(normalized, axis=0)
            best = int(np.argmax(aggregate))
            current = list(options[best])
            total_work += len(search_models) * len(options)
        found.append(current)
    # Score every individual on the same outcome, keeping Pareto semantics.
    # A model-independent decoded candidate is not a per-model optimization:
    # all models receive the same held-out accuracy (diagnostic/Pareto tie).
    # To preserve model-specific objective signal, score each model's own
    # selection from a finite, clean-blind candidate pool.
    per_model = np.zeros(len(genomes), dtype=np.float64)
    per_model_full = np.zeros(len(genomes), dtype=np.float64)
    ensemble_acc = []
    for case, decoded in zip(cases, found):
        # Scoring candidates NEVER includes the ground-truth clean string unless
        # the model independently proposed it.
        pool = [list(case.noisy), decoded]
        scores = m.evaluate_inference_candidates(genomes, pool, backend, max_output,
                                                mps_evaluator=mps_evaluator,
                                                hidden_rewrite_phases=hidden_rewrite_phases)
        for gi in range(len(genomes)):
            pick = int(np.argmax(scores[gi]))
            per_model[gi] += m._inference_recovery_accuracy(pool[pick], case)
            per_model_full[gi] += m._inference_full_accuracy(pool[pick], case)
        ensemble_acc.append(m._inference_recovery_accuracy(decoded, case))
        total_work += len(genomes) * len(pool)
    if cases:
        per_model /= len(cases)
        per_model_full /= len(cases)
    for g, a, b in zip(genomes, per_model, per_model_full):
        g.inference_accuracy = float(a)
        g.inference_full_accuracy = float(b)
    return {
        'seconds': float(time.perf_counter() - t0),
        'evaluations': int(total_work), 'ensemble_accuracy': float(np.mean(ensemble_acc)) if ensemble_acc else 0.,
        'oracle_accuracy': float('nan'), 'selected_change_rate': float('nan'),
        'mean_accuracy': float(np.mean(per_model)), 'best_accuracy': float(np.max(per_model)),
        'model_jobs': int(total_work), 'kernel_jobs': int(total_work),
        'search_rounds': 1, 'equivalent_work': total_work / max(1, len(genomes)),
    }
