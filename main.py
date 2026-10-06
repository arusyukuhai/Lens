#!/usr/bin/env python3
"""
Replacement-rule genetic programming core (v60.3 inference-Pareto, differential git-merge crossover).

v60.3 adds a three-parent categorical differential/git-style crossover: for base A, donor X, and target Y, non-conflicting A->X rule-sequence edits are applied to Y. It probabilistically shares the existing crossover budget with legacy two-point crossover.

v60.2 changes the string-side mutation radius to round(exp(uniform(0, log(k))))
for k known-corrupted loci. The scheduled focus locus is guaranteed to be included
without collapsing the radius to one. v60 adds a bounded string-side genetic inference phase. A small shared candidate
population edits only synthetically corrupted byte positions and is selected by
the current Replace ensemble. Per-genome final-pool recovery accuracy becomes a
second NSGA-II-style Pareto objective beside Spearman. The checked revision uses
ensemble-only inner search + one full-population final scoring pass, reuses MPS
rule/index packs across inner generations, fixes tie handling/GC roots/corpus edge
cases, and keeps a conservative inference-work cap well below the requested ~2x
wall-time envelope. v57/v59 ranking and Metal rewrite kernels are retained.
CPU path can run without PyTorch. Metal rewrite kernels are unchanged.

Original engine lineage (v51 len1-word-prefix v50 len1-direct v45 match-bitset v38 literal-3gram-bloom v55 rule-pack-memo v56 recode-pack-memo v33 dual-anchor-bloom v31 narrow-state wc1-inplace fused-hash v30 wc1-general-fast v28 inplace-literal anchor-guided score-reuse direct-diff variable-length GPU-index incremental-candidate rolling-HoF rolling-evaluation bounded-workload compacting-GC precise-mutation/nonexpanding cooperative MPS), ported from
at_jev(20261002-082442).nim.

v19 keeps the v18 Nim-style mutation geometry (local/balanced/explore row budgets,
mutually exclusive structural mutation classes, guided weak-row targeting and
log-uniform segment edits) and adds a hard non-expanding rewrite invariant.
Every rule is repaired so replacement literals <= pattern literals and every
wildcard capture is emitted at most once. Therefore each rewrite satisfies
output_length <= consumed_length for every possible capture length, eliminating
evolutionary state-length blow-up by construction rather than by a fitness penalty.

Kept on purpose:
  * codeparrot/github-code (streaming) OR local github-code.txt chunks
  * cumulative noisy trajectories built from clean code
  * ordered rewrite rules with wildcard captures / transforms
  * repeated sweeps until fixed point, cycle, overflow, or ceil(2*sqrt(LEN))
  * per-rule firing counts as linear-regression features
  * sparse pairwise rank-ridge readout fitted on all trajectories
  * in-sample per-case Spearman correlation against cleanliness (= 1 - noise rate)
  * elite selection, tournament selection, two-point crossover, mutation,
    random immigrants
  * per-genome learned 256-byte -> 512-token injective embedding
  * CSV/PNG training plots (fitness + embedding evolution)

Deliberately removed:
  FAST/FULL, rolling timescales, race/rejection, lexicase,
  full mutation bandits/probes, Jev/fusion, genealogy, Pareto runtime pressure,
  checkpoint migration and most diagnostics.

GPU path:
  MPS maps one whole trajectory to one cooperative Metal threadgroup. Threads
  jointly build the exact candidate bitmap, search match starts, emit replacements,
  compare outputs, and hash states; lane 0 only controls ordered rule execution.

The GPU helper file `gpu_replace_persistent.py` must be beside this script.
Raw noisy bytes are mapped through each genome's evolved injective embedding before rewrite evaluation.

v19 uses a compacting Global Rule Pool: immutable pattern/replacement structures
are interned across generations, but unreachable structures are periodically traced
from the live population and removed. The MPS backing buffers are genuinely shrunk
at GC instead of merely resetting logical ids. A lightweight
trajectory profiler reports rounds/candidates/index rebuilds/scan work/rewrites.
The training moving average uses an expanding prefix and becomes a fixed-width
window once enough generations exist, so it is visible from generation 0.

Example (small smoke run):
  python minimal_replacer_gp.py --backend cpu --population 8 --rules 32 \
      --cases 2 --samples 12 --generations 3 --local-corpus github-code.txt

Mac MPS run:
  python minimal_replacer_gp.py --backend mps --population 450 --rules 1500 \
      --cases 8 --samples 20 --case-rotate-every 2 --hof-size 24 --generations 100000
"""

from __future__ import annotations

import argparse
import csv
import copy
import gc
import json
import math
import os
import random
import pickle
import statistics
from difflib import SequenceMatcher
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None  # The reference CPU trainer needs only NumPy.

try:
    if torch is not None:
        import gpu_replace_persistent as gpu_replace
    else:
        gpu_replace = None
except ImportError as e:
    raise SystemExit(
        "gpu_replace_persistent.py must be in the same directory / PYTHONPATH"
    ) from e


# ---------------------------------------------------------------------------
# Constants / rule representation
# ---------------------------------------------------------------------------

VOCAB = 512
MAX_WILDCARDS = 16
MAX_RULE_TOKENS = 64
MAX_OUTPUT = 32768
RIDGE_LAMBDA = 4.0
TRAIN_MOD = 3  # legacy-holdout only
PAIRWISE_RANK_MAX_FEATURES = 256
INPUT_BYTE_COUNT = 256
EMBEDDING_ENTRY_COUNT = 256
EMBEDDING_OOV = VOCAB
EMBEDDING_MUTATION_RATE = 0.045
EMBEDDING_CROSSOVER_RATE = 0.12
LOCAL_MUTATION_MAX_RULES = 8
BALANCED_MUTATION_MAX_RULES = 64


# Persistent fused MPS intentionally does not support sort($capture), so the
# minimal GP does not generate -16..-31.  All other transform families used by
# the current Nim core remain available.
TRANSFORM_FAMILIES = ("capture", "reverse", "plus1", "minus1", "times2", "div2")


@dataclass(slots=True)
class Rule:
    pattern: List[int]
    replacement: List[int]
    weight: float = 0.0
    # Performance-only cache.  Any literal/pair chosen from the pattern is a
    # necessary match condition, so keeping the previously selected anchor
    # across generations is semantically exact even when input frequencies
    # change.  -1 means "not computed yet".
    anchor_kind: int = -1
    anchor_a: int = -1
    anchor_b: int = -1
    # Additional host-pack caches. Signs/opcode families survive embedding
    # recoding, so these values stay valid until the rule itself is mutated.
    pack_wc: int = -1
    pack_leading: int = -1
    pack_rep_literal: int = -1
    # Evaluator-local structural pool id; -1 means not interned yet.
    pool_id: int = -1
    pool_owner: int = 0
    # v55: rule-local pack memo. Any chosen secondary pair/triple remains
    # a necessary condition for this unchanged pattern, so it can be reused
    # across crossover and later input rotations until the rule is mutated.
    # -2 = not computed, -1 = valid "no filter".
    pack_filter_key: int = -2
    # Structural position of the primary anchor in pattern.
    # -2 = not computed, -1 = no usable anchor.
    pack_anchor_offset: int = -2


@dataclass(slots=True)
class Genome:
    rules: List[Rule]
    embedding: List[int] = field(default_factory=lambda: list(range(EMBEDDING_ENTRY_COUNT)))
    fitness: float = float("-inf")
    case_scores: List[float] = field(default_factory=list)
    # Ridge readout is genome-local. Structural Rule objects are intentionally
    # shared copy-on-write between offspring; storing fitted weights on Rule
    # would make one genome's evaluation overwrite its siblings/parents.
    readout_weights: List[float] = field(default_factory=list)
    # v60: second objective. Accuracy is measured only on the bytes deliberately
    # corrupted for the string-side inference task, so unchanged context cannot
    # inflate the score. Pareto rank/crowding are runtime selection metadata.
    inference_accuracy: float = float("nan")
    inference_full_accuracy: float = float("nan")
    pareto_rank: int = 2_147_483_647
    pareto_crowding: float = 0.0
    # v29: evaluator-local immutable host-pack cache. Offspring inherit these
    # arrays by reference and mark only structurally touched rows dirty. This
    # avoids rescanning ~1500 Python Rule objects for every new child each gen.
    _eval_case_serials: tuple = ()
    # v58 runtime-only provenance for delayed operator credit assignment.
    # The normal next-generation evaluation supplies the reward; no auxiliary
    # model evaluations are required.
    _origin_operator: str = ""
    _parent_case_scores: tuple = ()
    _parent_case_serials: tuple = ()
    _mix_rows: int = 0
    _pack_owner: int = 0
    _pack_rule_ids: object | None = None
    _pack_anchors: object | None = None
    _pack_dirty_all: bool = True
    _pack_dirty_rows: set[int] = field(default_factory=set)

    def as_program(self) -> List[Tuple[List[int], List[int]]]:
        return [(r.pattern, r.replacement) for r in self.rules]


def clone_rule(rule: Rule) -> Rule:
    """Cheap explicit Rule clone; faster than copy.deepcopy for this flat object."""
    return Rule(list(rule.pattern), list(rule.replacement), float(rule.weight), int(rule.anchor_kind), int(rule.anchor_a), int(rule.anchor_b), int(rule.pack_wc), int(rule.pack_leading), int(rule.pack_rep_literal), int(rule.pool_id), int(rule.pool_owner), int(rule.pack_filter_key), int(rule.pack_anchor_offset))


def _inherit_genome_pack_cache(src: Genome, dst: Genome) -> Genome:
    """Share immutable packed arrays; child mutations copy only dirty rows."""
    dst._pack_owner = int(src._pack_owner)
    dst._pack_rule_ids = src._pack_rule_ids
    dst._pack_anchors = src._pack_anchors
    dst._pack_dirty_all = bool(src._pack_dirty_all)
    dst._pack_dirty_rows = set(src._pack_dirty_rows)
    return dst


def _mark_genome_pack_rows(g: Genome, rows) -> None:
    if g._pack_dirty_all:
        return
    g._pack_dirty_rows.update(int(i) for i in rows)


def _invalidate_genome_pack(g: Genome) -> None:
    g._pack_dirty_all = True
    g._pack_dirty_rows.clear()


def clone_genome_shallow(g: Genome) -> Genome:
    """Copy genome metadata/list containers while sharing immutable Rule objects.

    Children use copy-on-write in mutate_genome/crossover/apply_embedding_change,
    so sharing untouched rules is safe and removes O(rule_count) deep copies from
    every offspring. v29 also shares immutable evaluator pack arrays.
    """
    out = Genome(
        list(g.rules),
        embedding=list(g.embedding),
        fitness=float(g.fitness),
        case_scores=list(g.case_scores),
        readout_weights=list(g.readout_weights),
        inference_accuracy=float(g.inference_accuracy),
        inference_full_accuracy=float(g.inference_full_accuracy),
        pareto_rank=int(g.pareto_rank),
        pareto_crowding=float(g.pareto_crowding),
        _eval_case_serials=g._eval_case_serials,
    )
    return _inherit_genome_pack_cache(g, out)


def clone_genome_deep(g: Genome) -> Genome:
    """Fully independent clone used for the persistent Hall-of-Fame.

    Normal offspring intentionally share untouched Rule objects for speed. The v25
    readout itself is genome-local, but a long-lived HoF still owns Rule objects so
    later structural cache mutations/recoding cannot leak across archive lineages.
    """
    out = Genome(
        [clone_rule(r) for r in g.rules],
        embedding=list(g.embedding),
        fitness=float(g.fitness),
        case_scores=list(g.case_scores),
        readout_weights=list(g.readout_weights),
        inference_accuracy=float(g.inference_accuracy),
        inference_full_accuracy=float(g.inference_full_accuracy),
        pareto_rank=int(g.pareto_rank),
        pareto_crowding=float(g.pareto_crowding),
        _eval_case_serials=g._eval_case_serials,
    )
    return _inherit_genome_pack_cache(g, out)


@dataclass(slots=True)
class HallOfFameEntry:
    genome: Genome
    score_cache: dict[int, float] = field(default_factory=dict)
    current_fitness: float = float("-inf")
    historical_fitness: float = float("-inf")
    archive_score: float = float("-inf")
    born_generation: int = -1


def genome_structural_distance(a: Genome, b: Genome) -> float:
    """Cheap deterministic structural distance in [0, 1], ignoring readout weights.

    90% of the distance comes from rule rows at corresponding coordinates and 10%
    from the learned byte embedding.  This is deliberately position-sensitive: the
    program is an ordered rewrite system, so moving an otherwise identical rule is
    a meaningful structural change.
    """
    n = min(len(a.rules), len(b.rules))
    if n <= 0:
        rule_distance = 1.0 if len(a.rules) != len(b.rules) else 0.0
    else:
        same = 0
        for ra, rb in zip(a.rules[:n], b.rules[:n]):
            if ra.pattern == rb.pattern and ra.replacement == rb.replacement:
                same += 1
        length_penalty = abs(len(a.rules) - len(b.rules)) / max(1, max(len(a.rules), len(b.rules)))
        rule_distance = min(1.0, 1.0 - same / float(n) + length_penalty)
    m = min(len(a.embedding), len(b.embedding))
    if m:
        emb_distance = sum(int(x != y) for x, y in zip(a.embedding[:m], b.embedding[:m])) / float(m)
    else:
        emb_distance = 0.0
    return min(1.0, 0.90 * rule_distance + 0.10 * emb_distance)




def same_genome_structure(a: Genome, b: Genome) -> bool:
    if a.embedding != b.embedding or len(a.rules) != len(b.rules):
        return False
    return all(
        ra.pattern == rb.pattern and ra.replacement == rb.replacement
        for ra, rb in zip(a.rules, b.rules)
    )


def genome_eval_signature(g: Genome, pool_owner: int) -> tuple:
    """Fast current-process structural signature for exact eval deduplication.

    Already-interned rules use their evaluator-local pool id, so deep HoF clones
    and shallow survivor clones collapse without hashing rule token lists again.
    Freshly mutated rows fall back to a structural hash. Exact equality is still
    checked before reusing a representative, so hash collisions are harmless.
    """
    ids = []
    for r in g.rules:
        if r.pool_owner == pool_owner and r.pool_id >= 0:
            ids.append(int(r.pool_id) << 1)
        else:
            ids.append((hash((tuple(r.pattern), tuple(r.replacement))) << 1) | 1)
    return (tuple(g.embedding), tuple(ids))


def refresh_hof_entry(
    entry: HallOfFameEntry,
    case_serials: Sequence[int],
    history_cases: int,
    current_weight: float,
) -> None:
    """Update rolling current/historical HoF scores after evaluating current cases."""
    entry.current_fitness = float(entry.genome.fitness)
    for serial, score in zip(case_serials, entry.genome.case_scores):
        if math.isfinite(float(score)):
            entry.score_cache[int(serial)] = float(score)
    keep = max(len(case_serials), int(history_cases))
    if keep > 0 and len(entry.score_cache) > keep:
        newest = sorted(entry.score_cache)[-keep:]
        entry.score_cache = {k: entry.score_cache[k] for k in newest}
    current_set = set(map(int, case_serials))
    old_scores = [v for k, v in entry.score_cache.items() if k not in current_set and math.isfinite(v)]
    entry.historical_fitness = (
        float(statistics.fmean(old_scores)) if old_scores else entry.current_fitness
    )
    w = min(1.0, max(0.0, float(current_weight)))
    entry.archive_score = w * entry.current_fitness + (1.0 - w) * entry.historical_fitness


def make_hof_entry(
    genome: Genome, case_serials: Sequence[int], generation: int,
    history_cases: int, current_weight: float,
) -> HallOfFameEntry:
    entry = HallOfFameEntry(clone_genome_deep(genome), born_generation=int(generation))
    refresh_hof_entry(entry, case_serials, history_cases, current_weight)
    return entry


def update_hall_of_fame(
    archive: List[HallOfFameEntry],
    candidates: Sequence[Genome],
    case_serials: Sequence[int],
    generation: int,
    capacity: int,
    history_cases: int,
    current_weight: float,
    min_distance: float,
) -> Tuple[int, int]:
    """Admit strong structurally distinct candidates into a rolling HoF.

    Near-duplicates compete directly with their nearest archived lineage; otherwise
    a candidate can open a new lineage.  Capacity trimming uses the rolling archive
    score, not a stale one-generation maximum.
    """
    capacity = max(0, int(capacity))
    if capacity == 0:
        archive.clear()
        return 0, 0
    admitted = 0
    replaced = 0
    threshold = max(0.0, float(min_distance))
    for genome in candidates:
        cand = make_hof_entry(genome, case_serials, generation, history_cases, current_weight)
        if not archive:
            archive.append(cand); admitted += 1; continue
        distances = [genome_structural_distance(cand.genome, e.genome) for e in archive]
        nearest = int(np.argmin(np.asarray(distances, dtype=np.float64)))
        if distances[nearest] < threshold:
            incumbent = archive[nearest]
            # For the same lineage, require a genuinely better current test score.
            # Historical cache belongs to the old structure and must not be copied.
            common = sorted(set(cand.score_cache) & set(incumbent.score_cache))
            if common and statistics.fmean(cand.score_cache[k] - incumbent.score_cache[k] for k in common) > 1.0e-12:
                archive[nearest] = cand
                replaced += 1
        else:
            archive.append(cand)
            admitted += 1
    archive.sort(key=lambda e: e.archive_score, reverse=True)
    if len(archive) > capacity:
        del archive[capacity:]
    return admitted, replaced


def tournament_hof(archive: Sequence[HallOfFameEntry], k: int) -> HallOfFameEntry:
    ids = random.sample(range(len(archive)), k=min(max(1, int(k)), len(archive)))
    return max((archive[i] for i in ids), key=lambda e: e.archive_score)


# ---------------------------------------------------------------------------
# Learned byte -> latent-token embedding (evolved with the genome)
# ---------------------------------------------------------------------------


def default_embedding() -> List[int]:
    return list(range(EMBEDDING_ENTRY_COUNT))


def valid_embedding(e: Sequence[int]) -> bool:
    return (
        len(e) == EMBEDDING_ENTRY_COUNT
        and all(0 <= int(v) < VOCAB for v in e)
        and len(set(map(int, e))) == EMBEDDING_ENTRY_COUNT
    )


def sample_unused_embedding_code(e: Sequence[int]) -> int:
    used = set(map(int, e))
    free = [v for v in range(VOCAB) if v not in used]
    if not free:
        raise ValueError("embedding has no unused internal token")
    return random.choice(free)


def initial_embedding(enabled: bool = True) -> List[int]:
    e = default_embedding()
    if not enabled:
        return e
    # Nim makeObj(): most newborns start near identity, while ~35% explore a
    # few latent coordinates immediately.
    if random.randrange(100) < 35:
        for _ in range(1 + random.randrange(4)):
            a = random.randrange(EMBEDDING_ENTRY_COUNT)
            if random.randrange(100) < 70:
                e[a] = sample_unused_embedding_code(e)
            else:
                b = random.randrange(EMBEDDING_ENTRY_COUNT)
                e[a], e[b] = e[b], e[a]
    assert valid_embedding(e)
    return e


def embedding_translation(from_map: Sequence[int], to_map: Sequence[int]) -> List[int]:
    """Extend 256 byte correspondences to a full 512-token permutation.

    This mirrors the Nim coordinate-change logic: mapped byte tokens follow the
    new dictionary; latent tokens preserve identity whenever that target remains
    free, and displaced latent tokens are paired with the remaining free targets.
    """
    if not valid_embedding(from_map) or not valid_embedding(to_map):
        raise ValueError("invalid embedding")
    tr = [-1] * VOCAB
    taken = [False] * VOCAB
    for b in range(EMBEDDING_ENTRY_COUNT):
        tr[int(from_map[b])] = int(to_map[b])
        taken[int(to_map[b])] = True
    for token in range(VOCAB):
        if tr[token] < 0 and not taken[token]:
            tr[token] = token
            taken[token] = True
    free = [token for token in range(VOCAB) if not taken[token]]
    it = iter(free)
    for token in range(VOCAB):
        if tr[token] < 0:
            tr[token] = next(it)
    assert sorted(tr) == list(range(VOCAB))
    return tr


def _translate_pack_filter_key(rule: Rule, translation: Sequence[int]) -> int:
    """Translate a cached exact Bloom guard into the new token coordinates.

    v38 chooses a triple guard for wildcard-free literal patterns of length >=3,
    otherwise a pair guard.  Coordinate recoding preserves contiguity and the
    necessary-condition property, so the cached guard can be transformed instead
    of rediscovered from corpus frequencies.
    """
    key = int(rule.pack_filter_key)
    if key < 0:
        return key  # -2=not computed, -1=valid no-filter
    literal_triple = len(rule.pattern) >= 3 and all(int(v) >= 0 for v in rule.pattern)
    if literal_triple:
        c = key % _GP_INDEX_TOKENS
        q = key // _GP_INDEX_TOKENS
        b = q % _GP_INDEX_TOKENS
        a = q // _GP_INDEX_TOKENS
        if not (0 <= a < VOCAB and 0 <= b < VOCAB and 0 <= c < VOCAB):
            return -2
        a2, b2, c2 = int(translation[a]), int(translation[b]), int(translation[c])
        return (a2 * _GP_INDEX_TOKENS + b2) * _GP_INDEX_TOKENS + c2
    b = key % _GP_INDEX_TOKENS
    a = key // _GP_INDEX_TOKENS
    if not (0 <= a < VOCAB and 0 <= b < VOCAB):
        return -2
    return int(translation[a]) * _GP_INDEX_TOKENS + int(translation[b])


def recode_rule(rule: Rule, translation: Sequence[int]) -> Rule:
    ak = int(rule.anchor_kind)
    aa = int(rule.anchor_a)
    ab = int(rule.anchor_b)
    if ak >= 1 and 0 <= aa < VOCAB:
        aa = int(translation[aa])
    if ak >= 2 and 0 <= ab < VOCAB:
        ab = int(translation[ab])
    return Rule(
        [translation[v] if 0 <= v < VOCAB else v for v in rule.pattern],
        [translation[v] if 0 <= v < VOCAB else v for v in rule.replacement],
        float(rule.weight), ak, aa, ab,
        int(rule.pack_wc), int(rule.pack_leading), int(rule.pack_rep_literal), -1, 0,
        _translate_pack_filter_key(rule, translation), int(rule.pack_anchor_offset),
    )


def apply_embedding_change(g: Genome, new_map: Sequence[int], preserve_rate: float = 0.90) -> None:
    if list(new_map) == g.embedding:
        return
    if not valid_embedding(g.embedding) or not valid_embedding(new_map):
        raise ValueError("invalid embedding change")
    tr = embedding_translation(g.embedding, new_map)
    # Copy-on-write: untouched rules may remain shared with parents/siblings.
    # Reinterpreted rules become fresh objects, so no shared parent rule is mutated.
    new_rules: List[Rule] = []
    for r in g.rules:
        if random.random() <= preserve_rate:
            rr = recode_rule(r, tr)
            rr.weight = 0.0
            new_rules.append(rr)
        else:
            # The rule is intentionally reinterpreted under the new embedding.
            # Its cached candidate anchor was chosen using the OLD embedding's
            # raw-byte frequencies.  Reusing it is semantically exact but can be
            # catastrophically bad for pruning after several generations.
            # Clone only these relatively rare reinterpreted rules and lazily
            # choose a fresh anchor on the next GPU pack.
            rr = clone_rule(r)
            rr.anchor_kind = -1
            rr.anchor_a = -1
            rr.anchor_b = -1
            new_rules.append(rr)
    g.rules = new_rules
    g.embedding = list(map(int, new_map))
    _invalidate_genome_pack(g)
    assert valid_embedding(g.embedding)


def mutate_embedding(g: Genome, rate: float = EMBEDDING_MUTATION_RATE) -> None:
    if random.random() >= rate:
        return
    new_map = list(g.embedding)
    changed = log_uniform_mutation_count(EMBEDDING_ENTRY_COUNT)
    for _ in range(changed):
        roll = random.randrange(100)
        if roll < 65:
            byte = random.randrange(EMBEDDING_ENTRY_COUNT)
            new_map[byte] = sample_unused_embedding_code(new_map)
        elif roll < 90:
            a, b = random.sample(range(EMBEDDING_ENTRY_COUNT), 2)
            new_map[a], new_map[b] = new_map[b], new_map[a]
        else:
            ln = random.randint(2, EMBEDDING_ENTRY_COUNT)
            start = random.randrange(0, EMBEDDING_ENTRY_COUNT - ln + 1)
            first = new_map[start]
            new_map[start:start + ln - 1] = new_map[start + 1:start + ln]
            new_map[start + ln - 1] = first
    apply_embedding_change(g, new_map, 0.90)


def crossover_embedding(child: Genome, donor: Genome, rate: float = EMBEDDING_CROSSOVER_RATE) -> None:
    if child.embedding == donor.embedding or random.random() >= rate:
        return
    new_map = list(child.embedding)
    owner = [-1] * VOCAB
    for byte, code in enumerate(new_map):
        owner[code] = byte
    k = log_uniform_mutation_count(EMBEDDING_ENTRY_COUNT)
    indices = list(range(EMBEDDING_ENTRY_COUNT))
    random.shuffle(indices)
    for byte in indices[:k]:
        target = donor.embedding[byte]
        previous = new_map[byte]
        if target == previous:
            continue
        other = owner[target]
        if other >= 0:
            new_map[other] = previous
            owner[previous] = other
        else:
            owner[previous] = -1
        new_map[byte] = target
        owner[target] = byte
    apply_embedding_change(child, new_map, 0.90)


def embed_inputs(inputs: Sequence[Sequence[int]], embedding: Sequence[int]) -> List[List[int]]:
    e = list(map(int, embedding))
    return [
        [e[v] if 0 <= int(v) < INPUT_BYTE_COUNT else EMBEDDING_OOV for v in row]
        for row in inputs
    ]


def embedding_metrics(e: Sequence[int]) -> Tuple[int, int]:
    moved = sum(int(v) != i for i, v in enumerate(e))
    latent = sum(int(v) >= INPUT_BYTE_COUNT for v in e)
    return moved, latent


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_local_corpus(path: str, limit: int, min_len: int, max_len: int) -> List[bytes]:
    """Read the same ===SPLIT=== format used by the Nim program.

    If no delimiter is present, non-empty lines are accepted as independent
    chunks so that a tiny hand-made corpus also works for smoke tests.
    """
    p = Path(path)
    if not p.exists():
        return []
    raw = p.read_bytes()
    chunks: List[bytes] = []
    marker = b"===SPLIT==="
    if marker in raw:
        parts = raw.split(marker)
    else:
        parts = raw.splitlines()
    for part in parts:
        part = part.strip(b"\r\n")
        if len(part) < min_len or len(part) > max_len:
            continue
        if max_len > 0 and len(part) > max_len:
            start = random.randrange(0, len(part) - max_len + 1)
            part = part[start : start + max_len]
        chunks.append(bytes(part))
        if len(chunks) >= limit:
            break
    return chunks


def stream_github_code(limit: int, min_len: int, max_len: int) -> List[bytes]:
    """Stream `codeparrot/github-code` through Hugging Face datasets.

    The dataset's public schema has a `code` field.  This is only used when a
    local github-code.txt is unavailable.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "No local corpus found and `datasets` is not installed. "
            "Run: pip install datasets"
        ) from e

    ds = load_dataset("codeparrot/github-code", streaming=True, split="train")
    chunks: List[bytes] = []
    for row in ds:
        text = row.get("code", "")
        if not isinstance(text, str):
            continue
        b = text.encode("utf-8", errors="ignore")
        if len(b) < min_len or len(b) > max_len:
            continue
        if max_len > 0 and len(b) > max_len:
            start = random.randrange(0, len(b) - max_len + 1)
            b = b[start : start + max_len]
        chunks.append(b)
        if len(chunks) >= limit:
            break
    return chunks


def effective_trajectory_span(args, corpus: Sequence[bytes]) -> int:
    """Maximum raw row length the evaluator must be able to hold.

    ``0`` means "unbounded" for --max-chunk / --trajectory-len.  The previous
    ``min(max_chunk, trajectory_len)`` turned an unbounded max_chunk=0 into a
    one-token MPS allocation, even when trajectory_len was positive.
    """
    max_chunk = int(args.max_chunk)
    trajectory_len = int(args.trajectory_len)
    if trajectory_len > 0:
        return max(1, min(max_chunk, trajectory_len) if max_chunk > 0 else trajectory_len)
    if max_chunk > 0:
        return max(1, max_chunk)
    return max(1, max((len(c) for c in corpus), default=1))


def load_corpus(args) -> List[bytes]:
    chunks = load_local_corpus(args.local_corpus, args.corpus_chunks, args.min_chunk, args.max_chunk)
    if chunks:
        print(f"corpus: loaded {len(chunks)} chunks from {args.local_corpus}")
        return chunks
    print("corpus: local file unavailable; streaming codeparrot/github-code")
    chunks = stream_github_code(args.corpus_chunks, args.min_chunk, args.max_chunk)
    if not chunks:
        raise RuntimeError("no source chunks found")
    return chunks


class CorpusSampler:
    """Corpus-grounded literals and short n-gram seeds.

    Important performance detail: the old implementation called
    ``np.random.choice(..., p=...)`` once per literal.  A default population
    contains roughly 675,000 rules, so that turns population construction into
    tens of seconds of tiny NumPy calls.  We instead draw bytes in large
    vectorized blocks and consume them from a cheap host buffer.
    """

    def __init__(self, chunks: Sequence[bytes], ngram_pool: int = 4096, byte_buffer_size: int = 1_048_576):
        hist = np.ones(256, dtype=np.float64)  # Laplace smoothing
        for c in chunks:
            if c:
                hist += np.bincount(np.frombuffer(c, dtype=np.uint8), minlength=256)
        self.byte_prob = hist / hist.sum()
        self.chunks = list(chunks)
        self.ngrams: List[List[int]] = []
        if self.chunks:
            for _ in range(ngram_pool):
                c = random.choice(self.chunks)
                if len(c) < 2:
                    continue
                n = random.randint(2, min(6, len(c)))
                start = random.randrange(0, len(c) - n + 1)
                self.ngrams.append(list(c[start : start + n]))

        # v60.1 inference proposal model.  The old string GA proposed replacement
        # bytes mostly from the global byte histogram.  With ~10 corrupted loci
        # and only a few dozen inner evaluations this made exact recovery far too
        # sparse: a correct byte often never entered the candidate pool at all.
        # Build a tiny 256x256 transition table from the already-sampled n-grams.
        # This costs ~256 KiB and no corpus rescan, while allowing mutation to ask
        # "which byte is plausible between these two neighbours?" without ever
        # consulting the hidden clean target.
        self._rebuild_context_model()

        self._byte_buffer_size = max(4096, int(byte_buffer_size))
        self._byte_buffer = np.empty(0, dtype=np.uint16)
        self._byte_pos = 0
        # Most genomes keep one embedding for many thousands of literal draws.
        # Cache the 256 unused latent coordinates per embedding instead of
        # rebuilding a set/list for every 3% exploratory literal.
        self._unused_embedding_cache: dict[tuple[int, ...], tuple[int, ...]] = {}

    def _rebuild_context_model(self) -> None:
        self._context_pairs = np.ones((INPUT_BYTE_COUNT, INPUT_BYTE_COUNT), dtype=np.float64) * 0.25
        for ng in self.ngrams:
            if len(ng) < 2:
                continue
            a = np.asarray(ng[:-1], dtype=np.int16)
            b = np.asarray(ng[1:], dtype=np.int16)
            np.add.at(self._context_pairs, (a, b), 1.0)
        self._context_row_sum = self._context_pairs.sum(axis=1)
        self._context_col_sum = self._context_pairs.sum(axis=0)

    def _refill_bytes(self) -> None:
        self._byte_buffer = np.random.choice(
            256, size=self._byte_buffer_size, p=self.byte_prob
        ).astype(np.uint16, copy=False)
        self._byte_pos = 0

    def byte(self) -> int:
        if self._byte_pos >= int(self._byte_buffer.size):
            self._refill_bytes()
        v = int(self._byte_buffer[self._byte_pos])
        self._byte_pos += 1
        return v

    def contextual_byte(self, sequence: Sequence[int], pos: int, avoid: int | None = None) -> int:
        """Sample a corpus-plausible raw byte from immediate left/right context.

        This is a proposal distribution only.  It never sees the clean inference
        target; the Replacer score still decides which candidate survives.
        """
        pos = int(pos)
        weights = self.byte_prob.astype(np.float64, copy=True)
        if pos > 0:
            left = int(sequence[pos - 1])
            if 0 <= left < INPUT_BYTE_COUNT:
                weights *= self._context_pairs[left, :] / max(1.0, float(self._context_row_sum[left]))
        if pos + 1 < len(sequence):
            right = int(sequence[pos + 1])
            if 0 <= right < INPUT_BYTE_COUNT:
                weights *= self._context_pairs[:, right] / max(1.0, float(self._context_col_sum[right]))
        if avoid is not None and 0 <= int(avoid) < INPUT_BYTE_COUNT:
            weights[int(avoid)] = 0.0
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 1.0e-300:
            weights = self.byte_prob.astype(np.float64, copy=True)
            if avoid is not None and 0 <= int(avoid) < INPUT_BYTE_COUNT:
                weights[int(avoid)] = 0.0
            total = float(weights.sum())
        if total <= 0.0:
            return (int(avoid or 0) + 1) & 255
        return int(np.random.choice(INPUT_BYTE_COUNT, p=weights / total))

    def _unused_codes(self, embedding: Sequence[int]) -> tuple[int, ...]:
        key = tuple(map(int, embedding))
        cached = self._unused_embedding_cache.get(key)
        if cached is not None:
            return cached
        used = set(key)
        free = tuple(v for v in range(VOCAB) if v not in used)
        # Prevent an unbounded cache after very long evolutionary runs.
        if len(self._unused_embedding_cache) >= 4096:
            self._unused_embedding_cache.clear()
        self._unused_embedding_cache[key] = free
        return free

    def literal(self, embedding: Sequence[int] | None = None) -> int:
        e = embedding if embedding is not None else default_embedding()
        if random.random() < 0.97:
            return int(e[self.byte()])
        free = self._unused_codes(e)
        return int(random.choice(free))

    def seeded_pattern(self, target_len: int, embedding: Sequence[int] | None = None) -> List[int]:
        e = embedding if embedding is not None else default_embedding()
        target_len = max(1, min(MAX_RULE_TOKENS, target_len))
        if self.ngrams and random.random() < 0.70:
            ng = random.choice(self.ngrams)
            if len(ng) >= target_len:
                start = random.randrange(0, len(ng) - target_len + 1)
                raw = ng[start : start + target_len]
            else:
                raw = list(ng)
            out = [int(e[b]) for b in raw]
            while len(out) < target_len:
                out.append(self.literal(e))
            return out
        return [self.literal(e) for _ in range(target_len)]


# ---------------------------------------------------------------------------
# Noise trajectories
# ---------------------------------------------------------------------------


def make_noise_trajectory(
    base: bytes,
    samples: int,
    sampler: CorpusSampler,
    max_noise: float,
) -> Tuple[List[List[int]], List[float], List[int]]:
    """Cumulative corruption path with an exact, monotone noise target.

    The Nim code uses a richer diffusion mixture, but the GP only needs a
    monotonically damaged trajectory and a target ordering.  For this minimal
    core we mutate distinct positions cumulatively.  `cleanliness=1-damage/n`
    has the same ordering as the Nim target `base.len - diffusionDamage`.
    """
    src = list(base)
    n = len(src)
    if n < 2:
        return [src], [1.0], [0]

    samples = max(3, samples)
    max_k = max(1, min(n - 1, int(round(max_noise * n))))
    # Distinct cumulative positions guarantee monotone Hamming damage.
    positions = list(range(n))
    random.shuffle(positions)
    positions = positions[:max_k]

    xs: List[List[int]] = []
    cleanliness: List[float] = []
    sample_ids: List[int] = []
    state = src.copy()
    prev_k = 0
    for s in range(samples):
        k = int(round(max_k * s / (samples - 1)))
        for pos in positions[prev_k:k]:
            old = state[pos]
            v = sampler.byte()
            guard = 0
            while v == old and guard < 16:
                v = sampler.byte()
                guard += 1
            if v == old:
                v = (old + 1) & 255
            state[pos] = v
        prev_k = k
        xs.append(state.copy())
        cleanliness.append(1.0 - (k / n))
        sample_ids.append(s)
    return xs, cleanliness, sample_ids


def build_dataset(
    corpus: Sequence[bytes],
    sampler: CorpusSampler,
    cases: int,
    samples: int,
    max_noise: float,
    trajectory_len: int = 0,
) -> Tuple[List[List[int]], np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened samples and per-row target/case/sample ids.

    v20 bounds one trajectory's work with a random contiguous code window.
    The replacement system is local (rules are at most MAX_RULE_TOKENS long),
    while evaluating 40 noisy copies of an arbitrary 15k-token source chunk
    creates enormous GPU stragglers and highly variable generation times.

    Sample from the full corpus length distribution. ``trajectory_len`` is an
    upper bound, not a target length: long chunks are cropped to a random window,
    while naturally short chunks stay short.  This preserves real short examples
    and reduces wasted GPU work without repeatedly using the beginning of files.
    Pass trajectory_len<=0 to retain full-chunk trajectories.
    """
    want = min(cases, len(corpus))
    tlen = max(0, int(trajectory_len))
    # v24: trajectory_len is a cap, not a minimum.  Sampling only chunks that are
    # >= tlen silently collapses the whole training distribution to one length.
    chosen = random.sample(list(corpus), k=want)
    if tlen > 0:
        windowed: List[bytes] = []
        for base in chosen:
            if len(base) <= tlen:
                windowed.append(base)
            else:
                start = random.randrange(0, len(base) - tlen + 1)
                windowed.append(base[start:start + tlen])
        chosen = windowed
    all_x: List[List[int]] = []
    all_y: List[float] = []
    all_case: List[int] = []
    all_sample: List[int] = []
    for ci, base in enumerate(chosen):
        xs, ys, sids = make_noise_trajectory(base, samples, sampler, max_noise)
        all_x.extend(xs)
        all_y.extend(ys)
        all_case.extend([ci] * len(xs))
        all_sample.extend(sids)
    return (
        all_x,
        np.asarray(all_y, dtype=np.float64),
        np.asarray(all_case, dtype=np.int32),
        np.asarray(all_sample, dtype=np.int32),
    )


@dataclass(slots=True)
class RollingEvalCase:
    inputs: List[List[int]]
    target: np.ndarray
    sample_ids: np.ndarray
    source_index: int
    window_start: int
    serial: int


class RollingEvaluationSet:
    """Fixed-size evaluation set with one case replaced at a time.

    v22 keeps most of the evaluation distribution stable while continuously
    exposing the population to fresh code. Case slots are stable identities for
    scoring; ``serial`` changes only when that slot is replaced, which lets the
    evolution loop compare stagnation on the exact overlap between generations.
    """

    def __init__(
        self,
        corpus: Sequence[bytes],
        sampler: CorpusSampler,
        cases: int,
        samples: int,
        max_noise: float,
        trajectory_len: int,
    ) -> None:
        if not corpus:
            raise ValueError("empty corpus")
        self.corpus = corpus
        self.sampler = sampler
        self.case_count = max(1, min(int(cases), len(corpus)))
        self.samples = max(3, int(samples))
        self.max_noise = float(max_noise)
        self.trajectory_len = max(0, int(trajectory_len))
        self._serial = 0
        self._flat_cache = None

        eligible = self._eligible_indices()
        if len(eligible) >= self.case_count:
            chosen = random.sample(eligible, self.case_count)
        else:
            chosen = random.sample(range(len(corpus)), self.case_count)
        self.slots: List[RollingEvalCase] = [self._make_case(i) for i in chosen]

    def _eligible_indices(self) -> List[int]:
        # v24: every corpus chunk is eligible. trajectory_len only caps long chunks;
        # short chunks remain short so the rolling set follows the corpus distribution.
        return list(range(len(self.corpus)))

    def _make_case(self, source_index: int) -> RollingEvalCase:
        base = self.corpus[int(source_index)]
        start = 0
        if self.trajectory_len > 0 and len(base) > self.trajectory_len:
            start = random.randrange(0, len(base) - self.trajectory_len + 1)
            base = base[start:start + self.trajectory_len]
        xs, ys, sids = make_noise_trajectory(base, self.samples, self.sampler, self.max_noise)
        serial = self._serial
        self._serial += 1
        return RollingEvalCase(
            inputs=xs,
            target=np.asarray(ys, dtype=np.float64),
            sample_ids=np.asarray(sids, dtype=np.int32),
            source_index=int(source_index),
            window_start=int(start),
            serial=int(serial),
        )

    def rotate(self, slot: int) -> Tuple[int, int]:
        """Replace one slot, preferring a corpus chunk not currently resident."""
        slot = int(slot) % self.case_count
        eligible = self._eligible_indices()
        source = eligible if eligible else list(range(len(self.corpus)))
        occupied = {c.source_index for i, c in enumerate(self.slots) if i != slot}
        candidates = [i for i in source if i not in occupied]
        if not candidates:
            candidates = list(source)
        old_serial = self.slots[slot].serial
        self.slots[slot] = self._make_case(random.choice(candidates))
        self._flat_cache = None
        return old_serial, self.slots[slot].serial

    @property
    def serials(self) -> Tuple[int, ...]:
        return tuple(c.serial for c in self.slots)

    def flatten(self) -> Tuple[List[List[int]], np.ndarray, np.ndarray, np.ndarray]:
        if self._flat_cache is not None:
            return self._flat_cache
        all_x: List[List[int]] = []
        all_y: List[float] = []
        all_case: List[int] = []
        all_sample: List[int] = []
        for slot, case in enumerate(self.slots):
            all_x.extend(case.inputs)
            all_y.extend(case.target.tolist())
            all_case.extend([slot] * len(case.inputs))
            all_sample.extend(case.sample_ids.tolist())
        self._flat_cache = (
            all_x,
            np.asarray(all_y, dtype=np.float64),
            np.asarray(all_case, dtype=np.int32),
            np.asarray(all_sample, dtype=np.int32),
        )
        return self._flat_cache


# ---------------------------------------------------------------------------
# Minimal rule construction / repair / genetic operators
# ---------------------------------------------------------------------------


def wildcard_count(pattern: Sequence[int]) -> int:
    return sum(1 for x in pattern if x < 0)


def capture_number(op: int) -> int:
    if op >= -15:
        return -op
    return 1 + ((-op - 16) % 16)


def opcode_with_capture(op: int, capture: int) -> int:
    if op >= -15:
        return -capture
    return -(16 + ((-op - 16) // 16) * 16 + capture - 1)


def random_replacement_opcode(wc: int) -> int:
    wc = max(1, min(MAX_WILDCARDS, wc))
    r = random.random()
    if r < 0.55:
        return -random.randint(1, min(15, wc))
    # sort bank is omitted; family bank numbers in the Nim opcode layout:
    # reverse=1, +1=2, -1=3, *2=4, //2=5 after the sort bank.
    family_bank = random.choice((1, 2, 3, 4, 5))
    cap = random.randint(1, wc)
    return -(16 + family_bank * 16 + cap - 1)


def sanitize_rule(rule: Rule, sampler: CorpusSampler, embedding: Sequence[int] | None = None) -> None:
    # At most 16 wildcards.
    seen = 0
    for i, v in enumerate(rule.pattern):
        if v < 0:
            seen += 1
            if seen > MAX_WILDCARDS:
                rule.pattern[i] = sampler.literal(embedding)
    if not rule.pattern:
        rule.pattern[:] = [sampler.literal(embedding)]

    wc = wildcard_count(rule.pattern)
    for i, t in enumerate(rule.replacement):
        if t >= 0:
            rule.replacement[i] = t % VOCAB
            continue
        # Never emit the unsupported sort bank in the minimal MPS core.
        if -31 <= t <= -16:
            t = -(32 + random.randrange(16))  # reverse family
        if t < -111:
            rule.replacement[i] = sampler.literal(embedding)
            continue
        if wc <= 0:
            rule.replacement[i] = sampler.literal(embedding)
            continue
        cap = capture_number(t)
        if cap > wc or cap <= 0:
            max_cap = min(15 if t >= -15 else 16, wc)
            t = opcode_with_capture(t, random.randint(1, max_cap))
        rule.replacement[i] = t
    if not rule.replacement:
        rule.replacement[:] = [sampler.literal(embedding)]
    if len(rule.pattern) > MAX_RULE_TOKENS:
        del rule.pattern[MAX_RULE_TOKENS:]
    if len(rule.replacement) > MAX_RULE_TOKENS:
        del rule.replacement[MAX_RULE_TOKENS:]

    # v18 hard anti-bloat invariant.  This is stronger than a runtime penalty:
    # for every match, emitted length can never exceed consumed match length.
    enforce_nonexpanding_rule(rule, sampler, embedding)


def invalidate_rule_pack_cache(rule: Rule) -> None:
    rule.anchor_kind = -1
    rule.anchor_a = -1
    rule.anchor_b = -1
    rule.pack_wc = -1
    rule.pack_leading = -1
    rule.pack_rep_literal = -1
    rule.pool_id = -1
    rule.pool_owner = 0
    rule.pack_filter_key = -2
    rule.pack_anchor_offset = -2


def rule_growth_signature(rule: Rule) -> Tuple[int, int, Tuple[int, ...]]:
    """Return (pattern_literals, replacement_literals, capture multiplicities).

    For a match with capture lengths c_i, consumed length is
      pattern_literals + sum(c_i)
    and emitted length is
      replacement_literals + sum(m_i * c_i).
    Non-expansion for *all* possible captures is therefore exactly guaranteed by
    replacement_literals <= pattern_literals and every m_i <= 1.
    """
    pat_literals = sum(1 for t in rule.pattern if t >= 0)
    rep_literals = sum(1 for t in rule.replacement if t >= 0)
    wc = wildcard_count(rule.pattern)
    refs = [0] * wc
    for t in rule.replacement:
        if t < 0 and wc > 0:
            ci = capture_number(t) - 1
            if 0 <= ci < wc:
                refs[ci] += 1
    return pat_literals, rep_literals, tuple(refs)


def is_rule_nonexpanding(rule: Rule) -> bool:
    pat_literals, rep_literals, refs = rule_growth_signature(rule)
    return rep_literals <= pat_literals and all(n <= 1 for n in refs)


def enforce_nonexpanding_rule(
    rule: Rule, sampler: CorpusSampler, embedding: Sequence[int] | None = None
) -> bool:
    """Repair one already-sanitized rule into the exact non-expanding subset.

    The repair is deterministic with respect to token order: keep the earliest
    legal replacement literals and the first reference to each capture.  This
    makes old checkpoints migrate gently instead of replacing whole rules.
    Returns True iff the replacement changed.
    """
    before = list(rule.replacement)
    literal_budget = sum(1 for t in rule.pattern if t >= 0)
    wc = wildcard_count(rule.pattern)
    used_caps = set()
    repaired: List[int] = []

    for t in rule.replacement:
        if t >= 0:
            if literal_budget > 0:
                repaired.append(t)
                literal_budget -= 1
            continue
        if wc <= 0:
            continue
        ci = capture_number(t)
        if 1 <= ci <= wc and ci not in used_caps:
            repaired.append(t)
            used_caps.add(ci)

    # Keep every rule executable. An all-wildcard pattern can preserve one
    # capture; an all-literal pattern can emit one of the literals it consumes.
    if not repaired:
        if wc > 0:
            repaired.append(-1)
        else:
            literal = next((t for t in rule.pattern if t >= 0), None)
            repaired.append(literal if literal is not None else sampler.literal(embedding))

    if repaired != before:
        rule.replacement[:] = repaired[:MAX_RULE_TOKENS]
        invalidate_rule_pack_cache(rule)
        return True
    return False


def stabilize_genome_nonexpanding(g: Genome, sampler: CorpusSampler) -> int:
    """Migrate an old/checkpoint genome into the v18 non-expanding space."""
    changed = 0
    changed_rows = []
    for i, old in enumerate(g.rules):
        if is_rule_nonexpanding(old):
            continue
        r = clone_rule(old)
        sanitize_rule(r, sampler, g.embedding)
        r.weight = 0.0
        invalidate_rule_pack_cache(r)
        g.rules[i] = r
        changed_rows.append(i)
        changed += 1
    if changed:
        _mark_genome_pack_rows(g, changed_rows)
        _clear_evaluation_state(g)
    return changed


def random_rule(
    sampler: CorpusSampler,
    embedding: Sequence[int],
    la: int | None = None,
    lb: int | None = None,
) -> Rule:
    # Nim uses chi-square(df=5) rule lengths.  ``random_genome`` supplies these
    # in one vectorized draw; keep the scalar fallback for mutation/tests.
    if la is None:
        la = int(math.ceil(sum(random.gauss(0, 1) ** 2 for _ in range(5))))
    if lb is None:
        lb = int(math.ceil(sum(random.gauss(0, 1) ** 2 for _ in range(5))))
    la = max(1, min(MAX_RULE_TOKENS, int(la)))
    lb = max(1, min(MAX_RULE_TOKENS, int(lb)))
    a = sampler.seeded_pattern(la, embedding)
    if a and random.random() < 0.15:
        a[random.randrange(len(a))] = -1
    b = [sampler.literal(embedding) for _ in range(lb)]
    wc = wildcard_count(a)
    if wc and b and random.random() < 0.20:
        b[random.randrange(len(b))] = random_replacement_opcode(wc)
    rule = Rule(a, b)
    sanitize_rule(rule, sampler, embedding)
    return rule


def random_genome(rule_count: int, sampler: CorpusSampler, embedding_enabled: bool = True) -> Genome:
    embedding = initial_embedding(embedding_enabled)
    if rule_count <= 0:
        return Genome([], embedding=embedding)
    # Sum of five squared standard normals == chi-square(df=5).  Drawing all
    # lengths at once removes millions of Python-level gaussian calls during
    # default 450 x 1500 population initialization.
    lens = np.ceil(np.random.chisquare(5.0, size=(rule_count, 2))).astype(np.int16)
    rules = [
        random_rule(sampler, embedding, int(lens[i, 0]), int(lens[i, 1]))
        for i in range(rule_count)
    ]
    return Genome(rules, embedding=embedding)


def sample_log_uniform_mutation_range(low: int, high: int) -> int:
    if high < low:
        return max(0, high)
    if high == low:
        return high
    x = math.exp(math.log(float(low)) + random.random() * math.log(float(high) / float(low)))
    return max(low, min(high, int(round(x))))


def log_uniform_mutation_count(n: int) -> int:
    if n <= 0:
        return 0
    return sample_log_uniform_mutation_range(1, n)


def sample_mutation_count_for_regime(n: int, regime: str) -> int:
    if n <= 0:
        return 0
    if regime == "local":
        return sample_log_uniform_mutation_range(1, min(n, LOCAL_MUTATION_MAX_RULES))
    if regime == "balanced":
        return sample_log_uniform_mutation_range(1, min(n, BALANCED_MUTATION_MAX_RULES))
    return log_uniform_mutation_count(n)


def choose_mutation_regime(stagnation: int) -> str:
    """Nim-style hierarchical mutation without the heavy FULL-score bandit.

    Normal training is deliberately dominated by 1..8-row local refinement.
    A stalled search shifts probability toward 1..64-row balanced edits and a
    small global lane.  Unlike v17, a normal mutation no longer redraws a
    1..1500-row budget on every child.
    """
    pressure = min(1.0, math.sqrt(max(0, stagnation)) / 6.0)
    p_local = 0.80 - 0.35 * pressure
    p_balanced = 0.18 + 0.22 * pressure
    draw = random.random()
    if draw < p_local:
        return "local"
    if draw < p_local + p_balanced:
        return "balanced"
    return "explore"


def _guided_rule_priority(g: Genome, rule_id: int) -> float:
    # Structural Rule objects are shared copy-on-write, so fitted ridge weights
    # must be genome-local. Fall back to Rule.weight only for legacy genomes.
    if len(g.readout_weights) == len(g.rules):
        return abs(float(g.readout_weights[rule_id]))
    return abs(float(g.rules[rule_id].weight))


def pick_guided_mutation_targets(g: Genome, count: int) -> List[int]:
    n = len(g.rules)
    count = max(0, min(n, int(count)))
    if count <= 0:
        return []
    if count >= n:
        return list(range(n))

    chosen: set[int] = set()
    out: List[int] = []
    while len(out) < count:
        candidate = -1
        # Same spirit as the Nim guided target sampler: keep a 20% unbiased
        # lane so a stale importance estimate can never freeze a locus forever.
        if random.random() < 0.20 or len(chosen) * 4 >= n * 3:
            for _ in range(16):
                p = random.randrange(n)
                if p not in chosen:
                    candidate = p
                    break
        else:
            best_priority = float("inf")
            for _ in range(8):
                p = random.randrange(n)
                if p in chosen:
                    continue
                priority = _guided_rule_priority(g, p)
                if candidate < 0 or priority < best_priority:
                    candidate = p
                    best_priority = priority
        if candidate < 0:
            start = random.randrange(n)
            for off in range(n):
                p = (start + off) % n
                if p not in chosen:
                    candidate = p
                    break
        if candidate < 0:
            break
        chosen.add(candidate)
        out.append(candidate)
    return out


def mutate_sequence(
    seq: List[int],
    is_pattern: bool,
    sampler: CorpusSampler,
    wc_hint: int = 1,
    embedding: Sequence[int] | None = None,
) -> None:
    """One small structural edit, matching the later Nim mutation geometry.

    Segment reversal uses a log-uniform span instead of v17's uniform 2..len
    draw, so local edits remain genuinely local.  Corpus n-gram insertion is a
    separate rare pattern proposal rather than one sixth of all mutations.
    """
    if not seq:
        if is_pattern and random.random() < 0.60:
            seq.append(-1)
        elif (not is_pattern) and random.random() < 0.20:
            seq.append(random_replacement_opcode(max(1, wc_hint)))
        else:
            seq.append(sampler.literal(embedding))
        return

    if is_pattern and embedding is not None and sampler.ngrams and random.random() < 0.10:
        raw = random.choice(sampler.ngrams)
        ng = [int(embedding[b]) for b in raw]
        room = max(0, MAX_RULE_TOKENS - len(seq))
        if room > 0 and ng:
            n = min(room, len(ng))
            p = random.randrange(len(seq) + 1)
            seq[p:p] = ng[:n]
            return

    op = random.randrange(5)
    if op == 0:  # substitute exactly one token/opcode
        p = random.randrange(len(seq))
        if is_pattern and random.random() < 0.20:
            seq[p] = -1 - random.randrange(min(15, MAX_WILDCARDS))
        elif (not is_pattern) and random.random() < 0.20:
            seq[p] = random_replacement_opcode(max(1, wc_hint))
        else:
            seq[p] = sampler.literal(embedding)
    elif op == 1 and len(seq) < MAX_RULE_TOKENS:  # insert one token
        p = random.randrange(len(seq) + 1)
        if is_pattern and random.random() < 0.15:
            v = -1 - random.randrange(min(15, MAX_WILDCARDS))
        elif (not is_pattern) and random.random() < 0.20:
            v = random_replacement_opcode(max(1, wc_hint))
        else:
            v = sampler.literal(embedding)
        seq.insert(p, v)
    elif op == 2 and len(seq) > 1:  # delete one token
        del seq[random.randrange(len(seq))]
    elif op == 3 and len(seq) >= 2:  # log-uniform local-to-global reversal
        ln = sample_log_uniform_mutation_range(2, len(seq))
        p = random.randrange(0, len(seq) - ln + 1)
        seq[p : p + ln] = reversed(seq[p : p + ln])
    elif op == 4 and len(seq) < MAX_RULE_TOKENS:  # duplicate one token
        seq.insert(random.randrange(len(seq) + 1), seq[random.randrange(len(seq))])


def _mutate_rule_once(r: Rule, sampler: CorpusSampler, embedding: Sequence[int], mode: int | None = None) -> None:
    if mode is None:
        mode = random.randrange(4)
    if mode == 0:
        mutate_sequence(r.pattern, True, sampler, embedding=embedding)
    elif mode == 1:
        mutate_sequence(r.replacement, False, sampler, wildcard_count(r.pattern), embedding)
    elif mode == 2:
        mutate_sequence(r.pattern, True, sampler, embedding=embedding)
        if random.random() < 0.65:
            mutate_sequence(r.replacement, False, sampler, wildcard_count(r.pattern), embedding)
    else:
        r.pattern, r.replacement = r.replacement, r.pattern
    r.weight = 0.0
    sanitize_rule(r, sampler, embedding)
    invalidate_rule_pack_cache(r)
    assert is_rule_nonexpanding(r)


def _mutate_selected_rules(g: Genome, ids: Sequence[int], sampler: CorpusSampler, mode: int | None = None) -> int:
    changed = 0
    changed_rows = []
    for idx in ids:
        old = g.rules[idx]
        r = clone_rule(old)
        _mutate_rule_once(r, sampler, g.embedding, mode)
        # A structural edit can sanitize back to the original. Try a bounded
        # second fine edit on the SAME locus; do not silently widen the budget.
        if r.pattern == old.pattern and r.replacement == old.replacement:
            _mutate_rule_once(r, sampler, g.embedding, 0 if mode == 1 else 1)
        if r.pattern != old.pattern or r.replacement != old.replacement:
            g.rules[idx] = r
            changed_rows.append(idx)
            changed += 1
    if changed_rows:
        _mark_genome_pack_rows(g, changed_rows)
    return changed


def _relocate_rule_block(rules: List[Rule], src: int, length: int, dst: int) -> None:
    if length <= 0 or src == dst:
        return
    block = rules[src : src + length]
    del rules[src : src + length]
    if dst > src:
        dst -= length
    dst = max(0, min(len(rules), dst))
    rules[dst:dst] = block


def mutate_genome_global(
    g: Genome,
    sampler: CorpusSampler,
) -> int:
    n = len(g.rules)
    if n <= 0:
        return 0
    count = log_uniform_mutation_count(n)
    ids = pick_guided_mutation_targets(g, count)
    changed = _mutate_selected_rules(g, ids, sampler)
    _clear_evaluation_state(g)
    return changed


def mutate_genome(
    g: Genome,
    sampler: CorpusSampler,
    embedding_mutation_rate: float = EMBEDDING_MUTATION_RATE,
    regime: str = "local",
) -> int:
    """Nim-v57-style precise structural mutation with strict row budgets."""
    n = len(g.rules)
    if n == 0:
        return 0
    if regime == "explore":
        return mutate_genome_global(g, sampler)

    scale = 0.72 if regime == "local" else 1.0
    changed = 0
    major_used = False

    if (not major_used) and n >= 2 and random.random() < min(0.45, 0.14 * scale):
        ids = pick_guided_mutation_targets(g, sample_mutation_count_for_regime(n, regime))
        touched = set()
        for i in ids:
            off = 1 + random.randrange(n - 1)
            j = (i + off) % n
            g.rules[i], g.rules[j] = g.rules[j], g.rules[i]
            touched.add(i); touched.add(j)
        changed += len(touched)
        _mark_genome_pack_rows(g, touched)
        major_used = True

    if (not major_used) and random.random() < min(0.28, 0.06 * scale):
        ids = pick_guided_mutation_targets(g, sample_mutation_count_for_regime(n, regime))
        changed += _mutate_selected_rules(g, ids, sampler, mode=3)
        major_used = True

    if (not major_used) and random.random() < min(0.60, 0.26 * scale):
        ids = pick_guided_mutation_targets(g, sample_mutation_count_for_regime(n, regime))
        changed += _mutate_selected_rules(g, ids, sampler, mode=2)
        major_used = True

    if (not major_used) and n >= 16 and random.random() < 0.045:
        block_len = sample_mutation_count_for_regime(n, regime)
        block_len = max(1, min(n, block_len))
        if block_len == n:
            g.rules.reverse()
            changed += n
        else:
            src = random.randrange(0, n - block_len + 1)
            dst = random.randrange(0, n - block_len + 1)
            if dst == src:
                dst = (dst + block_len) % (n - block_len + 1)
            if random.random() < 0.5:
                _relocate_rule_block(g.rules, src, block_len, dst)
            else:
                # Fixed-length genome: duplicate a module by replacing another
                # interval, exactly like the Nim block-duplication operator.
                copied = list(g.rules[src : src + block_len])
                g.rules[dst : dst + block_len] = copied
            changed += block_len
        _invalidate_genome_pack(g)
        major_used = True

    if not major_used:
        ids = pick_guided_mutation_targets(g, sample_mutation_count_for_regime(n, regime))
        # Fine fallback: mutate either a or b, not both, for each selected row.
        for idx in ids:
            mode = 0 if random.random() < 0.5 else 1
            changed += _mutate_selected_rules(g, [idx], sampler, mode=mode)
        major_used = True

    # As in the later Nim trainer, the embedding is kept out of strict local
    # refinement; coordinate remaps are a balanced-scale structural event.
    if regime == "balanced" and embedding_mutation_rate > 0.0:
        mutate_embedding(g, embedding_mutation_rate)

    _clear_evaluation_state(g)
    return changed


def _map_unchanged_base_span_to_target(
    matching_blocks: Sequence,
    base_len: int,
    target_len: int,
    start: int,
    end: int,
) -> Tuple[int, int] | None:
    """Map an A-coordinate interval into Y only when Y kept that A span unchanged.

    This is deliberately conservative: a donor edit is applied only through an
    exact A<->Y matching block.  If Y has independently edited the same region,
    the patch is treated like a git merge conflict and Y wins.
    """
    start = int(start)
    end = int(end)
    if start < 0 or end < start or end > base_len:
        return None

    # Replacements/deletions need the whole consumed base interval unchanged.
    if start < end:
        for block in matching_blocks:
            a0, b0, size = int(block.a), int(block.b), int(block.size)
            if size <= 0:
                continue
            if a0 <= start and end <= a0 + size:
                off = start - a0
                return b0 + off, b0 + off + (end - start)
        return None

    # Insertions consume no base token.  Interior boundaries of an exact match
    # are unambiguous.  Edge boundaries are accepted only when both sequences
    # share that edge, so a competing Y insertion is not overwritten.
    pos = start
    for block in matching_blocks:
        a0, b0, size = int(block.a), int(block.b), int(block.size)
        if size <= 0:
            continue
        if a0 < pos < a0 + size:
            return b0 + (pos - a0), b0 + (pos - a0)
        if pos == 0 and a0 == 0 and b0 == 0:
            return 0, 0
        if (
            pos == base_len
            and a0 + size == base_len
            and b0 + size == target_len
        ):
            return target_len, target_len
    return None


def merge_sequence_delta(
    base: Sequence[int], donor: Sequence[int], target: Sequence[int]
) -> Tuple[List[int], int]:
    """Apply the non-conflicting A->X edit script to Y.

    Think ``git merge`` on one Rule field: A is the merge base, X the donor
    branch and Y the target branch.  Exact Y-local edits win conflicts.  This is
    the categorical analogue of a differential-evolution delta; no arithmetic
    over token ids is invented.
    """
    a = list(map(int, base))
    x = list(map(int, donor))
    y = list(map(int, target))
    if x == a or y == x:
        return y, 0
    if y == a:
        return x, 1

    ax = SequenceMatcher(None, a, x, autojunk=False)
    ay = SequenceMatcher(None, a, y, autojunk=False)
    matching = ay.get_matching_blocks()
    patches: List[Tuple[int, int, List[int]]] = []

    for tag, a1, a2, x1, x2 in ax.get_opcodes():
        if tag == "equal":
            continue
        mapped = _map_unchanged_base_span_to_target(
            matching, len(a), len(y), a1, a2
        )
        if mapped is None:
            continue
        y1, y2 = mapped
        patches.append((int(y1), int(y2), list(x[x1:x2])))

    if not patches:
        return y, 0

    # Coordinates above refer to the original Y.  Apply from right to left so
    # insertions/deletions cannot shift a patch that has not yet been applied.
    out = list(y)
    applied = 0
    for y1, y2, replacement in sorted(patches, key=lambda z: (z[0], z[1]), reverse=True):
        out[y1:y2] = replacement
        applied += 1
    return out, applied


def diff_merge_crossover(
    target: Genome,
    base: Genome,
    donor: Genome,
    sampler: CorpusSampler,
    embedding_crossover_rate: float = EMBEDDING_CROSSOVER_RATE,
) -> Tuple[Genome, int]:
    """Three-parent A->X delta merge applied to Y (target).

    Rule literals from A and X are first recoded into Y's embedding coordinate
    system.  For every corresponding rule row, the structural edit script
    ``base -> donor`` is then applied to ``target`` only where target still
    contains the merge-base material.  Conflicting target edits are preserved.

    Returns ``(child, changed_rows)``.  The child keeps normal copy-on-write
    sharing for untouched target rows, so this remains cheap when the delta has
    little ancestry overlap.
    """
    n = min(len(target.rules), len(base.rules), len(donor.rules))
    if n <= 0:
        return clone_genome_shallow(target), 0

    base_tr = None if base.embedding == target.embedding else embedding_translation(base.embedding, target.embedding)
    donor_tr = None if donor.embedding == target.embedding else embedding_translation(donor.embedding, target.embedding)

    rules = list(target.rules)
    changed_rows: List[int] = []
    for i in range(n):
        rt = target.rules[i]
        rb = base.rules[i] if base_tr is None else recode_rule(base.rules[i], base_tr)
        rx = donor.rules[i] if donor_tr is None else recode_rule(donor.rules[i], donor_tr)

        # No A->X delta at this locus.
        if rb.pattern == rx.pattern and rb.replacement == rx.replacement:
            continue

        new_pattern, p_edits = merge_sequence_delta(rb.pattern, rx.pattern, rt.pattern)
        new_replacement, r_edits = merge_sequence_delta(rb.replacement, rx.replacement, rt.replacement)
        if p_edits <= 0 and r_edits <= 0:
            continue
        if new_pattern == rt.pattern and new_replacement == rt.replacement:
            continue

        rr = clone_rule(rt)
        rr.pattern = new_pattern
        rr.replacement = new_replacement
        rr.weight = 0.0
        # A pattern edit can invalidate capture references in the independently
        # inherited replacement (and vice versa).  Reuse the engine's canonical
        # repair, including the hard non-expanding invariant.
        sanitize_rule(rr, sampler, target.embedding)
        invalidate_rule_pack_cache(rr)
        if rr.pattern == rt.pattern and rr.replacement == rt.replacement:
            continue
        rules[i] = rr
        changed_rows.append(i)

    child = Genome(rules, embedding=list(target.embedding))
    _inherit_genome_pack_cache(target, child)
    if changed_rows:
        _mark_genome_pack_rows(child, changed_rows)

    # Keep the existing evolved-embedding crossover semantics.  This happens
    # after the structural merge so all A/X patches above were compared in one
    # stable target coordinate system.
    crossover_embedding(child, donor, embedding_crossover_rate)
    _clear_evaluation_state(child)
    return child, len(changed_rows)


def crossover(
    p1: Genome,
    p2: Genome,
    embedding_crossover_rate: float = EMBEDDING_CROSSOVER_RATE,
) -> Genome:
    n = min(len(p1.rules), len(p2.rules))
    if n <= 1:
        return clone_genome_shallow(p1)
    l = random.randrange(0, n - 1)
    r = random.randrange(l + 1, n + 1)

    # Copy-on-write crossover: p1 rules outside the splice are shared read-only.
    # Only donor rules crossing coordinate systems need fresh Rule objects.
    rules = list(p1.rules)
    # v56: when both parents already use the same token coordinates, donor
    # Rule objects are immutable/shareable and all evaluator-local pack memo is
    # valid as-is. Avoid manufacturing fresh recoded Rules just to copy tokens.
    if p2.embedding == p1.embedding:
        rules[l:r] = p2.rules[l:r]
    else:
        tr = embedding_translation(p2.embedding, p1.embedding)
        rules[l:r] = [recode_rule(x, tr) for x in p2.rules[l:r]]
    child = Genome(rules, embedding=list(p1.embedding))
    _inherit_genome_pack_cache(p1, child)
    _mark_genome_pack_rows(child, range(l, r))
    crossover_embedding(child, p2, embedding_crossover_rate)
    return child


# ---------------------------------------------------------------------------
# v58: budget-neutral linkage learning / quality diversity / operator learning
# ---------------------------------------------------------------------------


def _clear_evaluation_state(g: Genome) -> None:
    """Invalidate score/readout and v60 inference/Pareto state after a structural edit."""
    g.fitness = float("-inf")
    g.case_scores = []
    g.readout_weights = []
    g.inference_accuracy = float("nan")
    g.inference_full_accuracy = float("nan")
    g.pareto_rank = 2_147_483_647
    g.pareto_crowding = 0.0
    g._eval_case_serials = ()


def mark_offspring_origin(child: Genome, parent: Genome, operator: str, mix_rows: int = 0) -> None:
    """Store a comparable parent baseline for delayed, zero-cost credit assignment."""
    child._origin_operator = str(operator)
    child._parent_case_scores = tuple(map(float, parent.case_scores))
    child._parent_case_serials = tuple(parent._eval_case_serials)
    child._mix_rows = int(mix_rows)


def offspring_overlap_reward(g: Genome) -> float | None:
    """Child-parent fitness delta on case serials that are literally identical."""
    if not g._origin_operator or not g._parent_case_serials or not g._eval_case_serials:
        return None
    parent_by_serial = {
        int(serial): float(score)
        for serial, score in zip(g._parent_case_serials, g._parent_case_scores)
        if math.isfinite(float(score))
    }
    diffs = []
    for serial, score in zip(g._eval_case_serials, g.case_scores):
        old = parent_by_serial.get(int(serial))
        if old is not None and math.isfinite(float(score)):
            diffs.append(float(score) - old)
    return float(statistics.fmean(diffs)) if diffs else None


class AdaptiveOperatorBandit:
    """Small non-stationary UCB scheduler using already-paid-for evaluations."""
    def __init__(self, arms: Sequence[str], exploration: float = 3.0e-4, epsilon: float = 0.04):
        self.arms = tuple(map(str, arms))
        self.exploration = max(0.0, float(exploration))
        self.epsilon = min(1.0, max(0.0, float(epsilon)))
        self.counts = {a: 0 for a in self.arms}
        self.mean = {a: 0.0 for a in self.arms}
        self.success = {a: 0 for a in self.arms}

    def restore_from_history(self, history: Sequence[dict]) -> None:
        if not history:
            return
        row = history[-1]
        for arm in self.arms:
            try:
                self.counts[arm] = max(0, int(float(row.get(f"op_count_{arm}", 0) or 0)))
                self.mean[arm] = float(row.get(f"op_reward_{arm}", 0.0) or 0.0)
                self.success[arm] = max(0, int(float(row.get(f"op_success_{arm}", 0) or 0)))
            except (TypeError, ValueError):
                pass

    def choose(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        unseen = [a for a in self.arms if self.counts[a] == 0]
        if unseen:
            return random.choice(unseen)
        total = max(1, sum(self.counts.values()))
        log_total = math.log(total + 1.0)
        best_arm = self.arms[0]
        best_score = float("-inf")
        for arm in self.arms:
            bonus = self.exploration * math.sqrt(log_total / float(self.counts[arm]))
            score = self.mean[arm] + bonus
            if score > best_score:
                best_score = score
                best_arm = arm
        return best_arm

    def observe(self, arm: str, reward: float) -> None:
        if arm not in self.counts or not math.isfinite(float(reward)):
            return
        r = max(-0.05, min(0.05, float(reward)))
        self.counts[arm] += 1
        if r > 0.0:
            self.success[arm] += 1
        # Keep adapting after thousands of generations instead of converging to a
        # frozen lifetime average; the useful operator changes near saturation.
        alpha = max(0.06, 1.0 / min(16.0, float(self.counts[arm])))
        self.mean[arm] = (1.0 - alpha) * self.mean[arm] + alpha * r

    def observe_population(self, population: Sequence[Genome]) -> tuple[int, float]:
        updates = 0
        rewards = []
        for g in population:
            r = offspring_overlap_reward(g)
            if r is None:
                continue
            self.observe(g._origin_operator, r)
            updates += 1
            rewards.append(r)
        return updates, (float(statistics.fmean(rewards)) if rewards else float("nan"))

    def add_history(self, row: dict) -> None:
        for arm in self.arms:
            row[f"op_count_{arm}"] = self.counts[arm]
            row[f"op_reward_{arm}"] = self.mean[arm]
            row[f"op_success_{arm}"] = self.success[arm]


class SparseLinkageModel:
    """Bounded linkage family inferred from elite readout activity."""
    def __init__(self):
        self.modules: List[tuple[int, ...]] = []
        self.module_scores: List[float] = []
        self.generation = -1
        self.loci = 0

    def refresh(
        self,
        population: Sequence[Genome],
        generation: int,
        refresh_every: int,
        elite_count: int,
        max_loci: int,
        max_modules: int,
        max_module_size: int,
    ) -> None:
        if int(max_loci) <= 0 or int(max_modules) <= 0:
            self.modules = []
            self.module_scores = []
            self.loci = 0
            self.generation = int(generation)
            return
        if self.modules and (int(generation) - self.generation) < max(1, int(refresh_every)):
            return
        if not population or not population[0].rules:
            self.modules = []
            self.module_scores = []
            self.generation = int(generation)
            return
        n_rules = len(population[0].rules)
        valid = [g for g in population[:max(2, int(elite_count))] if len(g.readout_weights) == n_rules]
        if len(valid) < 2:
            self.modules = []
            self.module_scores = []
            self.generation = int(generation)
            return

        # At defaults this is only 96x1500 and the correlation matrix is 128x128.
        w = np.abs(np.asarray([g.readout_weights for g in valid], dtype=np.float32))
        importance = np.mean(w, axis=0, dtype=np.float64)
        k = min(n_rules, max(8, int(max_loci)))
        loci = np.argpartition(importance, -k)[-k:] if k < n_rules else np.arange(n_rules)
        loci = loci[np.argsort(importance[loci])[::-1]]
        self.loci = int(len(loci))

        z = np.log1p(w[:, loci].astype(np.float64, copy=False))
        z -= z.mean(axis=0, keepdims=True)
        norms = np.sqrt(np.sum(z * z, axis=0))
        nz = norms > 1.0e-12
        z[:, nz] /= norms[nz]
        corr = np.abs(z.T @ z)
        np.fill_diagonal(corr, 0.0)
        locus_to_sub = {int(v): i for i, v in enumerate(loci)}

        sizes = [x for x in (2, 4, 8, 16, 32) if x <= max(2, int(max_module_size)) and x <= len(loci)]
        if not sizes:
            sizes = [min(2, len(loci))]
        seed_count = min(len(loci), max(8, int(max_modules) // max(1, len(sizes))))
        candidates: dict[tuple[int, ...], float] = {}
        for si in range(seed_count):
            seed = int(loci[si])
            order = np.argsort(corr[si])[::-1]
            for size in sizes:
                chosen = [seed]
                for j in order:
                    locus = int(loci[int(j)])
                    if locus != seed:
                        chosen.append(locus)
                    if len(chosen) >= size:
                        break
                module = tuple(sorted(set(chosen)))
                if len(module) < 2:
                    continue
                idx = [locus_to_sub[q] for q in module]
                c = corr[np.ix_(idx, idx)]
                dep = float(c.sum() / max(1, len(module) * (len(module) - 1)))
                imp = float(np.mean(importance[list(module)]))
                candidates[module] = max(candidates.get(module, -1.0), dep * (imp + 1.0e-12))

            # Retain a small ordered-program prior in addition to data-driven
            # non-contiguous modules.
            span = min(max(2, int(max_module_size)), 8, n_rules)
            lo = max(0, min(n_rules - span, seed - span // 2))
            contiguous = tuple(range(lo, lo + span))
            candidates[contiguous] = max(candidates.get(contiguous, -1.0), float(importance[seed]) * 0.25)

        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:max(1, int(max_modules))]
        self.modules = [m for m, _ in ranked]
        self.module_scores = [float(v) for _, v in ranked]
        self.generation = int(generation)

    def choose_module(self) -> tuple[int, ...]:
        if not self.modules:
            return ()
        head = min(len(self.modules), max(8, len(self.modules) // 2))
        return random.choice(self.modules[:head] if random.random() < 0.85 else self.modules)

    def choose_module_for(self, base: Genome, donor: Genome, tries: int = 6) -> tuple[int, ...]:
        """Cheap surrogate-optimal module choice using already fitted readout mass."""
        if not self.modules:
            return ()
        if len(base.readout_weights) != len(base.rules) or len(donor.readout_weights) != len(donor.rules):
            return self.choose_module()
        head = min(len(self.modules), max(8, len(self.modules) // 2))
        best_module = ()
        best_gain = float("-inf")
        for _ in range(max(1, int(tries))):
            idx = random.randrange(head) if random.random() < 0.85 else random.randrange(len(self.modules))
            module = self.modules[idx]
            donor_mass = sum(abs(float(donor.readout_weights[j])) for j in module)
            base_mass = sum(abs(float(base.readout_weights[j])) for j in module)
            # Linkage score is only a tiny tie-breaker; donor-vs-base evidence
            # controls the zero-cost proposal.  Actual merit is learned next gen.
            gain = donor_mass - base_mass + 1.0e-6 * self.module_scores[idx]
            if gain > best_gain:
                best_gain = gain
                best_module = module
        return best_module or self.choose_module()


def linkage_mix(base: Genome, donor: Genome, model: SparseLinkageModel, module_count: int = 1) -> tuple[Genome, int]:
    """Transplant learned modules with copy-on-write and no extra evaluation."""
    child = clone_genome_shallow(base)
    if not model.modules or len(base.rules) != len(donor.rules):
        return child, 0
    changed_rows: set[int] = set()
    same_coords = donor.embedding == base.embedding
    translation = None if same_coords else embedding_translation(donor.embedding, base.embedding)
    attempts = max(2, int(module_count) * 3)
    used = 0
    for _ in range(attempts):
        if used >= max(1, int(module_count)):
            break
        module = model.choose_module_for(base, donor)
        if not module:
            continue
        local_changed = []
        for idx in module:
            if not (0 <= idx < len(child.rules)):
                continue
            src = donor.rules[idx]
            dst = child.rules[idx]
            if src.pattern == dst.pattern and src.replacement == dst.replacement:
                continue
            child.rules[idx] = src if same_coords else recode_rule(src, translation)
            local_changed.append(idx)
        if local_changed:
            changed_rows.update(local_changed)
            used += 1
    if changed_rows:
        _mark_genome_pack_rows(child, changed_rows)
        _clear_evaluation_state(child)
    return child, len(changed_rows)


def qd_descriptor(g: Genome, bins: int = 6) -> tuple[int, ...]:
    """MAP-Elites descriptor from cheap but behaviorally useful genome statistics.

    The axes deliberately separate sparse/dense readouts, embedding geometry and
    the structure of the rules that actually matter to the ridge readout.  Only a
    tiny top-k rule sample is inspected, keeping this host-side work negligible.
    """
    bins = max(2, int(bins))
    n = max(1, len(g.rules))
    if len(g.readout_weights) == len(g.rules):
        absw = np.abs(np.asarray(g.readout_weights, dtype=np.float64))
        active = int(np.count_nonzero(absw > 1.0e-12))
        k = min(12, active if active > 0 else n)
        if k > 0:
            if k < n:
                top_ids = np.argpartition(absw, -k)[-k:]
            else:
                top_ids = np.arange(n)
            pat_len = float(np.mean([len(g.rules[int(i)].pattern) for i in top_ids]))
            wc_frac = float(np.mean([
                sum(int(t) < 0 for t in g.rules[int(i)].pattern) / max(1, len(g.rules[int(i)].pattern))
                for i in top_ids
            ]))
        else:
            pat_len = 0.0
            wc_frac = 0.0
    else:
        active = 0
        pat_len = 0.0
        wc_frac = 0.0

    # Active counts in successful runs tend to live far below 1500, so a 256-rule
    # soft cap gives the sparse end useful resolution instead of collapsing it.
    active_scale = math.sqrt(min(active, 256) / 256.0)
    active_bin = min(bins - 1, int(active_scale * bins))
    latent = sum(int(v) >= INPUT_BYTE_COUNT for v in g.embedding)
    moved = sum(int(v) != i for i, v in enumerate(g.embedding))
    latent_bin = min(bins - 1, int(math.sqrt(latent / float(EMBEDDING_ENTRY_COUNT)) * bins))
    moved_bin = min(bins - 1, int((moved / float(EMBEDDING_ENTRY_COUNT)) * bins))
    # Most evolved useful rules are short; resolve 0..12 tokens finely and saturate.
    pat_bin = min(bins - 1, int((min(12.0, pat_len) / 12.0) * bins))
    wildcard_bin = min(bins - 1, int(min(1.0, wc_frac) * bins))
    specialist = int(np.argmax(np.asarray(g.case_scores, dtype=np.float64))) if g.case_scores else 0
    return active_bin, latent_bin, moved_bin, pat_bin, wildcard_bin, specialist


def build_qd_grid(population: Sequence[Genome], bins: int = 6) -> dict[tuple[int, ...], Genome]:
    grid: dict[tuple[int, ...], Genome] = {}
    for g in population:
        if not math.isfinite(float(g.fitness)):
            continue
        key = qd_descriptor(g, bins)
        old = grid.get(key)
        if old is None or pareto_better(g, old):
            grid[key] = g
    return grid


class PlateauTracker:
    """Long-window signal; rolling difficulty remains noise, not proof of convergence."""
    def __init__(self, window: int, min_gain: float):
        from collections import deque
        self.window = max(2, int(window))
        self.values = deque(maxlen=2 * self.window)
        self.min_gain = max(0.0, float(min_gain))
        self.stalled = 0

    def update(self, fitness: float):
        if not math.isfinite(fitness):
            return self.stalled, float("nan")
        self.values.append(fitness)
        if len(self.values) < 2 * self.window:
            return 0, float("nan")
        values = list(self.values)
        gain = statistics.fmean(values[self.window:]) - statistics.fmean(values[:self.window])
        self.stalled = self.stalled + 1 if gain <= self.min_gain else 0
        return self.stalled, gain


class CaseSelector:
    """MAD epsilon-lexicase keeps specialists that mean-only tournaments lose."""
    def __init__(self, population):
        self.population = population
        self.scores = np.asarray([g.case_scores for g in population], dtype=np.float64)
        self.epsilon = np.maximum(1e-12, np.median(
            np.abs(self.scores - np.median(self.scores, axis=0)), axis=0))

    def pick(self):
        candidates = np.arange(len(self.population))
        order = list(range(self.scores.shape[1]))
        random.shuffle(order)
        for case in order:
            scores = self.scores[candidates, case]
            candidates = candidates[scores >= scores.max() - self.epsilon[case]]
            if len(candidates) <= 1:
                break
        # Preserve lexicase's case-specialist pressure, but when several
        # candidates survive let the second (inference) objective participate in
        # the tie-break instead of choosing a dominated genome uniformly.
        if len(candidates) == 1:
            return self.population[int(candidates[0])]
        return min((self.population[int(i)] for i in candidates), key=pareto_sort_key)


def tournament(pop: Sequence[Genome], k: int) -> Genome:
    ids = random.sample(range(len(pop)), k=min(k, len(pop)))
    return min((pop[i] for i in ids), key=pareto_sort_key)


# ---------------------------------------------------------------------------
# CPU reference rewrite trajectory
# ---------------------------------------------------------------------------


def _match_at_cpu(state: Sequence[int], pattern: Sequence[int], start: int):
    n = len(state)
    pos = start
    p = 0
    caps: List[Tuple[int, int]] = []
    while p < len(pattern) and pattern[p] >= 0:
        if pos >= n or state[pos] != pattern[p]:
            return None
        pos += 1
        p += 1
    while p < len(pattern):
        if pattern[p] >= 0 or len(caps) >= MAX_WILDCARDS:
            return None
        p += 1
        cs = pos
        lit_begin = p
        while p < len(pattern) and pattern[p] >= 0:
            p += 1
        lit = pattern[lit_begin:p]
        if not lit:
            if p >= len(pattern):
                caps.append((cs, n - pos))
                pos = n
            else:
                caps.append((cs, 0))
            continue
        found = -1
        for cand in range(pos, n - len(lit) + 1):
            if state[cand : cand + len(lit)] == list(lit):
                found = cand
                break
        if found < 0:
            return None
        caps.append((cs, found - pos))
        pos = found + len(lit)
    return pos, caps


def _emit_replacement_cpu(state: Sequence[int], rep: Sequence[int], caps: Sequence[Tuple[int, int]]) -> List[int]:
    out: List[int] = []
    wc = len(caps)
    for t in rep:
        if t >= 0:
            out.append(t)
            continue
        if wc <= 0:
            continue
        if t >= -15:
            kind, ci = 1, -t - 1
        elif t >= -31:
            # Sort is intentionally excluded by generation; keep reference safe.
            kind, ci = 7, -t - 16
        elif t >= -47:
            kind, ci = 2, -t - 32
        elif t >= -111:
            off = -t - 48
            bank, ci = divmod(off, 16)
            kind = 3 + bank
        else:
            continue
        if ci >= wc:
            ci = 0
        s, ln = caps[ci]
        vals = list(state[s : s + ln])
        if kind == 2:
            vals.reverse()
        elif kind == 7:
            vals.sort()
        elif kind == 3:
            vals = [((v + 1) & 511) if 0 <= v < 512 else v for v in vals]
        elif kind == 4:
            vals = [((v + 511) & 511) if 0 <= v < 512 else v for v in vals]
        elif kind == 5:
            vals = [((v * 2) & 511) if 0 <= v < 512 else v for v in vals]
        elif kind == 6:
            vals = [v // 2 if v < 512 else v for v in vals]
        out.extend(vals)
    return out


def replace_once_cpu(state: Sequence[int], rule: Rule, max_output: int) -> Tuple[List[int], bool, bool]:
    pattern = rule.pattern
    if not pattern or not state:
        return list(state), False, False
    leading = pattern[0] >= 0
    out: List[int] = []
    prev = 0
    scan = 0
    matched = False
    while scan < len(state):
        chosen = None
        if leading:
            first = pattern[0]
            for cand in range(scan, len(state)):
                if state[cand] != first:
                    continue
                m = _match_at_cpu(state, pattern, cand)
                if m is not None:
                    chosen = (cand, m[0], m[1])
                    break
        else:
            m = _match_at_cpu(state, pattern, scan)
            if m is not None:
                chosen = (scan, m[0], m[1])
        if chosen is None:
            break
        s, finish, caps = chosen
        matched = True
        out.extend(state[prev:s])
        out.extend(_emit_replacement_cpu(state, rule.replacement, caps))
        if len(out) > max_output:
            return list(state), True, True
        prev = finish
        scan = finish
    if not matched:
        return list(state), False, False
    out.extend(state[prev:])
    if len(out) > max_output:
        return list(state), True, True
    return out, True, False


def trajectory_features_cpu(inputs: Sequence[Sequence[int]], genome: Genome, max_output: int):
    inputs = embed_inputs(inputs, genome.embedding)
    b = len(inputs)
    rcount = len(genome.rules)
    features = np.zeros((b, rcount), dtype=np.float32)
    baselines = np.zeros(b, dtype=np.float64)
    stats = np.zeros((b, 4), dtype=np.int32)  # rounds, total fires, overflow, cycle
    final_states: List[List[int]] = []
    for j, inp in enumerate(inputs):
        state = list(inp)
        limit = max(1, int(math.ceil(2.0 * math.sqrt(max(1, len(inp))))))
        seen = {hash(tuple(state))}
        for rnd in range(limit):
            rewrote = False
            overflow = False
            for ri, rule in enumerate(genome.rules):
                next_state, matched, of = replace_once_cpu(state, rule, max_output)
                if matched:
                    features[j, ri] += 1.0
                    stats[j, 1] += 1
                    rewrote = True
                if of:
                    baselines[j] -= 1.0
                    stats[j, 2] = 1
                    overflow = True
                    break
                state = next_state
            stats[j, 0] = rnd + 1
            if overflow or not rewrote:
                break
            h = hash(tuple(state))
            if h in seen:
                stats[j, 3] = 1
                break
            seen.add(h)
        final_states.append(state)
    return features, baselines, stats, final_states


# ---------------------------------------------------------------------------
# MPS population evaluator v17: cooperative threadgroups + literal rewrite specialization + persistent Global Rule Pool
# ---------------------------------------------------------------------------
# v6 still mapped one whole trajectory to one GPU thread, so the expensive
# sequential matcher stayed sequential and SIMD lanes diverged badly.  v7
# changes the execution model:
#   * one Metal threadgroup == one (genome, sample) trajectory
#   * 64 threads cooperatively embed/copy text, build an EXACT candidate bitset,
#     search match starts, emit replacements, compare outputs, and hash states
#   * lane 0 only owns the unavoidable ordered-rule control flow
#   * the candidate index is the Nim design in GPU-friendly form: dense 1-token
#     heads + sparse open-addressed 2-token heads + per-rule linked lists; v38
#     adds a no-false-negative 3-gram Bloom filter for literal rules.
#
# Candidate construction is exact (no Bloom false positives): each state token
# / adjacent pair activates only rules whose chosen necessary anchor is present.

_MPS_TRAJECTORY_LIB = None
_GP_TG_SIZE = 32
_GP_MAX_RULES = 2048
_GP_RULE_WORDS = (_GP_MAX_RULES + 31) // 32
_GP_INDEX_TOKENS = VOCAB + 1  # 0..511 plus OOV=512

_MPS_TRAJECTORY_SOURCE = (gpu_replace._MPS_FUSED_SOURCE if gpu_replace is not None else "") + r'''
constant int GP_TG_SIZE = 64;
constant int GP_MAX_RULES = 2048;
typedef short gp_token_t;
constant int GP_RULE_WORDS = 64;
constant int GP_INDEX_TOKENS = 513;
constant int GP_META_WIDTH = 7;
constant int GP_ANCHOR_WIDTH = 5;
constant int GP_PAIR_BLOOM_WORDS = 128;

inline uint gp_mix32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline int gp_pair_key(int a, int b) {
    return a * GP_INDEX_TOKENS + b;
}

// v45: packed-short2 literal compare. v38: a contiguous literal triple is also a necessary condition.  We do not
// build a second inverted index for triples; instead its exact state presence is
// summarized in the existing monotone Bloom prefilter.  513^3 < INT_MAX.
inline int gp_triple_key(int a, int b, int c) {
    return (a * GP_INDEX_TOKENS + b) * GP_INDEX_TOKENS + c;
}

// Fold high token bits into low table-address bits. Multiplication alone
// followed by a power-of-two mask preserved the low-bit structure of a*513+b.
inline uint gp_pair_hash(uint key) {
    key ^= key >> 16;
    key *= 0x7feb352du;
    key ^= key >> 15;
    key *= 0x846ca68bu;
    key ^= key >> 16;
    return key;
}

inline int gp_pair_head_lookup(
    const device int* pair_keys,
    const device int* pair_heads,
    int genome,
    int pair_capacity,
    int key)
{
    if (pair_capacity <= 0) return 0;
    int base = genome * pair_capacity;
    uint slot = gp_pair_hash(uint(key)) & uint(pair_capacity - 1);
    for (int probes = 0; probes < pair_capacity; ++probes) {
        int stored = pair_keys[base + int(slot)];
        if (stored == -1) return 0;
        if (stored == key) return pair_heads[base + int(slot)];
        slot = (slot + 1u) & uint(pair_capacity - 1);
    }
    return 0;
}

inline void gp_pair_bloom_add(threadgroup atomic_uint* pair_bloom, int key) {
    uint bit_index = gp_pair_hash(uint(key)) & uint(GP_PAIR_BLOOM_WORDS * 32 - 1);
    atomic_fetch_or_explicit(
        &pair_bloom[bit_index >> 5], 1u << (bit_index & 31), memory_order_relaxed);
}

inline bool gp_pair_bloom_maybe(threadgroup atomic_uint* pair_bloom, int key) {
    uint bit_index = gp_pair_hash(uint(key)) & uint(GP_PAIR_BLOOM_WORDS * 32 - 1);
    uint bits = atomic_load_explicit(&pair_bloom[bit_index >> 5], memory_order_relaxed);
    return (bits & (1u << (bit_index & 31))) != 0u;
}

inline void gp_mark_candidate(threadgroup atomic_uint* candidate_bits, int pid) {
    if (pid < 0 || pid >= GP_MAX_RULES) return;
    atomic_fetch_or_explicit(
        &candidate_bits[pid >> 5], 1u << (pid & 31), memory_order_relaxed);
}

inline bool gp_candidate_marked(threadgroup atomic_uint* candidate_bits, int pid) {
    if (pid < 0 || pid >= GP_MAX_RULES) return false;
    uint bits = atomic_load_explicit(&candidate_bits[pid >> 5], memory_order_relaxed);
    return (bits & (1u << (pid & 31))) != 0u;
}

inline void gp_build_candidates(
    const device gp_token_t* state,
    int base,
    int n,
    int genome,
    int rule_count,
    int start_rule,
    const device int* head1,
    const device int* next1,
    const device int* pair_keys,
    const device int* pair_heads,
    const device int* next2,
    const device int* always_bits,
    int pair_capacity,
    uint tid,
    uint tg_size,
    threadgroup atomic_uint* candidate_bits,
    threadgroup atomic_uint* token_seen,
    threadgroup atomic_uint* pair_bloom,
    threadgroup uint* hash_part1,
    threadgroup uint* hash_part2,
    threadgroup uint* hash_out)
{
    int words = (rule_count + 31) >> 5;
    for (int w = (start_rule >> 5) + int(tid); w < words; w += int(tg_size)) {
        uint seed = uint(always_bits[genome * GP_RULE_WORDS + w]);
        atomic_store_explicit(&candidate_bits[w], seed, memory_order_relaxed);
    }
    for (int w = int(tid); w < 17; w += int(tg_size)) {
        atomic_store_explicit(&token_seen[w], 0u, memory_order_relaxed);
    }
    for (int w = int(tid); w < GP_PAIR_BLOOM_WORDS; w += int(tg_size)) {
        atomic_store_explicit(&pair_bloom[w], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int h1base = genome * GP_INDEX_TOKENS;
    int nrbase = genome * rule_count;
    uint state_h1 = 0u;
    uint state_h2 = 0u;

    for (int i = int(tid); i < n; i += int(tg_size)) {
        int a = int(state[base + i]);
        uint hv = uint(a);
        state_h1 ^= gp_mix32(hv ^ (uint(i) * 0x9e3779b9u));
        state_h2 += gp_mix32(hv + (uint(i) * 0x85ebca6bu) + 0x27d4eb2du);
        if (a >= 0 && a < GP_INDEX_TOKENS) {
            int word = a >> 5;
            uint bit = 1u << (a & 31);
            uint old = atomic_fetch_or_explicit(
                &token_seen[word], bit, memory_order_relaxed);
            // Only one lane traverses each one-token bucket.
            if ((old & bit) == 0u) {
                int p = head1[h1base + a];
                // Host inserts in increasing rule order: these chains descend.
                // Earlier rules cannot run again before the next sweep.
                while (p > start_rule) {
                    int pid = p - 1;
                    gp_mark_candidate(candidate_bits, pid);
                    p = next1[nrbase + pid];
                }
            }
        }

        if (i + 1 < n) {
            int b = state[base + i + 1];
            if (a >= 0 && a < GP_INDEX_TOKENS &&
                b >= 0 && b < GP_INDEX_TOKENS) {
                int key = gp_pair_key(a, b);
                gp_pair_bloom_add(pair_bloom, key);
                // v38: fold exact contiguous triples into the same monotone
                // Bloom. Pair/triple hash collisions are harmless false positives.
                if (i + 2 < n) {
                    int c = int(state[base + i + 2]);
                    if (c >= 0 && c < GP_INDEX_TOKENS)
                        gp_pair_bloom_add(pair_bloom, gp_triple_key(a, b, c));
                }
                int p = gp_pair_head_lookup(
                    pair_keys, pair_heads, genome, pair_capacity, key);
                // Exact duplicate suppression without a separate pair cache:
                // every rule belongs to only one anchor bucket. If the head of
                // this pair chain is already marked, a previous lane traversing
                // the same pair has (or concurrently will have) marked the whole
                // descending chain before the barrier at the end of this build.
                // This removes the pathological repeated-pair chain walks that
                // appear once evolved states become long/repetitive.
                if (p > start_rule && !gp_candidate_marked(candidate_bits, p - 1)) {
                    while (p > start_rule) {
                        int pid = p - 1;
                        gp_mark_candidate(candidate_bits, pid);
                        p = next2[nrbase + pid];
                    }
                }
            }
        }
    }
    hash_part1[tid] = state_h1;
    hash_part2[tid] = state_h2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint a = gp_mix32(uint(n) ^ 0x243f6a88u);
        uint b = gp_mix32(uint(n) ^ 0x9e3779b9u);
        for (uint q = 0; q < tg_size; ++q) {
            a ^= hash_part1[q];
            b += hash_part2[q];
        }
        hash_out[0] = a;
        hash_out[1] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// v23: extend the current round's monotone candidate set after a rewrite.
// We intentionally DO NOT clear candidate_bits or token_seen.  A candidate that
// was valid earlier in the round may become stale, but stale bits are harmless:
// the matcher rechecks the rule before applying it.  The only correctness risk
// would be a false negative, so we only need to OR in anchors that become newly
// present.  Because start_rule only increases, every rule is still attempted at
// most once per round.
inline void gp_extend_candidates(
    const device gp_token_t* state,
    int base,
    int n,
    int genome,
    int rule_count,
    int start_rule,
    const device int* head1,
    const device int* next1,
    const device int* pair_keys,
    const device int* pair_heads,
    const device int* next2,
    int pair_capacity,
    uint tid,
    uint tg_size,
    threadgroup atomic_uint* candidate_bits,
    threadgroup atomic_uint* token_seen,
    threadgroup atomic_uint* pair_bloom)
{
    int h1base = genome * GP_INDEX_TOKENS;
    int nrbase = genome * rule_count;

    for (int i = int(tid); i < n; i += int(tg_size)) {
        int a = state[base + i];
        if (a >= 0 && a < GP_INDEX_TOKENS) {
            int word = a >> 5;
            uint bit = 1u << (a & 31);
            uint old = atomic_fetch_or_explicit(
                &token_seen[word], bit, memory_order_relaxed);
            // If this token existed in any earlier state in this round, its full
            // anchor chain was already ORed into candidate_bits.
            if ((old & bit) == 0u) {
                int p = head1[h1base + a];
                while (p > start_rule) {
                    int pid = p - 1;
                    gp_mark_candidate(candidate_bits, pid);
                    p = next1[nrbase + pid];
                }
            }
        }

        if (i + 1 < n) {
            int b = state[base + i + 1];
            if (a >= 0 && a < GP_INDEX_TOKENS &&
                b >= 0 && b < GP_INDEX_TOKENS) {
                int key = gp_pair_key(a, b);
                gp_pair_bloom_add(pair_bloom, key);
                // v38: fold exact contiguous triples into the same monotone
                // Bloom. Pair/triple hash collisions are harmless false positives.
                if (i + 2 < n) {
                    int c = int(state[base + i + 2]);
                    if (c >= 0 && c < GP_INDEX_TOKENS)
                        gp_pair_bloom_add(pair_bloom, gp_triple_key(a, b, c));
                }
                int p = gp_pair_head_lookup(
                    pair_keys, pair_heads, genome, pair_capacity, key);
                // A marked head means this pair's entire descending chain has
                // already been inserted.  Otherwise this is a newly seen pair.
                if (p > start_rule && !gp_candidate_marked(candidate_bits, p - 1)) {
                    while (p > start_rule) {
                        int pid = p - 1;
                        gp_mark_candidate(candidate_bits, pid);
                        p = next2[nrbase + pid];
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Add one exact anchor value to the monotone candidate set.  These helpers
// are used by the literal-rewrite local update, avoiding an O(state_length)
// rescan after the common wildcard-free rewrite path.
inline void gp_add_token_anchor(
    int a,
    int genome,
    int rule_count,
    int start_rule,
    const device int* head1,
    const device int* next1,
    threadgroup atomic_uint* candidate_bits,
    threadgroup atomic_uint* token_seen)
{
    if (a < 0 || a >= GP_INDEX_TOKENS) return;
    int word = a >> 5;
    uint bit = 1u << (a & 31);
    uint old = atomic_fetch_or_explicit(&token_seen[word], bit, memory_order_relaxed);
    if ((old & bit) != 0u) return;
    int p = head1[genome * GP_INDEX_TOKENS + a];
    int nrbase = genome * rule_count;
    while (p > start_rule) {
        int pid = p - 1;
        gp_mark_candidate(candidate_bits, pid);
        p = next1[nrbase + pid];
    }
}

inline void gp_add_pair_anchor(
    int a,
    int b,
    int genome,
    int rule_count,
    int start_rule,
    const device int* pair_keys,
    const device int* pair_heads,
    const device int* next2,
    int pair_capacity,
    threadgroup atomic_uint* candidate_bits,
    threadgroup atomic_uint* pair_bloom)
{
    if (a < 0 || a >= GP_INDEX_TOKENS || b < 0 || b >= GP_INDEX_TOKENS) return;
    int key = gp_pair_key(a, b);
    gp_pair_bloom_add(pair_bloom, key);
    int p = gp_pair_head_lookup(pair_keys, pair_heads, genome, pair_capacity, key);
    if (p <= start_rule || gp_candidate_marked(candidate_bits, p - 1)) return;
    int nrbase = genome * rule_count;
    while (p > start_rule) {
        int pid = p - 1;
        gp_mark_candidate(candidate_bits, pid);
        p = next2[nrbase + pid];
    }
}

// v38: add the triple starting at q to the monotone state Bloom.  This is
// performance-only filtering; stale/colliding bits can only admit extra work.
inline void gp_add_triple_bloom_at(
    const device gp_token_t* state,
    int base,
    int n,
    int q,
    threadgroup atomic_uint* pair_bloom)
{
    if (q < 0 || q + 2 >= n) return;
    int a = int(state[base + q]);
    int b = int(state[base + q + 1]);
    int c = int(state[base + q + 2]);
    if (a < 0 || a >= GP_INDEX_TOKENS ||
        b < 0 || b >= GP_INDEX_TOKENS ||
        c < 0 || c >= GP_INDEX_TOKENS) return;
    gp_pair_bloom_add(pair_bloom, gp_triple_key(a, b, c));
}

// v45: index of the lowest set bit in a non-zero 32-bit match word.
// Kept local to the literal fast path so we can scan only successful starts.
inline int gp_match_lsb32(uint bits) {
    int b = 0;
    uint x = bits;
    if ((x & 0x0000ffffu) == 0u) { b += 16; x >>= 16; }
    if ((x & 0x000000ffu) == 0u) { b += 8;  x >>= 8;  }
    if ((x & 0x0000000fu) == 0u) { b += 4;  x >>= 4;  }
    if ((x & 0x00000003u) == 0u) { b += 2;  x >>= 2;  }
    if ((x & 0x00000001u) == 0u) { b += 1; }
    return b;
}

// v50: sparse popcount for the len1 1->1 fast path.
inline int gp_popcount32_sparse(uint bits) {
    int c = 0;
    while (bits != 0u) { bits &= bits - 1u; c += 1; }
    return c;
}

inline void gp_extend_literal_candidates_local(
    const device gp_token_t* new_state,
    int base,
    int new_n,
    int old_n,
    const device gp_token_t* replacement,
    int replacement_len,
    int pattern_len,
    device atomic_uint* match_bits,
    device int* selected_starts,
    int selected_count,
    int genome,
    int rule_count,
    int start_rule,
    const device int* head1,
    const device int* next1,
    const device int* pair_keys,
    const device int* pair_heads,
    const device int* next2,
    int pair_capacity,
    uint tid,
    uint tg_size,
    threadgroup atomic_uint* candidate_bits,
    threadgroup atomic_uint* token_seen,
    threadgroup atomic_uint* pair_bloom)
{
    // Replacement-internal tokens/pairs are independent of occurrence count.
    for (int k = int(tid); k < replacement_len; k += int(tg_size)) {
        gp_add_token_anchor(replacement[k], genome, rule_count, start_rule,
            head1, next1, candidate_bits, token_seen);
        if (k + 1 < replacement_len) {
            gp_add_pair_anchor(replacement[k], replacement[k + 1], genome,
                rule_count, start_rule, pair_keys, pair_heads, next2,
                pair_capacity, candidate_bits, pair_bloom);
        }
        if (k + 2 < replacement_len) {
            int a = int(replacement[k]);
            int b = int(replacement[k + 1]);
            int c = int(replacement[k + 2]);
            if (a >= 0 && a < GP_INDEX_TOKENS && b >= 0 && b < GP_INDEX_TOKENS &&
                c >= 0 && c < GP_INDEX_TOKENS)
                gp_pair_bloom_add(pair_bloom, gp_triple_key(a, b, c));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int delta = replacement_len - pattern_len;
    if (pattern_len == 1 && replacement_len == 1) {
        // v50: coordinates do not shift, so the match bitset is the exact
        // changed-position list. Visit only hits and do not materialise the
        // old per-q prefix-count array.
        int match_words = (old_n + 31) >> 5;
        for (int w = int(tid); w < match_words; w += int(tg_size)) {
            uint bits = atomic_load_explicit(&match_bits[w], memory_order_relaxed);
            while (bits != 0u) {
                int b = gp_match_lsb32(bits);
                bits &= bits - 1u;
                int out0 = (w << 5) + b;
                if (out0 >= old_n) continue;
                gp_add_triple_bloom_at(new_state, base, new_n, out0 - 2, pair_bloom);
                gp_add_triple_bloom_at(new_state, base, new_n, out0 - 1, pair_bloom);
                int after3 = out0 + 1;
                gp_add_triple_bloom_at(new_state, base, new_n, after3 - 2, pair_bloom);
                gp_add_triple_bloom_at(new_state, base, new_n, after3 - 1, pair_bloom);
                if (out0 > 0)
                    gp_add_pair_anchor(new_state[base + out0 - 1], new_state[base + out0],
                        genome, rule_count, start_rule, pair_keys, pair_heads, next2,
                        pair_capacity, candidate_bits, pair_bloom);
                int after = out0 + 1;
                if (after < new_n)
                    gp_add_pair_anchor(new_state[base + after - 1], new_state[base + after],
                        genome, rule_count, start_rule, pair_keys, pair_heads, next2,
                        pair_capacity, candidate_bits, pair_bloom);
            }
        }
    } else if (pattern_len == 1) {
        // v52: length-changing len1 local extension now consumes the match
        // bitset word-by-word instead of rescanning every q in the old state.
        // selected_starts[w] is the number of hits in all preceding words.
        // Because gp_match_lsb32 visits set bits in ascending order, the exact
        // per-hit prefix count is just `before++`; no per-q bit test or popcount
        // is needed here.  This preserves the v51 coordinates exactly.
        int match_words = (old_n + 31) >> 5;
        for (int w = int(tid); w < match_words; w += int(tg_size)) {
            uint bits = atomic_load_explicit(&match_bits[w], memory_order_relaxed);
            int before = selected_starts[w];
            while (bits != 0u) {
                int b = gp_match_lsb32(bits);
                bits &= bits - 1u;
                int q = (w << 5) + b;
                if (q >= old_n) continue;
                int out0 = q + before * delta;
                before += 1;
                // Any new triple must be internal to the replacement or cross one
                // of its two boundaries. Internal triples were added above.
                gp_add_triple_bloom_at(new_state, base, new_n, out0 - 2, pair_bloom);
                gp_add_triple_bloom_at(new_state, base, new_n, out0 - 1, pair_bloom);
                int after3 = out0 + replacement_len;
                gp_add_triple_bloom_at(new_state, base, new_n, after3 - 2, pair_bloom);
                gp_add_triple_bloom_at(new_state, base, new_n, after3 - 1, pair_bloom);
                if (replacement_len > 0) {
                    if (out0 > 0)
                        gp_add_pair_anchor(new_state[base + out0 - 1], new_state[base + out0],
                            genome, rule_count, start_rule, pair_keys, pair_heads, next2,
                            pair_capacity, candidate_bits, pair_bloom);
                    int after = out0 + replacement_len;
                    if (after < new_n)
                        gp_add_pair_anchor(new_state[base + after - 1], new_state[base + after],
                            genome, rule_count, start_rule, pair_keys, pair_heads, next2,
                            pair_capacity, candidate_bits, pair_bloom);
                } else if (out0 > 0 && out0 < new_n) {
                    gp_add_pair_anchor(new_state[base + out0 - 1], new_state[base + out0],
                        genome, rule_count, start_rule, pair_keys, pair_heads, next2,
                        pair_capacity, candidate_bits, pair_bloom);
                }
            }
        }
    } else {
        for (int j = int(tid); j < selected_count; j += int(tg_size)) {
            int out0 = int(selected_starts[j]) + j * delta;
            gp_add_triple_bloom_at(new_state, base, new_n, out0 - 2, pair_bloom);
            gp_add_triple_bloom_at(new_state, base, new_n, out0 - 1, pair_bloom);
            int after3 = out0 + replacement_len;
            gp_add_triple_bloom_at(new_state, base, new_n, after3 - 2, pair_bloom);
            gp_add_triple_bloom_at(new_state, base, new_n, after3 - 1, pair_bloom);
            if (replacement_len > 0) {
                if (out0 > 0)
                    gp_add_pair_anchor(new_state[base + out0 - 1], new_state[base + out0],
                        genome, rule_count, start_rule, pair_keys, pair_heads, next2,
                        pair_capacity, candidate_bits, pair_bloom);
                int after = out0 + replacement_len;
                if (after < new_n)
                    gp_add_pair_anchor(new_state[base + after - 1], new_state[base + after],
                        genome, rule_count, start_rule, pair_keys, pair_heads, next2,
                        pair_capacity, candidate_bits, pair_bloom);
            } else if (out0 > 0 && out0 < new_n) {
                gp_add_pair_anchor(new_state[base + out0 - 1], new_state[base + out0],
                    genome, rule_count, start_rule, pair_keys, pair_heads, next2,
                    pair_capacity, candidate_bits, pair_bloom);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void gp_extend_generic_candidates_local(
    const device gp_token_t* new_state,
    int base,
    int new_n,
    device int* changed_regions,
    int region_count,
    int genome,
    int rule_count,
    int start_rule,
    const device int* head1,
    const device int* next1,
    const device int* pair_keys,
    const device int* pair_heads,
    const device int* next2,
    int pair_capacity,
    uint tid,
    uint tg_size,
    threadgroup atomic_uint* candidate_bits,
    threadgroup atomic_uint* token_seen,
    threadgroup atomic_uint* pair_bloom)
{
    // Generic/wildcard rewrites may transform captures, but every genuinely new
    // token/pair must lie inside emitted replacement regions or on their two
    // boundaries. Untouched copied spans preserve their internal anchors even
    // when they shift in output coordinates. Scan only those exact regions.
    for (int j = 0; j < region_count; ++j) {
        int packed = changed_regions[j];
        int lo = (packed >> 16) & 0xffff;
        int hi = packed & 0xffff;
        lo = max(0, min(new_n, lo));
        hi = max(lo, min(new_n, hi));

        for (int q = lo + int(tid); q < hi; q += int(tg_size)) {
            int a = new_state[base + q];
            gp_add_token_anchor(a, genome, rule_count, start_rule,
                head1, next1, candidate_bits, token_seen);
            if (q + 1 < hi) {
                gp_add_pair_anchor(a, new_state[base + q + 1], genome,
                    rule_count, start_rule, pair_keys, pair_heads, next2,
                    pair_capacity, candidate_bits, pair_bloom);
            }
            if (q + 2 < hi)
                gp_add_triple_bloom_at(new_state, base, new_n, q, pair_bloom);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            gp_add_triple_bloom_at(new_state, base, new_n, lo - 2, pair_bloom);
            gp_add_triple_bloom_at(new_state, base, new_n, lo - 1, pair_bloom);
            gp_add_triple_bloom_at(new_state, base, new_n, hi - 2, pair_bloom);
            gp_add_triple_bloom_at(new_state, base, new_n, hi - 1, pair_bloom);
            if (hi > lo) {
                if (lo > 0)
                    gp_add_pair_anchor(new_state[base + lo - 1], new_state[base + lo],
                        genome, rule_count, start_rule, pair_keys, pair_heads, next2,
                        pair_capacity, candidate_bits, pair_bloom);
                if (hi < new_n)
                    gp_add_pair_anchor(new_state[base + hi - 1], new_state[base + hi],
                        genome, rule_count, start_rule, pair_keys, pair_heads, next2,
                        pair_capacity, candidate_bits, pair_bloom);
            } else if (lo > 0 && lo < new_n) {
                // Deletion: only the newly adjacent boundary pair can be novel.
                gp_add_pair_anchor(new_state[base + lo - 1], new_state[base + lo],
                    genome, rule_count, start_rule, pair_keys, pair_heads, next2,
                    pair_capacity, candidate_bits, pair_bloom);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline int gp_next_candidate(
    threadgroup atomic_uint* candidate_bits,
    int start_rule,
    int rule_count)
{
    if (start_rule < 0) start_rule = 0;
    if (start_rule >= rule_count) return -1;
    int words = (rule_count + 31) >> 5;
    int w = start_rule >> 5;
    int bit0 = start_rule & 31;
    uint bits = atomic_load_explicit(&candidate_bits[w], memory_order_relaxed);
    bits &= (0xffffffffu << uint(bit0));
    while (w < words) {
        if (bits != 0u) {
            // Find the first set bit without visiting 32 absent rules.
            int b = 0;
            uint x = bits;
            if ((x & 0x0000ffffu) == 0u) { b += 16; x >>= 16; }
            if ((x & 0x000000ffu) == 0u) { b += 8;  x >>= 8;  }
            if ((x & 0x0000000fu) == 0u) { b += 4;  x >>= 4;  }
            if ((x & 0x00000003u) == 0u) { b += 2;  x >>= 2;  }
            if ((x & 0x00000001u) == 0u) { b += 1; }
            int r = (w << 5) + b;
            return (r < rule_count) ? r : -1;
        }
        ++w;
        if (w >= words) break;
        bits = atomic_load_explicit(&candidate_bits[w], memory_order_relaxed);
    }
    return -1;
}

// v45: compare literal regions two int16 tokens at a time.  packed_short2
// has scalar alignment, so this remains valid for odd token offsets where a
// device uint* load would be misaligned.  The scalar tail handles odd lengths.
inline bool gp_literal_region_equal_packed2(
    const device gp_token_t* input,
    int input_base,
    const device gp_token_t* pattern,
    int begin,
    int end)
{
    int k = begin;
    for (; k + 1 < end; k += 2) {
        const device packed_short2* ip =
            (const device packed_short2*)(input + input_base + k);
        const device packed_short2* pp =
            (const device packed_short2*)(pattern + k);
        packed_short2 iv = *ip;
        packed_short2 pv = *pp;
        if (iv[0] != pv[0] || iv[1] != pv[1]) return false;
    }
    if (k < end && input[input_base + k] != pattern[k]) return false;
    return true;
}

inline bool gp_literal_match_at(
    const device gp_token_t* input,
    int input_base,
    int n,
    const device gp_token_t* pattern,
    int pattern_len,
    int start)
{
    if (start < 0 || pattern_len <= 0 || start + pattern_len > n) return false;
    // v45: this helper is hot in wc1 prefix/suffix discovery as well as the
    // generic leading-literal path. Reuse the exact packed region compare.
    return gp_literal_region_equal_packed2(
        input, input_base + start, pattern, 0, pattern_len);
}

// Match semantics mirror gpu_replace.match_at, but this form avoids writing
// capture arrays while 64 lanes search independent start positions.
inline bool gp_match_finish_only(
    const device gp_token_t* input,
    int input_base,
    int n,
    const device gp_token_t* pattern,
    int pattern_len,
    int start,
    thread int& finish)
{
    int pos = start;
    int p = 0;
    int wc = 0;

    while (p < pattern_len && pattern[p] >= 0) {
        if (pos >= n || input[input_base + pos] != pattern[p]) return false;
        ++pos; ++p;
    }

    while (p < pattern_len) {
        if (pattern[p] >= 0 || wc >= 16) return false;
        ++p; ++wc;
        int lit_begin = p;
        while (p < pattern_len && pattern[p] >= 0) ++p;
        int lit_len = p - lit_begin;
        if (lit_len == 0) {
            if (p >= pattern_len) pos = n;
            continue;
        }
        int found = -1;
        int last = n - lit_len;
        for (int cand = pos; cand <= last; ++cand) {
            bool ok = true;
            for (int k = 0; k < lit_len; ++k) {
                if (input[input_base + cand + k] != pattern[lit_begin + k]) {
                    ok = false; break;
                }
            }
            if (ok) { found = cand; break; }
        }
        if (found < 0) return false;
        pos = found + lit_len;
    }
    finish = pos;
    return finish > start;
}

inline bool gp_match_capture_selected(
    const device gp_token_t* input,
    int input_base,
    int n,
    const device gp_token_t* pattern,
    int pattern_len,
    int start,
    threadgroup int* cap_start,
    threadgroup int* cap_len,
    thread int& finish)
{
    int pos = start;
    int p = 0;
    int wc = 0;
    while (p < pattern_len && pattern[p] >= 0) {
        if (pos >= n || input[input_base + pos] != pattern[p]) return false;
        ++pos; ++p;
    }
    while (p < pattern_len) {
        if (pattern[p] >= 0 || wc >= 16) return false;
        ++p;
        int ci = wc++;
        int cs = pos;
        int lit_begin = p;
        while (p < pattern_len && pattern[p] >= 0) ++p;
        int lit_len = p - lit_begin;
        cap_start[ci] = cs;
        if (lit_len == 0) {
            if (p >= pattern_len) {
                cap_len[ci] = n - pos;
                pos = n;
            } else {
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
                    ok = false; break;
                }
            }
            if (ok) { found = cand; break; }
        }
        if (found < 0) return false;
        cap_len[ci] = found - pos;
        pos = found + lit_len;
    }
    finish = pos;
    return pos > start;
}


// Fast path for wildcard-free pattern + all-literal replacement on short states.
// All candidate starts are matched in parallel once; lane 0 performs only the
// greedy non-overlap selection.  Emission then maps output positions in
// parallel, avoiding the generic path's repeated leftmost-search/barrier cycle.
inline void gp_apply_literal_fast(
    device gp_token_t* src,
    device gp_token_t* dst,
    int base,
    int n,
    int capacity,
    int logical_max_output,
    const device gp_token_t* pattern,
    int pattern_len,
    const device gp_token_t* replacement,
    int replacement_len,
    int anchor_kind,
    int anchor_off,
    uint tid,
    uint tg_size,
    threadgroup int* ctrl,
    device atomic_uint* match_bits,
    device int* selected_starts,
    threadgroup int* lane_counts)
{
    if (tid == 0) {
        for (int i = 0; i < 15; ++i) ctrl[i] = 0;
        ctrl[5] = n;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // Flags alias dst; selected positions live in a separate reusable buffer.
    // No length threshold or ushort position/count truncation.
    if (n <= 0 || pattern_len <= 0) return;

    // Single-token literal patterns are extremely common after evolution.
    // For ANY replacement length, every input position is an independent
    // selected/non-selected segment.  Build one prefix count on lane 0, then
    // emit those disjoint segments cooperatively.  This removes the old
    // O(next_n * log(matches)) inverse mapping for 1 -> N expansions.
    if (pattern_len == 1) {
        int pv = pattern[0];
        int match_words = (n + 31) >> 5;
        // v45: 32 starts share one uint. Clear only the words that are live.
        for (int w = int(tid); w < match_words; w += int(tg_size))
            atomic_store_explicit(&match_bits[w], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // Failures write nothing. Only actual hits perform a relaxed OR, so the
        // hot overwhelmingly-failing path no longer stores one int per start.
        for (int q = int(tid); q < n; q += int(tg_size)) {
            if (src[base + q] == pv) {
                atomic_fetch_or_explicit(
                    &match_bits[q >> 5], 1u << (q & 31), memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // v50: 1->1 never changes coordinates. Count sparse match words and
        // mutate only set bits; no O(n) lane-0 prefix scan or selected_starts
        // materialisation is needed. The bitset remains live for exact local
        // candidate extension after return.
        if (replacement_len == 1) {
            if (tid == 0) {
                int count = 0;
                for (int w = 0; w < match_words; ++w) {
                    uint bits = atomic_load_explicit(&match_bits[w], memory_order_relaxed);
                    count += gp_popcount32_sparse(bits);
                }
                ctrl[4] = count;
                if (count > 0) {
                    ctrl[0] = 1;
                    ctrl[1] = 1;
                    ctrl[5] = n;
                    ctrl[2] = (replacement[0] != pv) ? 1 : 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ctrl[1] == 0 || ctrl[2] == 0) return;

            int rv = replacement[0];
            for (int w = int(tid); w < match_words; w += int(tg_size)) {
                uint bits = atomic_load_explicit(&match_bits[w], memory_order_relaxed);
                while (bits != 0u) {
                    int b = gp_match_lsb32(bits);
                    bits &= bits - 1u;
                    int q = (w << 5) + b;
                    if (q < n) src[base + q] = rv;
                }
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (tid == 0) ctrl[14] = 1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return;
        }

        // v51: length-changing 1-token rules need only a prefix per 32-start
        // word, not one prefix value per q.  selected_starts[w] stores the
        // number of hits in all preceding words.  Each lane reconstructs its
        // exact per-q prefix from the sparse bits before q inside that word.
        if (tid == 0) {
            int count = 0;
            for (int w = 0; w < match_words; ++w) {
                selected_starts[w] = count;
                uint bits = atomic_load_explicit(&match_bits[w], memory_order_relaxed);
                count += gp_popcount32_sparse(bits);
            }
            ctrl[4] = count;
            if (count > 0) {
                ctrl[0] = 1;
                ctrl[1] = 1;
                int next_n_fast = n + count * (replacement_len - 1);
                if (next_n_fast < 0 || next_n_fast > capacity ||
                    next_n_fast > logical_max_output) {
                    ctrl[3] = 1;
                } else {
                    ctrl[5] = next_n_fast;
                    // This branch excludes 1->1, so any selected hit changes
                    // output length and therefore changes the state.
                    ctrl[2] = 1;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (ctrl[1] == 0 || ctrl[3] != 0 || ctrl[2] == 0) return;

        int delta1 = replacement_len - 1;
        for (int q = int(tid); q < n; q += int(tg_size)) {
            int w = q >> 5;
            int bit = q & 31;
            uint word_bits = atomic_load_explicit(&match_bits[w], memory_order_relaxed);
            uint before_mask = (bit == 0) ? 0u : ((1u << uint(bit)) - 1u);
            int before = selected_starts[w] + gp_popcount32_sparse(word_bits & before_mask);
            bool hit = ((word_bits >> uint(bit)) & 1u) != 0u;
            int out0 = q + before * delta1;
            if (hit) {
                for (int k = 0; k < replacement_len; ++k)
                    dst[base + out0 + k] = replacement[k];
            } else {
                dst[base + out0] = src[base + q];
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        return;
    }

    int last = n - pattern_len;
    if (last < 0) return;

    int match_words = (last + 32) >> 5; // ceil((last + 1) / 32)
    for (int w = int(tid); w < match_words; w += int(tg_size))
        atomic_store_explicit(&match_bits[w], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (int cand = int(tid); cand <= last; cand += int(tg_size)) {
        bool anchor_ok = true;
        // v28: the host candidate index already chose a rare necessary anchor.
        // For wildcard-free rules its pattern offset is fixed, so use that same
        // rare token/pair to reject starts before touching the rest of the
        // pattern.  This is exact: a real match must contain this anchor at the
        // recorded offset.
        if (anchor_kind == 2 && anchor_off >= 0 && anchor_off + 1 < pattern_len) {
            int q = cand + anchor_off;
            if (q >= 0 && q + 1 < n) {
                // v45: one packed int16x2 load per side instead of two scalar
                // loads/comparisons.  This check runs for every candidate start.
                const device packed_short2* ip =
                    (const device packed_short2*)(src + base + q);
                const device packed_short2* pp =
                    (const device packed_short2*)(pattern + anchor_off);
                packed_short2 iv = *ip;
                packed_short2 pv = *pp;
                anchor_ok = (iv[0] == pv[0] && iv[1] == pv[1]);
            } else {
                anchor_ok = false;
            }
        } else if (anchor_kind == 1 && anchor_off >= 0 && anchor_off < pattern_len) {
            int q = cand + anchor_off;
            anchor_ok = (q >= 0 && q < n && src[base + q] == pattern[anchor_off]);
        }

        bool ok = false;
        if (anchor_ok) {
            // v45: preserve the exact anchor-skip semantics, but compare the
            // unchecked prefix/suffix in packed int16 pairs.  No extra state
            // scan, barrier, index or Bloom work is introduced.
            if (anchor_kind == 2 && anchor_off >= 0 && anchor_off + 1 < pattern_len) {
                ok = gp_literal_region_equal_packed2(
                         src, base + cand, pattern, 0, anchor_off) &&
                     gp_literal_region_equal_packed2(
                         src, base + cand, pattern, anchor_off + 2, pattern_len);
            } else if (anchor_kind == 1 && anchor_off >= 0 && anchor_off < pattern_len) {
                ok = gp_literal_region_equal_packed2(
                         src, base + cand, pattern, 0, anchor_off) &&
                     gp_literal_region_equal_packed2(
                         src, base + cand, pattern, anchor_off + 1, pattern_len);
            } else {
                ok = gp_literal_region_equal_packed2(
                    src, base + cand, pattern, 0, pattern_len);
            }
        }
        if (ok) {
            atomic_fetch_or_explicit(
                &match_bits[cand >> 5], 1u << (cand & 31), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (tid == 0) {
        int count = 0;
        int next_allowed = 0;
        // v45: load each 32-start word once and visit only successful starts.
        for (int w = 0; w < match_words; ++w) {
            uint bits = atomic_load_explicit(&match_bits[w], memory_order_relaxed);
            while (bits != 0u) {
                int b = gp_match_lsb32(bits);
                bits &= bits - 1u;
                int cand = (w << 5) + b;
                if (cand > last) break;
                if (cand < next_allowed) continue;
                selected_starts[count++] = cand;
                next_allowed = cand + pattern_len;
            }
        }
        ctrl[4] = count;
        if (count > 0 && ctrl[3] == 0) {
            ctrl[0] = 1;
            ctrl[1] = 1;
            int next_n_fast = n + count * (replacement_len - pattern_len);
            if (next_n_fast < 0 || next_n_fast > capacity ||
                next_n_fast > logical_max_output) {
                ctrl[3] = 1;
            } else {
                ctrl[5] = next_n_fast;
                bool differs = replacement_len != pattern_len;
                if (!differs) {
                    for (int k = 0; k < pattern_len; ++k) {
                        if (replacement[k] != pattern[k]) { differs = true; break; }
                    }
                }
                ctrl[2] = differs ? 1 : 0;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (ctrl[1] == 0 || ctrl[3] != 0 || ctrl[2] == 0) return;

    int count = ctrl[4];
    int next_n = ctrl[5];
    int delta = replacement_len - pattern_len;

    // v28 equal-length literal rewrite: all selected match locations were
    // computed before emission and are disjoint, so writing the replacement
    // directly into the current state cannot affect matching.  This removes an
    // O(n) state copy for roughly the `same=` fraction reported by the profiler.
    if (delta == 0) {
        for (int j = int(tid); j < count; j += int(tg_size)) {
            int s0 = int(selected_starts[j]);
            for (int k = 0; k < replacement_len; ++k)
                src[base + s0 + k] = replacement[k];
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (tid == 0) ctrl[14] = 1; // state changed in-place
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return;
    }

    // When there are many matches (the expensive late-generation case),
    // emitting by disjoint source segments is O(n + matches*replacement_len)
    // and avoids a binary search for every output token.  With very few
    // matches, the old output-parallel inverse map keeps more lanes occupied.
    if (count >= max(4, int(tg_size) >> 2)) {
        for (int seg = int(tid); seg <= count; seg += int(tg_size)) {
            if (seg < count) {
                int s0 = int(selected_starts[seg]);
                int prev_end = (seg == 0) ? 0 :
                    (int(selected_starts[seg - 1]) + pattern_len);
                int gap_len = s0 - prev_end;
                int out_gap = prev_end + seg * delta;
                for (int k = 0; k < gap_len; ++k)
                    dst[base + out_gap + k] = src[base + prev_end + k];
                int out_rep = s0 + seg * delta;
                for (int k = 0; k < replacement_len; ++k)
                    dst[base + out_rep + k] = replacement[k];
            } else {
                int prev_end = (count == 0) ? 0 :
                    (int(selected_starts[count - 1]) + pattern_len);
                int out_tail = prev_end + count * delta;
                for (int k = 0; k < n - prev_end; ++k)
                    dst[base + out_tail + k] = src[base + prev_end + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        return;
    }

    // Few-match fallback: map final output positions independently.
    for (int q = int(tid); q < next_n; q += int(tg_size)) {
        int lo = 0;
        int hi = count;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            int out_start = int(selected_starts[mid]) + mid * delta;
            if (out_start <= q) lo = mid + 1;
            else hi = mid;
        }
        int j = lo - 1;
        if (j >= 0) {
            int out_start = int(selected_starts[j]) + j * delta;
            if (q < out_start + replacement_len) {
                dst[base + q] = replacement[q - out_start];
            } else {
                int src_q = q - (j + 1) * delta;
                dst[base + q] = src[base + src_q];
            }
        } else {
            dst[base + q] = src[base + q];
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}

// Fast path for exactly one wildcard and an all-literal replacement.
//
// The generic matcher repeatedly searches the suffix literal run from many
// candidate starts.  For the overwhelmingly common one-wildcard rule, build a
// nearest-suffix table once (O(n)), derive every candidate finish in parallel,
// then perform the exact same greedy non-overlap selection.  This preserves the
// wildcard semantics while avoiding the generic path's repeated suffix scans and
// per-match barrier loop.  changed_regions is rewritten at the end to the output
// replacement spans expected by the incremental candidate updater.
inline int gp_wc1_transform_value(int v, int kind) {
    if (kind == 1) return v;
    if (kind == 3) return (v >= 0 && v < 512) ? ((v + 1) & 511) : v;
    if (kind == 4) return (v >= 0 && v < 512) ? ((v + 511) & 511) : v;
    if (kind == 5) return (v >= 0 && v < 512) ? ((v * 2) & 511) : v;
    if (kind == 6) return (v >= 0 && v < 512) ? (v / 2) : v;
    return v;
}

// v32 keeps the v30 fast path for exactly one wildcard, including one optional capture
// reference/transform in the replacement.  The non-expanding repair guarantees
// that a one-wildcard rule references that capture at most once, so each emitted
// replacement is [literal prefix] + [optional transformed capture] + [literal
// suffix].  Match discovery is the v27 nearest-suffix algorithm; emission is
// cooperative and never enters the barrier-heavy generic wildcard matcher.
// v32: suffix occurrences for the one-wildcard matcher are represented as a
// compact bitset in aux instead of a full nearest-suffix table.  This removes
// the serial O(n) reverse fill previously paid for every wc1 candidate.
inline int gp_first_set_bit32(uint bits) {
    int b = 0;
    uint x = bits;
    if ((x & 0x0000ffffu) == 0u) { b += 16; x >>= 16; }
    if ((x & 0x000000ffu) == 0u) { b += 8;  x >>= 8;  }
    if ((x & 0x0000000fu) == 0u) { b += 4;  x >>= 4;  }
    if ((x & 0x00000003u) == 0u) { b += 2;  x >>= 2;  }
    if ((x & 0x00000001u) == 0u) { b += 1; }
    return b;
}

inline int gp_wc1_next_suffix(
    const device int* suffix_bits,
    int words,
    int pos,
    int n)
{
    if (pos < 0) pos = 0;
    if (pos >= n || words <= 0) return -1;
    int w = pos >> 5;
    int bit0 = pos & 31;
    uint bits = uint(suffix_bits[w]) & (0xffffffffu << uint(bit0));
    while (w < words) {
        if (bits != 0u) {
            int q = (w << 5) + gp_first_set_bit32(bits);
            return (q < n) ? q : -1;
        }
        ++w;
        if (w >= words) break;
        bits = uint(suffix_bits[w]);
    }
    return -1;
}

inline void gp_apply_wc1_fast(
    device gp_token_t* src_rw,
    device gp_token_t* dst,
    int base,
    int n,
    int capacity,
    int logical_max_output,
    const device gp_token_t* pattern,
    int pattern_len,
    const device gp_token_t* replacement,
    int replacement_len,
    uint tid,
    uint tg_size,
    threadgroup int* ctrl,
    device int* changed_regions,
    device int* aux,
    threadgroup atomic_uint* diff_flag)
{
    const device gp_token_t* src = src_rw;
    if (tid == 0) {
        for (int i = 0; i < 15; ++i) ctrl[i] = 0;
        ctrl[5] = n;
        ctrl[13] = -1; // wildcard position
        ctrl[11] = -1; // replacement capture opcode position, -1 => literal-only
        ctrl[10] = 0;  // replacement capture transform kind
        atomic_store_explicit(diff_flag, 0u, memory_order_relaxed);
        for (int k = 0; k < pattern_len; ++k) {
            if (pattern[k] < 0) { ctrl[13] = k; break; }
        }
        // Under the host non-expanding invariant there is at most one negative
        // replacement token when wc==1. Decode it once per firing.
        for (int k = 0; k < replacement_len; ++k) {
            int t = replacement[k];
            if (t >= 0) continue;
            ctrl[11] = k;
            if (t >= -15) ctrl[10] = 1;          // capture
            else if (t >= -31) ctrl[10] = 7;     // unsupported sort (defensive)
            else if (t >= -47) ctrl[10] = 2;     // reverse
            else if (t >= -111) {
                int off = -t - 48;
                int bank = off / 16;
                ctrl[10] = 3 + bank;             // +1,-1,*2,//2
            } else ctrl[10] = 7;
            break;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (n <= 0 || pattern_len <= 0 || ctrl[13] < 0 || ctrl[10] == 7) return;

    int wild_pos = ctrl[13];
    int rep_cap_pos = ctrl[11];
    int rep_kind = ctrl[10];
    int prefix_len = wild_pos;
    int suffix_len = pattern_len - wild_pos - 1;

    // v32: build a suffix-occurrence bitset in parallel.  Each lane owns
    // whole 32-position words, so there are no atomics and no serial reverse
    // nearest-table fill.  aux is reused for selected output spans after match
    // discovery has finished.
    int suffix_words = (n + 31) >> 5;
    if (suffix_len > 0) {
        for (int w = int(tid); w < suffix_words; w += int(tg_size)) {
            uint bits = 0u;
            int q0 = w << 5;
            int q1 = min(n, q0 + 32);
            for (int q = q0; q < q1; ++q) {
                bool ok = (q + suffix_len <= n) &&
                    gp_literal_match_at(src, base, n,
                        pattern + wild_pos + 1, suffix_len, q);
                if (ok) bits |= 1u << uint(q - q0);
            }
            aux[w] = int(bits);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    // Prefix-leading rules can begin at many positions.  dst is scratch here.
    if (prefix_len > 0) {
        for (int cand = int(tid); cand < n; cand += int(tg_size)) {
            int finish = -1;
            if (cand + prefix_len <= n &&
                gp_literal_match_at(src, base, n, pattern, prefix_len, cand)) {
                if (suffix_len == 0) {
                    finish = n;
                } else {
                    int pos = cand + prefix_len;
                    int ss = gp_wc1_next_suffix(aux, suffix_words, pos, n);
                    if (ss >= 0) finish = ss + suffix_len;
                }
            }
            changed_regions[cand] = (finish > cand) ? finish : -1;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        int count = 0;
        int scan_pos = 0;
        int next_n = n;
        int delta = 0;
        int emitted_total = 0;

        while (scan_pos < n) {
            int start = -1;
            int finish = -1;
            if (prefix_len == 0) {
                start = scan_pos;
                if (suffix_len == 0) {
                    finish = n;
                } else {
                    int ss = gp_wc1_next_suffix(aux, suffix_words, scan_pos, n);
                    if (ss >= 0) finish = ss + suffix_len;
                }
            } else {
                for (int cand = scan_pos; cand < n; ++cand) {
                    int f = changed_regions[cand];
                    if (f > cand) { start = cand; finish = f; break; }
                }
            }
            if (start < 0 || finish <= start) break;

            int cap_start_q = start + prefix_len;
            int cap_end_q = finish - suffix_len;
            int cap_n = max(0, cap_end_q - cap_start_q);
            int emit_n = (rep_cap_pos < 0) ? replacement_len
                                           : (replacement_len - 1 + cap_n);
            if (emit_n > 0xffff) {
                ctrl[3] = 1; break;
            }
            // Preserve selected input spans in changed_regions itself.  The
            // compacted prefix [0..count) is always behind scan_pos, so it cannot
            // overwrite a candidate entry that the greedy scan may still read.
            // This also leaves the suffix bitset in aux intact until selection
            // is completely finished (required for leading-wildcard rules).
            changed_regions[count] =
                ((start & 0xffff) << 16) | (finish & 0xffff);
            ++count;
            emitted_total += emit_n;
            next_n += emit_n - (finish - start);
            delta += emit_n - (finish - start);
            if (next_n < 0 || next_n > capacity || next_n > logical_max_output) {
                ctrl[3] = 1; break;
            }
            scan_pos = finish;
            if (prefix_len == 0 && suffix_len == 0) break;
        }

        // Selection is complete; convert compacted input spans into output
        // spans in aux. changed_regions already contains the selected input spans.
        if (ctrl[3] == 0) {
            int out_delta = 0;
            for (int j = 0; j < count; ++j) {
                int in_pack = changed_regions[j];
                int start = (in_pack >> 16) & 0xffff;
                int finish = in_pack & 0xffff;
                int cap_start_q = start + prefix_len;
                int cap_end_q = finish - suffix_len;
                int cap_n = max(0, cap_end_q - cap_start_q);
                int emit_n = (rep_cap_pos < 0) ? replacement_len
                                               : (replacement_len - 1 + cap_n);
                int out_start = start + out_delta;
                if (out_start < 0 || out_start > 0xffff || emit_n > 0xffff) {
                    ctrl[3] = 1; break;
                }
                aux[j] = ((out_start & 0xffff) << 16) | (emit_n & 0xffff);
                out_delta += emit_n - (finish - start);
            }
        }

        ctrl[4] = count;
        ctrl[12] = emitted_total;
        if (count > 0 && ctrl[3] == 0) {
            ctrl[0] = 1;
            ctrl[1] = 1;
            ctrl[5] = next_n;
            if (next_n != n)
                atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (ctrl[1] == 0 || ctrl[3] != 0) return;

    int count = ctrl[4];
    int next_n = ctrl[5];
    bool need_compare = (next_n == n);

    // v31: a one-wildcard replacement that references its capture exactly once
    // has a per-match length delta independent of capture length.  When that
    // delta is zero, every selected output span is exactly the same length as
    // its consumed input span.  Preserve the original selected spans in the
    // opposite state buffer, then rewrite only those spans in-place.  Gaps and
    // the tail never move, so we avoid an O(n) ping-pong copy for this rewrite.
    bool wc1_inplace = (rep_cap_pos >= 0 && next_n == n &&
                        (replacement_len - 1) == (prefix_len + suffix_len));
    if (wc1_inplace) {
        // Snapshot all disjoint matched spans before any in-place write.  One
        // barrier is sufficient because spans do not overlap.
        for (int j = 0; j < count; ++j) {
            int in_pack = changed_regions[j];
            int start = (in_pack >> 16) & 0xffff;
            int finish = in_pack & 0xffff;
            for (int k = int(tid); k < finish - start; k += int(tg_size))
                dst[base + start + k] = src[base + start + k];
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        for (int j = 0; j < count; ++j) {
            int in_pack = changed_regions[j];
            int start = (in_pack >> 16) & 0xffff;
            int finish = in_pack & 0xffff;
            int cap_start_q = start + prefix_len;
            int cap_end_q = finish - suffix_len;
            int cap_n = max(0, cap_end_q - cap_start_q);
            int out_pack = aux[j];
            int out_start = (out_pack >> 16) & 0xffff;

            for (int k = int(tid); k < rep_cap_pos; k += int(tg_size)) {
                int y = int(replacement[k]);
                int oq = out_start + k;
                if (y != int(dst[base + oq]))
                    atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
                src_rw[base + oq] = gp_token_t(y);
            }
            for (int k = int(tid); k < cap_n; k += int(tg_size)) {
                int src_k = (rep_kind == 2) ? (cap_start_q + cap_n - 1 - k)
                                            : (cap_start_q + k);
                int v = int(dst[base + src_k]);
                int y = gp_wc1_transform_value(v, rep_kind);
                int oq = out_start + rep_cap_pos + k;
                if (y != int(dst[base + oq]))
                    atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
                src_rw[base + oq] = gp_token_t(y);
            }
            int tail_literals = replacement_len - rep_cap_pos - 1;
            for (int k = int(tid); k < tail_literals; k += int(tg_size)) {
                int y = int(replacement[rep_cap_pos + 1 + k]);
                int oq = out_start + rep_cap_pos + cap_n + k;
                if (y != int(dst[base + oq]))
                    atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
                src_rw[base + oq] = gp_token_t(y);
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // For delta==0 the output replacement spans are the same intervals as
        // the consumed input spans; keep the exact format expected by the local
        // candidate updater.
        if (tid == 0) {
            ctrl[2] = atomic_load_explicit(diff_flag, memory_order_relaxed) != 0u ? 1 : 0;
            ctrl[14] = 1; // current state remains in src_rw
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return;
    }

    // Each selected match owns one disjoint input gap and one disjoint output
    // replacement span.  All lanes cooperate on every span; unlike the generic
    // matcher this requires no per-capture barrier loop.
    for (int j = 0; j < count; ++j) {
        int in_pack = changed_regions[j];
        int start = (in_pack >> 16) & 0xffff;
        int finish = in_pack & 0xffff;
        int out_pack = aux[j];
        int out_start = (out_pack >> 16) & 0xffff;
        int emit_n = out_pack & 0xffff;
        int prev_finish = (j == 0) ? 0 : (changed_regions[j - 1] & 0xffff);
        int prev_out_end = (j == 0) ? 0 :
            (((aux[j - 1] >> 16) & 0xffff) + (aux[j - 1] & 0xffff));
        int gap_len = start - prev_finish;

        for (int k = int(tid); k < gap_len; k += int(tg_size)) {
            int v = src[base + prev_finish + k];
            int oq = prev_out_end + k;
            dst[base + oq] = v;
            if (need_compare && v != src[base + oq])
                atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
        }

        if (rep_cap_pos < 0) {
            for (int k = int(tid); k < replacement_len; k += int(tg_size)) {
                int v = replacement[k];
                int oq = out_start + k;
                dst[base + oq] = v;
                if (need_compare && v != src[base + oq])
                    atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
            }
        } else {
            int cap_start_q = start + prefix_len;
            int cap_end_q = finish - suffix_len;
            int cap_n = max(0, cap_end_q - cap_start_q);

            for (int k = int(tid); k < rep_cap_pos; k += int(tg_size)) {
                int v = replacement[k];
                int oq = out_start + k;
                dst[base + oq] = v;
                if (need_compare && v != src[base + oq])
                    atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
            }
            for (int k = int(tid); k < cap_n; k += int(tg_size)) {
                int src_k = (rep_kind == 2) ? (cap_start_q + cap_n - 1 - k)
                                            : (cap_start_q + k);
                int v = src[base + src_k];
                int y = gp_wc1_transform_value(v, rep_kind);
                int oq = out_start + rep_cap_pos + k;
                dst[base + oq] = y;
                if (need_compare && y != src[base + oq])
                    atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
            }
            int tail_literals = replacement_len - rep_cap_pos - 1;
            for (int k = int(tid); k < tail_literals; k += int(tg_size)) {
                int v = replacement[rep_cap_pos + 1 + k];
                int oq = out_start + rep_cap_pos + cap_n + k;
                dst[base + oq] = v;
                if (need_compare && v != src[base + oq])
                    atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
            }
            (void)emit_n;
        }
    }

    // Tail after the last consumed match.
    int last_finish = changed_regions[count - 1] & 0xffff;
    int last_out_end = ((aux[count - 1] >> 16) & 0xffff) +
                       (aux[count - 1] & 0xffff);
    int tail_len = n - last_finish;
    for (int k = int(tid); k < tail_len; k += int(tg_size)) {
        int v = src[base + last_finish + k];
        int oq = last_out_end + k;
        dst[base + oq] = v;
        if (need_compare && v != src[base + oq])
            atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // Exact local candidate updater expects output replacement spans.
    for (int j = int(tid); j < count; j += int(tg_size)) {
        int packed = aux[j];
        int out_start = (packed >> 16) & 0xffff;
        int emit_n = packed & 0xffff;
        changed_regions[j] =
            ((out_start & 0xffff) << 16) | ((out_start + emit_n) & 0xffff);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0)
        ctrl[2] = atomic_load_explicit(diff_flag, memory_order_relaxed) != 0u ? 1 : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Cooperative replacement. Shared ctrl layout:
// 0 matched, 1 applied, 2 changed, 3 overflow, 4 selected, 5 next_n,
// 6 scan_pos, 7 prev, 8 out_pos, 9 best_s, 10 best_f,
// 11 segment_start, 12 segment_len, 13 segment_kind, 14 segment_aux.
inline void gp_apply_rule_coop(
    const device gp_token_t* src,
    device gp_token_t* dst,
    int base,
    int n,
    int capacity,
    int logical_max_output,
    const device gp_token_t* pattern,
    int pattern_len,
    const device gp_token_t* replacement,
    int replacement_len,
    int wildcard_count,
    int has_leading_literal,
    int replacement_all_literal,
    uint tid,
    uint tg_size,
    threadgroup int* ctrl,
    threadgroup int* best_s_lane,
    threadgroup int* best_f_lane,
    threadgroup int* cap_start,
    threadgroup int* cap_len,
    threadgroup atomic_uint* diff_flag,
    device int* changed_regions)
{
    if (tid == 0) {
        for (int i = 0; i < 15; ++i) ctrl[i] = 0;
        ctrl[5] = n;
        ctrl[6] = 0;
        ctrl[7] = 0;
        ctrl[8] = 0;
        atomic_store_explicit(diff_flag, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n <= 0 || pattern_len <= 0) return;

    int cached_tile = -1;
    while (true) {
        int scan_pos = ctrl[6];
        if (scan_pos >= n || ctrl[3] != 0) break;

        if (has_leading_literal != 0) {
            // Evaluate a small contiguous window once and keep its results for
            // subsequent non-overlapping matches. Never scan the full suffix
            // independently in every lane for every selected match.
            int tile = (scan_pos / int(tg_size)) * int(tg_size);
            while (tile < n) {
                if (tile != cached_tile) {
                    int cand = tile + int(tid);
                    int f = -1;
                    if (cand < n && src[base + cand] == pattern[0]) {
                        if (wildcard_count == 0) {
                            if (gp_literal_match_at(src, base, n, pattern, pattern_len, cand))
                                f = cand + pattern_len;
                        } else {
                            int finish = -1;
                            if (gp_match_finish_only(src, base, n, pattern, pattern_len, cand, finish))
                                f = finish;
                        }
                    }
                    best_f_lane[tid] = f;
                    cached_tile = tile;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (tid == 0) {
                    ctrl[9] = -1;
                    for (int q = max(0, scan_pos - tile); q < int(tg_size); ++q) {
                        if (best_f_lane[q] > tile + q) {
                            ctrl[9] = tile + q;
                            ctrl[10] = best_f_lane[q];
                            break;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (ctrl[9] >= 0) break;
                tile += int(tg_size);
            }
            if (tid == 0 && ctrl[9] >= 0 && wildcard_count > 0 && replacement_all_literal == 0) {
                int finish = -1;
                bool ok = gp_match_capture_selected(src, base, n, pattern, pattern_len,
                    ctrl[9], cap_start, cap_len, finish);
                if (!ok) ctrl[9] = -1;
                else ctrl[10] = finish;
            }
        } else if (tid == 0) {
            // Leading wildcard rules only try scan_pos. Get captures in the
            // same pass instead of matching the same suffix twice.
            int finish = -1;
            bool ok = gp_match_capture_selected(src, base, n, pattern, pattern_len,
                scan_pos, cap_start, cap_len, finish);
            ctrl[9] = ok ? scan_pos : -1;
            ctrl[10] = finish;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ctrl[9] < 0) break;

        if (tid == 0) {
            ctrl[0] = 1;
            ctrl[1] = 1;
            ctrl[4] += 1;
            ctrl[11] = ctrl[7];
            ctrl[12] = ctrl[9] - ctrl[7];
            if (ctrl[8] + ctrl[12] > capacity ||
                ctrl[8] + ctrl[12] > logical_max_output) ctrl[3] = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ctrl[3] != 0) break;

        // Copy untouched prefix in parallel.
        int seg_start = ctrl[11];
        int seg_len = ctrl[12];
        int out0 = ctrl[8];
        bool prefix_shifted = (out0 != seg_start);
        for (int q = int(tid); q < seg_len; q += int(tg_size)) {
            int v = src[base + seg_start + q];
            dst[base + out0 + q] = v;
            if (prefix_shifted) {
                int oq = out0 + q;
                if (oq >= n || v != src[base + oq])
                    atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) ctrl[8] += seg_len;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int replacement_out_start = ctrl[8];

        // Fast path for the overwhelmingly common all-literal replacement:
        // reserve once, copy the entire replacement cooperatively, and pay one
        // barrier instead of one barrier per output token.
        if (replacement_all_literal != 0) {
            if (tid == 0) {
                if (ctrl[8] + replacement_len > capacity ||
                    ctrl[8] + replacement_len > logical_max_output) ctrl[3] = 1;
                ctrl[11] = ctrl[8];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ctrl[3] != 0) break;
            int rout = ctrl[11];
            for (int rr = int(tid); rr < replacement_len; rr += int(tg_size)) {
                int v = replacement[rr];
                int oq = rout + rr;
                dst[base + oq] = v;
                if (oq >= n || v != src[base + oq])
                    atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) ctrl[8] += replacement_len;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else for (int rr = 0; rr < replacement_len; ++rr) {
            int t = replacement[rr];
            if (t >= 0) {
                if (tid == 0) {
                    if (ctrl[8] >= capacity || ctrl[8] >= logical_max_output)
                        ctrl[3] = 1;
                    else {
                        int oq = ctrl[8];
                        dst[base + oq] = t;
                        if (oq >= n || t != src[base + oq])
                            atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
                        ctrl[8] += 1;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (ctrl[3] != 0) break;
                continue;
            }

            if (wildcard_count <= 0) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                continue;
            }

            if (tid == 0) {
                int kind = 0, ci = 0;
                if (t >= -15) {
                    kind = 1; ci = -t - 1;
                } else if (t >= -31) {
                    // sort($capture) is rejected by the host path.
                    ctrl[3] = 1;
                } else if (t >= -47) {
                    kind = 2; ci = -t - 32;
                } else if (t >= -111) {
                    int off = -t - 48;
                    int bank = off / 16;
                    ci = off - bank * 16;
                    kind = 3 + bank;
                } else {
                    ctrl[3] = 1;
                }
                if (ci >= wildcard_count) ci = 0;
                ctrl[13] = kind;
                ctrl[14] = ci;
                ctrl[11] = cap_start[ci];
                ctrl[12] = cap_len[ci];
                if (ctrl[8] + ctrl[12] > capacity ||
                    ctrl[8] + ctrl[12] > logical_max_output) ctrl[3] = 1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ctrl[3] != 0) break;

            int kind = ctrl[13];
            int cs = ctrl[11];
            int cl = ctrl[12];
            out0 = ctrl[8];
            for (int q = int(tid); q < cl; q += int(tg_size)) {
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
                    if (v < 512) y = (v >= 0) ? (v / 2) : -(((-v) + 1) / 2);
                }
                int oq = out0 + q;
                dst[base + oq] = y;
                if (oq >= n || y != src[base + oq])
                    atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) ctrl[8] += cl;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (ctrl[3] != 0) break;

        if (tid == 0) {
            int region_id = ctrl[4] - 1;
            // capacity/logical max output are <= 32768 in this evaluator, so
            // two 16-bit endpoints fit in one scratch int. Empty regions encode
            // deletions and are still useful for their new boundary pair.
            changed_regions[region_id] =
                ((replacement_out_start & 0xffff) << 16) | (ctrl[8] & 0xffff);
            ctrl[7] = ctrl[10];
            ctrl[6] = ctrl[10];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (ctrl[3] == 0 && ctrl[0] != 0) {
        if (tid == 0) {
            ctrl[11] = ctrl[7];
            ctrl[12] = n - ctrl[7];
            if (ctrl[8] + ctrl[12] > capacity ||
                ctrl[8] + ctrl[12] > logical_max_output) ctrl[3] = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ctrl[3] == 0) {
            int seg_start = ctrl[11];
            int seg_len = ctrl[12];
            int out0 = ctrl[8];
            bool tail_shifted = (out0 != seg_start);
            for (int q = int(tid); q < seg_len; q += int(tg_size)) {
                int v = src[base + seg_start + q];
                int oq = out0 + q;
                dst[base + oq] = v;
                if (tail_shifted && (oq >= n || v != src[base + oq]))
                    atomic_store_explicit(diff_flag, 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (tid == 0) {
                ctrl[8] += seg_len;
                ctrl[5] = ctrl[8];
                if (ctrl[8] != n || atomic_load_explicit(diff_flag, memory_order_relaxed) != 0u)
                    ctrl[2] = 1;
                int local_scan_tokens = 0;
                for (int j = 0; j < ctrl[4]; ++j) {
                    int packed = changed_regions[j];
                    int lo = (packed >> 16) & 0xffff;
                    int hi = packed & 0xffff;
                    if (hi > lo) local_scan_tokens += hi - lo;
                }
                ctrl[12] = local_scan_tokens;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}

inline void gp_hash_state_coop(
    const device gp_token_t* x,
    int base,
    int n,
    uint tid,
    uint tg_size,
    threadgroup uint* part1,
    threadgroup uint* part2,
    threadgroup uint* out_hash)
{
    uint h1 = 0u;
    uint h2 = 0u;
    for (int i = int(tid); i < n; i += int(tg_size)) {
        uint v = uint(x[base + i]);
        uint m1 = gp_mix32(v ^ (uint(i) * 0x9e3779b9u));
        uint m2 = gp_mix32(v + (uint(i) * 0x85ebca6bu) + 0x27d4eb2du);
        h1 ^= m1;
        h2 += m2;
    }
    part1[tid] = h1;
    part2[tid] = h2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint a = gp_mix32(uint(n) ^ 0x243f6a88u);
        uint b = gp_mix32(uint(n) ^ 0x9e3779b9u);
        for (uint q = 0; q < tg_size; ++q) {
            a ^= part1[q];
            b += part2[q];
        }
        out_hash[0] = a;
        out_hash[1] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// v23: build the per-genome inverted candidate index on the GPU.  v22 did
// this with Python nested loops and an open-addressed hash build every generation,
// which dominated host pack time.  One threadgroup owns one genome; lanes clear
// the tables in parallel and lane 0 performs the ordered insertion so linked lists
// remain strictly descending by rule id (required by the start_rule early stop).
kernel void build_gp_candidate_index_v23_i32(
    const device int* anchors [[buffer(0)]],
    device int* head1 [[buffer(1)]],
    device int* next1 [[buffer(2)]],
    device int* pair_keys [[buffer(3)]],
    device int* pair_heads [[buffer(4)]],
    device int* next2 [[buffer(5)]],
    device int* always_bits [[buffer(6)]],
    constant int& rule_count [[buffer(7)]],
    constant int& genome_count [[buffer(8)]],
    constant int& pair_capacity [[buffer(9)]],
    uint group_u [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    int genome = int(group_u);
    if (genome >= genome_count) return;

    int h1base = genome * GP_INDEX_TOKENS;
    int nrbase = genome * rule_count;
    int pbase = genome * pair_capacity;
    int wbase = genome * GP_RULE_WORDS;

    for (int i = int(tid); i < GP_INDEX_TOKENS; i += int(tg_size))
        head1[h1base + i] = 0;
    for (int r = int(tid); r < rule_count; r += int(tg_size)) {
        next1[nrbase + r] = 0;
        next2[nrbase + r] = 0;
    }
    for (int w = int(tid); w < GP_RULE_WORDS; w += int(tg_size))
        always_bits[wbase + w] = 0;
    for (int i = int(tid); i < pair_capacity; i += int(tg_size)) {
        pair_keys[pbase + i] = -1;
        pair_heads[pbase + i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (tid == 0) {
        for (int r = 0; r < rule_count; ++r) {
            int aq = (nrbase + r) * GP_ANCHOR_WIDTH;
            int kind = anchors[aq + 0];
            int a = anchors[aq + 1];
            int b = anchors[aq + 2];
            if (kind == 0) {
                uint old = uint(always_bits[wbase + (r >> 5)]);
                old |= 1u << uint(r & 31);
                always_bits[wbase + (r >> 5)] = int(old);
            } else if (kind == 1 && a >= 0 && a < GP_INDEX_TOKENS) {
                next1[nrbase + r] = head1[h1base + a];
                head1[h1base + a] = r + 1;
            } else if (kind == 2 && a >= 0 && a < GP_INDEX_TOKENS &&
                       b >= 0 && b < GP_INDEX_TOKENS) {
                int key = gp_pair_key(a, b);
                uint slot = gp_pair_hash(uint(key)) & uint(pair_capacity - 1);
                for (int probes = 0; probes < pair_capacity; ++probes) {
                    int q = pbase + int(slot);
                    int stored = pair_keys[q];
                    if (stored == -1) {
                        pair_keys[q] = key;
                        pair_heads[q] = r + 1;
                        break;
                    }
                    if (stored == key) {
                        next2[nrbase + r] = pair_heads[q];
                        pair_heads[q] = r + 1;
                        break;
                    }
                    slot = (slot + 1u) & uint(pair_capacity - 1);
                }
            } else {
                // Defensive fallback: an invalid anchor must never create a
                // false negative, so treat the rule as always-candidate.
                uint old = uint(always_bits[wbase + (r >> 5)]);
                old |= 1u << uint(r & 31);
                always_bits[wbase + (r >> 5)] = int(old);
            }
        }
    }
}

kernel void evaluate_gp_population_v17_i32(
    device gp_token_t* buffer_a [[buffer(0)]],
    device gp_token_t* buffer_b [[buffer(1)]],
    const device int* raw_inputs [[buffer(2)]],
    const device int* raw_lengths [[buffer(3)]],
    const device int* embeddings [[buffer(4)]],
    const device gp_token_t* token_pool [[buffer(5)]],
    const device int* metadata [[buffer(6)]],
    const device int* rule_ids [[buffer(7)]],
    const device int* head1 [[buffer(8)]],
    const device int* next1 [[buffer(9)]],
    const device int* pair_keys [[buffer(10)]],
    const device int* pair_heads [[buffer(11)]],
    const device int* next2 [[buffer(12)]],
    const device int* always_bits [[buffer(13)]],
    device uchar* feature_rows [[buffer(14)]],
    device int* stat_rows [[buffer(15)]],
    device int* profile_rows [[buffer(16)]],
    device int* seen_hashes [[buffer(17)]],
    constant int& capacity [[buffer(18)]],
    constant int& logical_max_output [[buffer(19)]],
    constant int& raw_stride [[buffer(20)]],
    constant int& sample_count [[buffer(21)]],
    constant int& rule_count [[buffer(22)]],
    constant int& genome_count [[buffer(23)]],
    constant int& global_genome_start [[buffer(24)]],
    constant int& output_genome_start [[buffer(25)]],
    constant int& max_rounds_cap [[buffer(26)]],
    constant int& pair_capacity [[buffer(27)]],
    device int* literal_positions [[buffer(28)]],
    device int* wildcard_aux [[buffer(29)]],
    const device int* anchors [[buffer(30)]],
    uint group_u [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_size > GP_TG_SIZE) return;
    int job = int(group_u);
    int total_jobs = sample_count * genome_count;
    if (job >= total_jobs) return;

    int local_genome = job / sample_count;
    int sample = job - local_genome * sample_count;
    int global_genome = global_genome_start + local_genome;
    int output_genome = output_genome_start + local_genome;
    int base = job * capacity;
    int raw_base = sample * raw_stride;
    int n = raw_lengths[sample];
    int original_n = max(1, n);
    int rounds_limit = min(max_rounds_cap,
        max(1, int(ceil(2.0f * sqrt(float(original_n))))));

    threadgroup atomic_uint candidate_bits[GP_RULE_WORDS];
    threadgroup atomic_uint token_seen[17];
    threadgroup atomic_uint pair_bloom[GP_PAIR_BLOOM_WORDS];
    threadgroup atomic_uint diff_flag;
    threadgroup int best_s_lane[GP_TG_SIZE];
    threadgroup int best_f_lane[GP_TG_SIZE];
    threadgroup int cap_start[16];
    threadgroup int cap_len[16];

    threadgroup int ctrl[15];
    threadgroup int shared[16];
    // v28: profiling/statistics are diagnostic outputs. Keep hot-loop
    // increments in threadgroup memory and flush once per trajectory instead
    // of issuing hundreds of device-global writes per job.
    threadgroup int prof_local[15];
    threadgroup int stat_local[4];
    threadgroup uint hash_part1[GP_TG_SIZE];
    threadgroup uint hash_part2[GP_TG_SIZE];
    threadgroup uint hash_out[2];

    int emb_base = global_genome * 256;
    for (int i = int(tid); i < n; i += int(tg_size)) {
        int b = raw_inputs[raw_base + i];
        buffer_a[base + i] = (b >= 0 && b < 256) ? embeddings[emb_base + b] : 512;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    int feature_base = (output_genome * sample_count + sample) * rule_count;
    int stat_base = (output_genome * sample_count + sample) * 4;
    int profile_base = (output_genome * sample_count + sample) * 15;
    if (tid == 0) {
        for (int i = 0; i < 4; ++i) stat_local[i] = 0;
        for (int i = 0; i < 15; ++i) prof_local[i] = 0;
        shared[0] = 1; // current_is_a
        shared[1] = n;
        shared[2] = 0; // stop
        shared[3] = 0; // seen_count
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int genome_rule_base = global_genome * rule_count;

    for (int round_idx = 0; round_idx < rounds_limit; ++round_idx) {
        if (tid == 0) {
            shared[4] = 0; // rewrote
            shared[5] = 0; // overflowed
            shared[6] = 0; // round_state_changed
            shared[7] = 0; // next rule index
            shared[8] = 1; // 1=full build, 2=monotone extension
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        while (true) {
            int current_n = shared[1];
            bool current_is_a = shared[0] != 0;
            device gp_token_t* src_rw = current_is_a ? buffer_a : buffer_b;
            const device gp_token_t* src = src_rw;

            // No candidate build is needed after the last rule or overflow.
            if (shared[7] >= rule_count || shared[5] != 0) break;
            if (shared[8] == 1) {
                gp_build_candidates(
                    src, base, current_n, global_genome, rule_count, shared[7],
                    head1, next1, pair_keys, pair_heads, next2, always_bits,
                    pair_capacity, tid, tg_size, candidate_bits, token_seen, pair_bloom,
                    hash_part1, hash_part2, hash_out);
                if (tid == 0) {
                    // v31: candidate construction already scanned every token.
                    // Fold cycle hashing into that same pass instead of reading
                    // the whole state again at the end of the previous round.
                    bool repeated = false;
                    int seen_count = shared[3];
                    for (int i = 0; i < seen_count; ++i) {
                        int q = (job * max_rounds_cap + i) * 2;
                        if (uint(seen_hashes[q + 0]) == hash_out[0] &&
                            uint(seen_hashes[q + 1]) == hash_out[1]) {
                            repeated = true; break;
                        }
                    }
                    if (repeated) {
                        stat_local[3] = 1;
                        shared[2] = 1;
                    } else if (seen_count < max_rounds_cap) {
                        int q = (job * max_rounds_cap + seen_count) * 2;
                        seen_hashes[q + 0] = int(hash_out[0]);
                        seen_hashes[q + 1] = int(hash_out[1]);
                        shared[3] = seen_count + 1;
                    }
                    shared[8] = 0;
                    prof_local[0] += 1;
                    prof_local[1] += current_n;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (shared[2] != 0) break;
            } else if (shared[8] == 2) {
                gp_extend_candidates(
                    src, base, current_n, global_genome, rule_count, shared[7],
                    head1, next1, pair_keys, pair_heads, next2,
                    pair_capacity, tid, tg_size, candidate_bits, token_seen, pair_bloom);
                if (tid == 0) {
                    shared[8] = 0;
                    prof_local[10] += 1;
                    prof_local[11] += current_n;
                    prof_local[1] += current_n;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                int r = gp_next_candidate(candidate_bits, shared[7], rule_count);
                shared[9] = r;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int r = shared[9];
            if (r < 0 || shared[5] != 0) break;
            if (tid == 0) prof_local[2] += 1;
            // Only lane 0 accesses profile counters; no barrier is needed here.

            int pool_rule = rule_ids[genome_rule_base + r];
            int mq = pool_rule * GP_META_WIDTH;
            const device gp_token_t* pat = token_pool + metadata[mq + 0];
            int pat_len = metadata[mq + 1];
            const device gp_token_t* rep = token_pool + metadata[mq + 2];
            int rep_len = metadata[mq + 3];
            int wc = metadata[mq + 4];
            int leading = metadata[mq + 5];
            int rep_literal = metadata[mq + 6];
            device gp_token_t* dst = current_is_a ? buffer_b : buffer_a;
            int aq = (global_genome * rule_count + r) * GP_ANCHOR_WIDTH;
            int anchor_kind = anchors[aq + 0];
            int anchor_off = anchors[aq + 3];
            int secondary_pair_key = anchors[aq + 4];

            // v38: literal rules may carry a necessary contiguous 3-gram; wildcard
            // rules retain v33's second-pair condition. The shared state Bloom
            // is monotone within the round, so a missing bit
            // is an exact proof that the rule cannot match. Stale/set bits only
            // cause false positives, never false negatives.
            if (secondary_pair_key >= 0 && !gp_pair_bloom_maybe(pair_bloom, secondary_pair_key)) {
                if (tid == 0) {
                    prof_local[14] += 1;
                    shared[7] = r + 1;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                continue;
            }

            bool use_literal_fast = (wc == 0 && rep_literal != 0);
            bool use_wc1_fast = (wc == 1);
            if (use_literal_fast) {
                gp_apply_literal_fast(
                    src_rw, dst, base, current_n, capacity, logical_max_output,
                    pat, pat_len, rep, rep_len, anchor_kind, anchor_off,
                    tid, tg_size, ctrl,
                    (device atomic_uint*)(wildcard_aux + base), literal_positions + base,
                    best_s_lane);
            } else if (use_wc1_fast) {
                gp_apply_wc1_fast(
                    src_rw, dst, base, current_n, capacity, logical_max_output,
                    pat, pat_len, rep, rep_len, tid, tg_size, ctrl,
                    literal_positions + base, wildcard_aux + base, &diff_flag);
            } else {
                gp_apply_rule_coop(
                    src, dst, base, current_n, capacity, logical_max_output,
                    pat, pat_len, rep, rep_len, wc, leading, rep_literal,
                    tid, tg_size, ctrl, best_s_lane, best_f_lane,
                    cap_start, cap_len, &diff_flag, literal_positions + base);
            }

            // v23 fast path: literal rewrites already expose every changed span.
            // Add only replacement/internal/boundary anchors instead of rescanning
            // the whole new state.  This is exact because untouched spans cannot
            // create a previously absent token or adjacent pair.
            if (use_literal_fast && ctrl[1] != 0 && ctrl[2] != 0 && ctrl[3] == 0) {
                const device gp_token_t* literal_new_state = (ctrl[14] != 0) ? src_rw : dst;
                gp_extend_literal_candidates_local(
                    literal_new_state, base, ctrl[5], current_n, rep, rep_len, pat_len,
                    (device atomic_uint*)(wildcard_aux + base),
                    literal_positions + base, ctrl[4], global_genome, rule_count,
                    r + 1, head1, next1, pair_keys, pair_heads, next2,
                    pair_capacity, tid, tg_size, candidate_bits, token_seen, pair_bloom);
            } else if (!use_literal_fast && ctrl[1] != 0 && ctrl[2] != 0 && ctrl[3] == 0) {
                const device gp_token_t* generic_new_state =
                    (use_wc1_fast && ctrl[14] != 0) ? src_rw : dst;
                gp_extend_generic_candidates_local(
                    generic_new_state, base, ctrl[5], literal_positions + base, ctrl[4],
                    global_genome, rule_count, r + 1, head1, next1,
                    pair_keys, pair_heads, next2, pair_capacity, tid, tg_size,
                    candidate_bits, token_seen, pair_bloom);
                if (tid == 0) {
                    prof_local[10] += 1;
                    prof_local[11] += ctrl[12];
                    prof_local[1] += ctrl[12];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                if (ctrl[1] != 0) {
                    uchar old = feature_rows[feature_base + r];
                    if (old < uchar(255)) feature_rows[feature_base + r] = uchar(old + 1);
                    stat_local[1] += 1;
                    prof_local[3] += 1;
                    if (use_literal_fast) {
                        prof_local[4] += 1;
                        if (pat_len == rep_len) prof_local[6] += 1;
                        if (pat_len == 1) prof_local[7] += 1;
                        if (ctrl[2] != 0 && ctrl[14] != 0) prof_local[13] += 1;
                    } else {
                        prof_local[5] += 1;
                        if (use_wc1_fast) {
                            prof_local[12] += 1;
                            if (ctrl[2] != 0 && ctrl[14] != 0) prof_local[13] += 1;
                        }
                    }
                    prof_local[8] += ctrl[4];
                    prof_local[9] += current_n;
                    shared[4] = 1;
                }
                if (ctrl[3] != 0) {
                    stat_local[2] = 1;
                    shared[5] = 1;
                }
                shared[7] = r + 1;
                if (ctrl[1] != 0 && ctrl[2] != 0) {
                    shared[1] = ctrl[5];
                    // v28: an equal-length literal rewrite is emitted directly
                    // into the current buffer after all match starts are fixed.
                    // Keep ownership instead of copying n tokens to the other
                    // ping-pong buffer only to swap immediately back later.
                    if (!((use_literal_fast || use_wc1_fast) && ctrl[14] != 0))
                        shared[0] = current_is_a ? 0 : 1;
                    shared[6] = 1;
                    shared[8] = 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }

        // A repeated state is detected during the full candidate build before
        // any rule of this round executes. Preserve the old end-of-previous-round
        // termination semantics and do not count a phantom extra round.
        if (shared[2] != 0) break;

        if (tid == 0) {
            stat_local[0] = round_idx + 1;
            if (shared[5] != 0 || shared[4] == 0) shared[2] = 1;
            else if (shared[6] == 0) {
                stat_local[3] = 1;
                shared[2] = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shared[2] != 0) break;

    }

    // One diagnostic flush per trajectory.
    if (tid == 0) {
        for (int i = 0; i < 4; ++i) stat_rows[stat_base + i] = stat_local[i];
        for (int i = 0; i < 15; ++i) profile_rows[profile_base + i] = prof_local[i];
    }
}
'''


def _get_mps_trajectory_lib():
    global _MPS_TRAJECTORY_LIB
    if _MPS_TRAJECTORY_LIB is None:
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("torch.mps.compile_shader is unavailable; PyTorch 2.12+ is recommended")
        _MPS_TRAJECTORY_LIB = torch.mps.compile_shader(_MPS_TRAJECTORY_SOURCE)
    return _MPS_TRAJECTORY_LIB


class MpsPopulationEvaluator:
    META_WIDTH = 7

    def __init__(self, sample_count: int, rule_count: int, max_raw_len: int,
                 max_output: int, genome_batch: int = 4,
                 result_chunk: int = 32, progress: bool = True,
                 threadgroup_size: int = _GP_TG_SIZE,
                 pool_max_rules: int = 2_000_000,
                 pool_gc_ratio: float = 1.35,
                 pool_gc_min_dead: int = 150_000):
        if rule_count > _GP_MAX_RULES:
            raise ValueError(
                f"v17 cooperative MPS path supports at most {_GP_MAX_RULES} rules; got {rule_count}"
            )
        if threadgroup_size not in (32, 64):
            raise ValueError("--mps-threadgroup must be 32 or 64")
        self.device = torch.device("mps")
        self.sample_count = int(sample_count)
        self.rule_count = int(rule_count)
        self.raw_stride = max(1, int(max_raw_len))
        # v20: every rule is hard non-expanding, so no rewritten state can ever
        # exceed its raw input length. The old max_output-sized scratch (32,768
        # ints per trajectory by default) wasted >2x memory for max_chunk=15k
        # and amplified unified-memory pressure at genome_batch=32. Keep the
        # logical max_output check, but physically allocate only raw_stride.
        self.capacity = self.raw_stride
        self.max_output = int(max_output)
        if self.max_output > 65535:
            raise ValueError("v25 local generic candidate regions require --max-output <= 65535")
        self.genome_batch = max(1, int(genome_batch))
        self.result_chunk = max(self.genome_batch, int(result_chunk))
        self.progress = bool(progress)
        self.threadgroup_size = int(threadgroup_size)
        self.pool_max_rules = max(self.rule_count, int(pool_max_rules))
        self.pool_gc_ratio = max(1.05, float(pool_gc_ratio))
        self.pool_gc_min_dead = max(0, int(pool_gc_min_dead))
        self.max_rounds = max(1, int(math.ceil(2.0 * math.sqrt(self.raw_stride))))
        if self.max_rounds > 255:
            raise ValueError("uint8 firing counters require ceil(2*sqrt(max_chunk)) <= 255")
        jobs = self.sample_count * self.genome_batch

        self.raw_inputs = torch.zeros(
            (self.sample_count, self.raw_stride), dtype=torch.int32, device=self.device)
        self.raw_lengths = torch.zeros(self.sample_count, dtype=torch.int32, device=self.device)
        self.buffer_a = torch.empty((jobs, self.capacity), dtype=torch.int16, device=self.device)
        self.buffer_b = torch.empty_like(self.buffer_a)
        self.literal_positions = torch.empty((jobs, self.capacity), dtype=torch.int32, device=self.device)
        # v27: one extra int scratch row lets the one-wildcard matcher keep a
        # nearest-suffix table/output-start table without touching the live state.
        self.wildcard_aux = torch.empty((jobs, self.capacity), dtype=torch.int32, device=self.device)
        self.seen = torch.empty((jobs, self.max_rounds, 2), dtype=torch.int32, device=self.device)
        self._raw_byte_counts = np.ones(INPUT_BYTE_COUNT, dtype=np.float64)
        self._raw_pair_counts = np.zeros((INPUT_BYTE_COUNT, INPUT_BYTE_COUNT), dtype=np.int64)
        self._raw_triple_counts: dict[int, int] = {}
        # Exact duplicate-input memoization. Duplicate trajectory rows retain
        # separate regression observations but share one structural GPU result.
        self._sample_alias = np.arange(self.sample_count, dtype=np.int32)
        self.active_sample_count = self.sample_count
        self.unique_input_count = self.sample_count
        self._lib = _get_mps_trajectory_lib()
        self.last_pack_seconds = 0.0
        self.last_pack_cache_rows = 0
        self.last_pack_rebuilt_rows = 0
        self.last_rule_pack_memo_hits = 0
        self.last_rule_pack_memo_misses = 0
        self.last_dispatch_seconds = 0.0

        self.feature_stage = torch.empty(
            (self.result_chunk, self.sample_count, self.rule_count),
            dtype=torch.uint8, device=self.device,
        )
        self.stat_stage = torch.empty(
            (self.result_chunk, self.sample_count, 4), dtype=torch.int32,
            device=self.device,
        )
        self.profile_stage = torch.empty(
            (self.result_chunk, self.sample_count, 15), dtype=torch.int32,
            device=self.device,
        )

        # v23: reusable per-generation candidate-index buffers.  v22 rebuilt
        # head1/next/pair hash tables in Python for every genome every generation.
        # We now upload only (pool_id, embedding, anchor metadata); Metal builds
        # the ordered inverted index in-place before trajectory evaluation.
        self._index_genome_capacity = 0
        self._index_pair_capacity = self._pair_capacity(self.rule_count)
        self._rule_ids_d = None
        self._embeddings_d = None
        self._anchors_d = None
        self._head1_d = None
        self._next1_d = None
        self._next2_d = None
        self._always_d = None
        self._pair_keys_d = None
        self._pair_heads_d = None
        self._host_rule_ids = None
        self._host_embeddings = None
        self._host_anchors = None

        # Global structural rule pool.  Rules are interned by pattern/replacement
        # and remain resident on the GPU across generations.  Only newly created
        # structures are appended/uploaded.
        self._pool_owner = id(self)
        self._pool_key_to_id: dict[tuple, int] = {}
        # (pool_id, anchor_kind, a, b) -> fixed pattern offset.  This avoids
        # rescanning ~675k rule patterns every generation merely to feed the
        # v28 anchor-guided literal matcher.  New structural rules pay once.
        self._anchor_offset_cache: dict[tuple[int, int, int, int], int] = {}
        self._pool_meta_host: list[tuple[int, int, int, int, int, int, int]] = []
        self._pool_tokens_host: list[int] = []
        self._pool_rule_capacity = max(4096, self.rule_count * 2)
        self._pool_token_capacity = max(65536, self.rule_count * 16)
        self._pool_meta_d = torch.empty(
            (self._pool_rule_capacity, self.META_WIDTH), dtype=torch.int32, device=self.device)
        self._pool_tokens_d = torch.empty(
            self._pool_token_capacity, dtype=torch.int16, device=self.device)
        self.last_pool_upload_seconds = 0.0
        self.last_pool_new_rules = 0
        self.last_pool_new_tokens = 0
        self.last_pool_gc_seconds = 0.0
        self.last_pool_gc_triggered = False
        self.last_pool_gc_reason = ""
        self.last_pool_gc_rules_before = 0
        self.last_pool_gc_rules_after = 0
        self.last_pool_gc_tokens_before = 0
        self.last_pool_gc_tokens_after = 0
        self.last_pool_gc_alloc_before = 0
        self.last_pool_gc_alloc_after = 0
        self.pool_gc_count = 0
        self.last_profile: dict[str, float] = {}

        # Unified-memory visibility is important on M1: a large genome_batch
        # multiplies three max_output-sized scratch buffers.  Report it so batch
        # tuning is based on the actual working set rather than "bigger is faster".
        scratch_bytes = int(jobs) * int(self.capacity) * 12
        self.scratch_bytes = scratch_bytes
        if self.progress:
            msg = f"  mps scratch={scratch_bytes / (1024**3):.2f} GiB"
            try:
                rec = int(torch.mps.recommended_max_memory())
                cur = int(torch.mps.current_allocated_memory())
                msg += f" recommended={rec / (1024**3):.2f} GiB allocated={cur / (1024**3):.2f} GiB"
                if rec > 0 and scratch_bytes > rec * 0.30:
                    msg += " [large batch: try --mps-genome-batch 8 or 16]"
            except Exception:
                pass
            print(msg, flush=True)

    def set_inputs(self, inputs: Sequence[Sequence[int]]) -> None:
        # v60 permits a smaller active prefix for the bounded inference GA.  The
        # large training scratch buffers stay allocated, but kernels launch only
        # active_sample_count trajectories, avoiding padded zero-work threadgroups.
        active = len(inputs)
        if active <= 0 or active > self.sample_count:
            raise ValueError(f"expected 1..{self.sample_count} samples, got {active}")
        host = torch.zeros((self.sample_count, self.raw_stride), dtype=torch.int32)
        lengths = torch.zeros(self.sample_count, dtype=torch.int32)
        counts = np.ones(INPUT_BYTE_COUNT, dtype=np.float64)
        pair_counts = np.zeros((INPUT_BYTE_COUNT, INPUT_BYTE_COUNT), dtype=np.int64)
        triple_counts: dict[int, int] = {}
        alias = np.arange(active, dtype=np.int32)
        seen_inputs: dict[tuple[int, ...], int] = {}
        for j, row in enumerate(inputs):
            n = len(row)
            if n > self.raw_stride:
                raise ValueError(f"raw input length {n} exceeds evaluator stride={self.raw_stride}")

            # Preserve anchor-frequency statistics over all observations, even
            # duplicate rows: the regression dataset still contains all of them.
            if n:
                arr = np.asarray(row, dtype=np.int64)
                ok = arr[(arr >= 0) & (arr < INPUT_BYTE_COUNT)]
                if ok.size:
                    counts += np.bincount(ok, minlength=INPUT_BYTE_COUNT)
                if n > 1:
                    a = arr[:-1]
                    b = arr[1:]
                    good = (a >= 0) & (a < INPUT_BYTE_COUNT) & (b >= 0) & (b < INPUT_BYTE_COUNT)
                    if np.any(good):
                        np.add.at(pair_counts, (a[good], b[good]), 1)
                if n > 2:
                    a3 = arr[:-2]
                    b3 = arr[1:-1]
                    c3 = arr[2:]
                    good3 = ((a3 >= 0) & (a3 < INPUT_BYTE_COUNT) &
                             (b3 >= 0) & (b3 < INPUT_BYTE_COUNT) &
                             (c3 >= 0) & (c3 < INPUT_BYTE_COUNT))
                    if np.any(good3):
                        keys3 = ((a3[good3] * INPUT_BYTE_COUNT + b3[good3]) *
                                 INPUT_BYTE_COUNT + c3[good3])
                        uniq3, cnt3 = np.unique(keys3, return_counts=True)
                        for kk, cc in zip(uniq3.tolist(), cnt3.tolist()):
                            triple_counts[int(kk)] = triple_counts.get(int(kk), 0) + int(cc)

            key = tuple(map(int, row))
            rep = seen_inputs.get(key)
            if rep is not None:
                # Zero-length placeholder makes the duplicate threadgroup cheap;
                # its feature/stat/profile rows are replaced by the representative
                # after the batched GPU result copy.
                lengths[j] = 0
                alias[j] = int(rep)
                continue
            seen_inputs[key] = j
            lengths[j] = n
            if n:
                host[j, :n] = torch.as_tensor(row, dtype=torch.int32)

        self.raw_inputs.copy_(host)
        self.raw_lengths.copy_(lengths)
        self._raw_byte_counts = counts
        self._raw_pair_counts = pair_counts
        self._raw_triple_counts = triple_counts
        self._sample_alias = alias
        self.active_sample_count = int(active)
        self.unique_input_count = int(np.sum(alias == np.arange(active, dtype=np.int32)))

    @staticmethod
    def _inverse_embedding(embedding: Sequence[int]):
        return {int(code): byte for byte, code in enumerate(embedding)}

    def _candidate_anchor(self, pattern: Sequence[int], inverse: dict):
        # Match Nim's candidateAnchorForPattern: prefer the rarest adjacent
        # literal pair, otherwise the rarest single literal. Every returned
        # anchor is a necessary condition, so candidate pruning is exact.
        best_pair = None
        best_pair_freq = None
        for a, b in zip(pattern, pattern[1:]):
            ia, ib = int(a), int(b)
            if ia < 0 or ib < 0 or ia >= _GP_INDEX_TOKENS or ib >= _GP_INDEX_TOKENS:
                continue
            ba, bb = inverse.get(ia), inverse.get(ib)
            freq = 0 if ba is None or bb is None else int(self._raw_pair_counts[ba, bb])
            if best_pair is None or freq < best_pair_freq:
                best_pair = (ia, ib)
                best_pair_freq = freq
                if freq == 0:
                    break
        if best_pair is not None:
            return 2, best_pair[0], best_pair[1]

        best = None
        best_freq = None
        for v in pattern:
            iv = int(v)
            if iv < 0 or iv >= _GP_INDEX_TOKENS:
                continue
            byte = inverse.get(iv)
            freq = 0 if byte is None else float(self._raw_byte_counts[byte])
            if best is None or freq < best_freq:
                best, best_freq = iv, freq
                if freq == 0:
                    break
        if best is not None:
            return 1, int(best), -1
        return 0, -1, -1

    def _secondary_pair_filter(self, pattern: Sequence[int], inverse: dict,
                               primary_a: int, primary_b: int) -> int:
        """Return one extra exact Bloom condition, or -1.

        v38: for wildcard-free literal patterns of length >=3, prefer the
        rarest contiguous 3-gram.  Presence of that triple is a much stronger
        condition than a second independent pair and costs no per-rule state
        scan: state triples are folded into the existing monotone Bloom while
        candidate construction already walks the state.  For wildcard patterns
        retain v33's second-pair filter. Bloom collisions/stale bits can only
        admit extra work and therefore cannot change matching semantics.
        """
        literal_only = all(int(v) >= 0 for v in pattern)
        if literal_only and len(pattern) >= 3:
            best_key = -1
            best_freq = None
            seen3 = set()
            for a, b, c in zip(pattern, pattern[1:], pattern[2:]):
                ia, ib, ic = int(a), int(b), int(c)
                if (ia < 0 or ib < 0 or ic < 0 or
                    ia >= _GP_INDEX_TOKENS or ib >= _GP_INDEX_TOKENS or ic >= _GP_INDEX_TOKENS):
                    continue
                key = (ia * _GP_INDEX_TOKENS + ib) * _GP_INDEX_TOKENS + ic
                if key in seen3:
                    continue
                seen3.add(key)
                ba, bb, bc = inverse.get(ia), inverse.get(ib), inverse.get(ic)
                if ba is None or bb is None or bc is None:
                    freq = 0
                else:
                    ext_key = (int(ba) * INPUT_BYTE_COUNT + int(bb)) * INPUT_BYTE_COUNT + int(bc)
                    freq = int(self._raw_triple_counts.get(ext_key, 0))
                if best_key < 0 or freq < best_freq:
                    best_key, best_freq = key, freq
                    if freq == 0:
                        break
            if best_key >= 0:
                return int(best_key)

        primary_key = (int(primary_a) * _GP_INDEX_TOKENS + int(primary_b)
                       if primary_a >= 0 and primary_b >= 0 else -1)
        best_key = -1
        best_freq = None
        seen = set()
        for a, b in zip(pattern, pattern[1:]):
            ia, ib = int(a), int(b)
            if ia < 0 or ib < 0 or ia >= _GP_INDEX_TOKENS or ib >= _GP_INDEX_TOKENS:
                continue
            key = ia * _GP_INDEX_TOKENS + ib
            if key == primary_key or key in seen:
                continue
            seen.add(key)
            ba, bb = inverse.get(ia), inverse.get(ib)
            freq = 0 if ba is None or bb is None else int(self._raw_pair_counts[ba, bb])
            if best_key < 0 or freq < best_freq:
                best_key, best_freq = key, freq
                if freq == 0:
                    break
        return int(best_key)

    @staticmethod
    def _pair_capacity(rule_count: int) -> int:
        cap = 32
        wanted = max(1, int(rule_count)) * 2
        while cap < wanted:
            cap <<= 1
        return cap

    @staticmethod
    def _pair_slot(key: int, cap: int) -> int:
        # Must match gp_pair_hash in Metal, including uint32 wraparound.
        x = int(key) & 0xFFFFFFFF
        x ^= x >> 16
        x = (x * 0x7FEB352D) & 0xFFFFFFFF
        x ^= x >> 15
        x = (x * 0x846CA68B) & 0xFFFFFFFF
        x ^= x >> 16
        return x & (cap - 1)

    @staticmethod
    def _aligned_capacity(need: int, minimum: int, alignment: int, headroom: float = 1.25) -> int:
        target = max(int(minimum), int(math.ceil(max(0, need) * float(headroom))))
        return ((target + alignment - 1) // alignment) * alignment

    @staticmethod
    def _mps_allocated_bytes() -> int:
        try:
            return int(torch.mps.current_allocated_memory())
        except Exception:
            return 0

    def _gc_trigger(self, genomes: Sequence[Genome]) -> str:
        """Return a reason when the persistent pool should be compacted.

        The number of live unique structures is never larger than the number of
        Rule references in the current population.  That gives a cheap upper
        bound on liveness without hashing ~700k structures every generation.
        Actual liveness is measured only once a collection is triggered.
        """
        pool_rules = len(self._pool_meta_host)
        if pool_rules <= 0:
            return ""
        live_ref_upper = sum(len(g.rules) for g in genomes)
        if pool_rules >= self.pool_max_rules:
            return "hard-cap"
        guaranteed_dead = max(0, pool_rules - live_ref_upper)
        gc_ratio = max(1.05, float(getattr(self, "pool_gc_ratio", 1.35)))
        gc_min_dead = max(0, int(getattr(self, "pool_gc_min_dead", 150_000)))
        if (
            live_ref_upper > 0
            and pool_rules >= int(math.ceil(live_ref_upper * gc_ratio))
            and guaranteed_dead >= gc_min_dead
        ):
            return "dead-ratio"
        return ""

    def _trace_live_rule_pool(self, genomes: Sequence[Genome]):
        """Build a dense host pool containing exactly the current live rules.

        Kept separate from the MPS buffer replacement so it can be validated on
        non-Apple machines and so the mark/compact semantics are explicit.
        """
        new_key_to_id: dict[tuple, int] = {}
        new_meta: list[tuple[int, int, int, int, int, int, int]] = []
        new_tokens: list[int] = []
        object_pid: dict[int, int] = {}

        for genome in genomes:
            for rule in genome.rules:
                oid = id(rule)
                cached = object_pid.get(oid)
                if cached is not None:
                    rule.pool_id = int(cached)
                    rule.pool_owner = self._pool_owner
                    continue

                key = (tuple(map(int, rule.pattern)), tuple(map(int, rule.replacement)))
                pid = new_key_to_id.get(key)
                if pid is None:
                    p, rep = rule.pattern, rule.replacement
                    if int(rule.pack_wc) < 0:
                        wc = sum(v < 0 for v in p)
                        if wc > MAX_WILDCARDS:
                            raise ValueError(f"rule has {wc} wildcards; max={MAX_WILDCARDS}")
                        if any(-31 <= v <= -16 for v in rep):
                            raise ValueError("sort($capture) is not supported by the fused MPS path")
                        rule.pack_wc = int(wc)
                        rule.pack_leading = 1 if p and p[0] >= 0 else 0
                        rule.pack_rep_literal = 1 if all(v >= 0 for v in rep) else 0

                    po = len(new_tokens)
                    new_tokens.extend(map(int, p))
                    ro = len(new_tokens)
                    new_tokens.extend(map(int, rep))
                    pid = len(new_meta)
                    new_meta.append((
                        po, len(p), ro, len(rep), int(rule.pack_wc),
                        int(rule.pack_leading), int(rule.pack_rep_literal),
                    ))
                    new_key_to_id[key] = pid
                object_pid[oid] = int(pid)
                rule.pool_id = int(pid)
                rule.pool_owner = self._pool_owner

        return new_key_to_id, new_meta, new_tokens

    def _compact_rule_pool(self, genomes: Sequence[Genome], reason: str) -> None:
        """Trace live rules from ``genomes`` and rebuild host + MPS pool densely.

        This is a real compacting collection: unreachable structural rules are
        removed, live pool ids are rewritten, the old Metal buffers are released,
        the MPS allocator cache is drained, and smaller backing buffers are then
        allocated and repopulated from the compacted host representation.
        """
        t0 = time.perf_counter()
        before_rules = len(self._pool_meta_host)
        before_tokens = len(self._pool_tokens_host)
        before_alloc = self._mps_allocated_bytes()

        new_key_to_id, new_meta, new_tokens = self._trace_live_rule_pool(genomes)

        after_rules = len(new_meta)
        after_tokens = len(new_tokens)
        new_rule_capacity = self._aligned_capacity(after_rules, 4096, 4096)
        new_token_capacity = self._aligned_capacity(after_tokens, 65536, 65536)

        # Everything that can refer to the old MPS buffers is local to completed
        # evaluate_population calls at this point. Synchronize before dropping
        # them so queued kernels cannot outlive their storage.
        try:
            torch.mps.synchronize()
        except Exception:
            pass
        old_meta_d = self._pool_meta_d
        old_tokens_d = self._pool_tokens_d
        del self._pool_meta_d
        del self._pool_tokens_d
        del old_meta_d
        del old_tokens_d

        # Drop the large Python containers before asking both Python and MPS to
        # reclaim cached storage. This runs only at compaction, never per-gen.
        self._pool_key_to_id = new_key_to_id
        self._anchor_offset_cache.clear()
        self._pool_meta_host = new_meta
        self._pool_tokens_host = new_tokens
        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

        self._pool_rule_capacity = new_rule_capacity
        self._pool_token_capacity = new_token_capacity
        self._pool_meta_d = torch.empty(
            (new_rule_capacity, self.META_WIDTH), dtype=torch.int32, device=self.device)
        self._pool_tokens_d = torch.empty(
            new_token_capacity, dtype=torch.int16, device=self.device)

        if after_tokens:
            arr = np.asarray(new_tokens, dtype=np.int16)
            self._pool_tokens_d[:after_tokens].copy_(torch.from_numpy(arr))
        if after_rules:
            arr = np.asarray(new_meta, dtype=np.int32)
            self._pool_meta_d[:after_rules].copy_(torch.from_numpy(arr))
        try:
            torch.mps.synchronize()
        except Exception:
            pass

        self.pool_gc_count += 1
        self.last_pool_gc_triggered = True
        self.last_pool_gc_reason = str(reason)
        self.last_pool_gc_rules_before = before_rules
        self.last_pool_gc_rules_after = after_rules
        self.last_pool_gc_tokens_before = before_tokens
        self.last_pool_gc_tokens_after = after_tokens
        self.last_pool_gc_alloc_before = before_alloc
        self.last_pool_gc_alloc_after = self._mps_allocated_bytes()
        self.last_pool_gc_seconds = time.perf_counter() - t0

        if self.progress:
            freed_rules = before_rules - after_rules
            freed_tokens = before_tokens - after_tokens
            print(
                f"  rule-pool compact GC[{reason}]: rules {before_rules:,}->{after_rules:,} "
                f"(-{freed_rules:,}), tokens {before_tokens:,}->{after_tokens:,} "
                f"(-{freed_tokens:,}), cap={new_rule_capacity:,}/{new_token_capacity:,} "
                f"mps={before_alloc / (1024**2):.1f}->{self.last_pool_gc_alloc_after / (1024**2):.1f} MiB "
                f"({self.last_pool_gc_seconds:.2f}s)",
                flush=True,
            )

    def _ensure_pool_capacity(
        self,
        rule_need: int,
        token_need: int,
        rule_used: int,
        token_used: int,
    ) -> None:
        """Grow resident GPU pool buffers without reading past old capacity.

        ``rule_need``/``token_need`` are the *new* logical host sizes after
        interning this generation.  Only ``rule_used``/``token_used`` entries
        are already valid on the old GPU buffers and therefore need copying.
        This distinction matters especially on the first generation, where the
        host pool may jump from 0 to hundreds of thousands of rules while the
        device buffer still has only its small initial capacity.
        """
        if rule_need > self._pool_rule_capacity:
            new_cap = self._pool_rule_capacity
            while new_cap < rule_need:
                new_cap *= 2
            new_d = torch.empty(
                (new_cap, self.META_WIDTH), dtype=torch.int32, device=self.device)
            copy_n = min(int(rule_used), int(self._pool_rule_capacity))
            if copy_n > 0:
                new_d[:copy_n].copy_(self._pool_meta_d[:copy_n])
            self._pool_meta_d = new_d
            self._pool_rule_capacity = new_cap

        if token_need > self._pool_token_capacity:
            new_cap = self._pool_token_capacity
            while new_cap < token_need:
                new_cap *= 2
            new_d = torch.empty(new_cap, dtype=torch.int16, device=self.device)
            copy_n = min(int(token_used), int(self._pool_token_capacity))
            if copy_n > 0:
                new_d[:copy_n].copy_(self._pool_tokens_d[:copy_n])
            self._pool_tokens_d = new_d
            self._pool_token_capacity = new_cap

    def _intern_rule(self, rule: Rule) -> int:
        if rule.pool_owner == self._pool_owner and 0 <= rule.pool_id < len(self._pool_meta_host):
            return int(rule.pool_id)

        key = (tuple(map(int, rule.pattern)), tuple(map(int, rule.replacement)))
        pid = self._pool_key_to_id.get(key)
        if pid is not None:
            rule.pool_id = int(pid)
            rule.pool_owner = self._pool_owner
            return int(pid)

        p, rep = rule.pattern, rule.replacement
        if int(rule.pack_wc) < 0:
            wc = sum(v < 0 for v in p)
            if wc > MAX_WILDCARDS:
                raise ValueError(f"rule has {wc} wildcards; max={MAX_WILDCARDS}")
            if any(-31 <= v <= -16 for v in rep):
                raise ValueError("sort($capture) is not supported by the fused MPS path")
            rule.pack_wc = int(wc)
            rule.pack_leading = 1 if p and p[0] >= 0 else 0
            rule.pack_rep_literal = 1 if all(v >= 0 for v in rep) else 0

        po = len(self._pool_tokens_host)
        self._pool_tokens_host.extend(map(int, p))
        ro = len(self._pool_tokens_host)
        self._pool_tokens_host.extend(map(int, rep))
        pid = len(self._pool_meta_host)
        self._pool_meta_host.append((
            po, len(p), ro, len(rep), int(rule.pack_wc),
            int(rule.pack_leading), int(rule.pack_rep_literal),
        ))
        self._pool_key_to_id[key] = pid
        rule.pool_id = pid
        rule.pool_owner = self._pool_owner
        return pid

    def _flush_pool_growth(self, old_rule_count: int, old_token_count: int) -> None:
        new_rule_count = len(self._pool_meta_host)
        new_token_count = len(self._pool_tokens_host)
        self.last_pool_new_rules = new_rule_count - old_rule_count
        self.last_pool_new_tokens = new_token_count - old_token_count
        t0 = time.perf_counter()
        self._ensure_pool_capacity(
            new_rule_count, new_token_count, old_rule_count, old_token_count)
        if new_token_count > old_token_count:
            arr = np.asarray(
                self._pool_tokens_host[old_token_count:new_token_count], dtype=np.int16)
            host = torch.from_numpy(arr)
            self._pool_tokens_d[old_token_count:new_token_count].copy_(host)
        if new_rule_count > old_rule_count:
            arr = np.asarray(self._pool_meta_host[old_rule_count:new_rule_count], dtype=np.int32)
            host = torch.from_numpy(arr)
            self._pool_meta_d[old_rule_count:new_rule_count].copy_(host)
        self.last_pool_upload_seconds = time.perf_counter() - t0

    def _ensure_index_capacity(self, genome_need: int) -> None:
        """Grow reusable host/device candidate-index buffers geometrically."""
        if genome_need <= self._index_genome_capacity:
            return
        cap = max(8, self._index_genome_capacity)
        while cap < genome_need:
            cap *= 2
        pair_cap = self._index_pair_capacity

        self._host_rule_ids = np.empty((cap, self.rule_count), dtype=np.int32)
        self._host_embeddings = np.empty((cap, EMBEDDING_ENTRY_COUNT), dtype=np.int32)
        self._host_anchors = np.empty((cap, self.rule_count, 5), dtype=np.int32)

        self._rule_ids_d = torch.empty((cap, self.rule_count), dtype=torch.int32, device=self.device)
        self._embeddings_d = torch.empty((cap, EMBEDDING_ENTRY_COUNT), dtype=torch.int32, device=self.device)
        self._anchors_d = torch.empty((cap, self.rule_count, 5), dtype=torch.int32, device=self.device)
        self._head1_d = torch.empty((cap, _GP_INDEX_TOKENS), dtype=torch.int32, device=self.device)
        self._next1_d = torch.empty((cap, self.rule_count), dtype=torch.int32, device=self.device)
        self._next2_d = torch.empty((cap, self.rule_count), dtype=torch.int32, device=self.device)
        self._always_d = torch.empty((cap, _GP_RULE_WORDS), dtype=torch.int32, device=self.device)
        self._pair_keys_d = torch.empty((cap, pair_cap), dtype=torch.int32, device=self.device)
        self._pair_heads_d = torch.empty((cap, pair_cap), dtype=torch.int32, device=self.device)
        self._index_genome_capacity = cap

    def _pack_population(self, genomes: Sequence[Genome], pool_live_genomes: Sequence[Genome] | None = None):
        t0 = time.perf_counter()
        self.last_pool_gc_triggered = False
        self.last_pool_gc_reason = ""
        self.last_pool_gc_seconds = 0.0
        gcount = len(genomes)
        self._ensure_index_capacity(gcount)
        pair_cap = self._index_pair_capacity

        gc_roots = genomes if pool_live_genomes is None else pool_live_genomes
        gc_reason = self._gc_trigger(gc_roots)
        if gc_reason:
            # Score reuse may evaluate only a subset, but compacting GC must trace
            # every live population/HoF rule or cached pool ids would become stale.
            self._compact_rule_pool(gc_roots, gc_reason)
            for g in gc_roots:
                _invalidate_genome_pack(g)
                g._pack_owner = 0

        old_rule_count = len(self._pool_meta_host)
        old_token_count = len(self._pool_tokens_host)
        rule_ids = self._host_rule_ids[:gcount]
        embeddings = self._host_embeddings[:gcount]
        anchors = self._host_anchors[:gcount]

        # v29 incremental host pack. Most offspring differ from a parent in only
        # a handful of rows. Their immutable parent rule-id/anchor arrays are
        # shared copy-on-write and only dirty rows are rebuilt here.
        cache_rows = 0
        rebuilt_rows = 0
        rule_memo_hits = 0
        rule_memo_misses = 0
        for gi, genome in enumerate(genomes):
            if len(genome.rules) != self.rule_count:
                raise ValueError("all genomes must keep a fixed rule count")
            embeddings[gi, :] = genome.embedding

            cache_valid = (
                int(genome._pack_owner) == int(self._pool_owner)
                and isinstance(genome._pack_rule_ids, np.ndarray)
                and isinstance(genome._pack_anchors, np.ndarray)
                and genome._pack_rule_ids.shape == (self.rule_count,)
                and genome._pack_anchors.shape == (self.rule_count, 5)
                and not genome._pack_dirty_all
            )
            if cache_valid:
                dirty = sorted(i for i in genome._pack_dirty_rows if 0 <= int(i) < self.rule_count)
                ids_local = genome._pack_rule_ids if not dirty else genome._pack_rule_ids.copy()
                anchors_local = genome._pack_anchors if not dirty else genome._pack_anchors.copy()
                cache_rows += self.rule_count - len(dirty)
            else:
                dirty = range(self.rule_count)
                ids_local = np.empty(self.rule_count, dtype=np.int32)
                anchors_local = np.empty((self.rule_count, 5), dtype=np.int32)

            inverse = None
            for r in dirty:
                r = int(r)
                rule = genome.rules[r]
                pid = self._intern_rule(rule)
                ids_local[r] = pid
                p = rule.pattern
                if not p:
                    kind, a, b = 0, -1, -1
                elif int(rule.anchor_kind) < 0:
                    if inverse is None:
                        inverse = self._inverse_embedding(genome.embedding)
                    kind, a, b = self._candidate_anchor(p, inverse)
                    rule.anchor_kind = int(kind)
                    rule.anchor_a = int(a)
                    rule.anchor_b = int(b)
                else:
                    kind = int(rule.anchor_kind)
                    a = int(rule.anchor_a)
                    b = int(rule.anchor_b)
                anchors_local[r, 0] = int(kind)
                anchors_local[r, 1] = int(a)
                anchors_local[r, 2] = int(b)
                # v55: secondary Bloom guard is structural. Once selected, it
                # stays an exact necessary condition for this unchanged Rule, so
                # crossover rows can reuse it instead of rescanning the pattern.
                if int(rule.pack_filter_key) != -2:
                    secondary_pair_key = int(rule.pack_filter_key)
                    rule_memo_hits += 1
                else:
                    if inverse is None:
                        inverse = self._inverse_embedding(genome.embedding)
                    secondary_pair_key = self._secondary_pair_filter(p, inverse, a, b)
                    rule.pack_filter_key = int(secondary_pair_key)
                    rule_memo_misses += 1
                anchors_local[r, 4] = int(secondary_pair_key)

                if int(rule.pack_anchor_offset) != -2:
                    off = int(rule.pack_anchor_offset)
                else:
                    off = -1
                    if kind == 2:
                        off = next((q for q in range(len(p) - 1)
                                    if int(p[q]) == int(a) and int(p[q + 1]) == int(b)), -1)
                    elif kind == 1:
                        off = next((q for q, v in enumerate(p) if int(v) == int(a)), -1)
                    rule.pack_anchor_offset = int(off)
                anchors_local[r, 3] = int(off)
                rebuilt_rows += 1

            genome._pack_owner = int(self._pool_owner)
            genome._pack_rule_ids = ids_local
            genome._pack_anchors = anchors_local
            genome._pack_dirty_all = False
            genome._pack_dirty_rows.clear()
            rule_ids[gi, :] = ids_local
            anchors[gi, :, :] = anchors_local

        self.last_pack_cache_rows = int(cache_rows)
        self.last_pack_rebuilt_rows = int(rebuilt_rows)
        self.last_rule_pack_memo_hits = int(rule_memo_hits)
        self.last_rule_pack_memo_misses = int(rule_memo_misses)

        self._flush_pool_growth(old_rule_count, old_token_count)

        # Three compact H->MPS uploads replace six large Python-built index uploads.
        self._rule_ids_d[:gcount].copy_(torch.from_numpy(rule_ids))
        self._embeddings_d[:gcount].copy_(torch.from_numpy(embeddings))
        self._anchors_d[:gcount].copy_(torch.from_numpy(anchors))

        # Ordered index construction stays on the GPU.  It is queued
        # asynchronously; the later result copy is the only synchronization.
        self._lib.build_gp_candidate_index_v23_i32(
            self._anchors_d[:gcount].reshape(-1),
            self._head1_d[:gcount].reshape(-1),
            self._next1_d[:gcount].reshape(-1),
            self._pair_keys_d[:gcount].reshape(-1),
            self._pair_heads_d[:gcount].reshape(-1),
            self._next2_d[:gcount].reshape(-1),
            self._always_d[:gcount].reshape(-1),
            int(self.rule_count), int(gcount), int(pair_cap),
            threads=[gcount * self.threadgroup_size, 1, 1],
            group_size=[self.threadgroup_size, 1, 1],
        )

        packed = (
            self._pool_tokens_d,
            self._pool_meta_d,
            self._rule_ids_d[:gcount],
            self._embeddings_d[:gcount],
            self._head1_d[:gcount],
            self._next1_d[:gcount],
            self._pair_keys_d[:gcount],
            self._pair_heads_d[:gcount],
            self._next2_d[:gcount],
            self._always_d[:gcount],
            pair_cap,
        )
        self.last_pack_seconds = time.perf_counter() - t0
        return packed

    def evaluate_population(
        self,
        genomes: Sequence[Genome],
        pool_live_genomes: Sequence[Genome] | None = None,
        prepacked=None,
    ):
        total = len(genomes)
        if total <= 0:
            return []
        # Inner string inference evaluates the exact same model subset for
        # several candidate generations; only raw inputs change.  Rebuilding and
        # re-uploading the rule/index pack on every inner round is pure overhead.
        # A prepacked tuple is safe as long as the genome sequence is unchanged
        # and no other _pack_population call occurs between uses.
        if prepacked is None:
            packed = self._pack_population(genomes, pool_live_genomes=pool_live_genomes)
        else:
            packed = prepacked
        (
            tokens_d, meta_d, rule_ids_d, emb_d, head1_d, next1_d,
            pair_keys_d, pair_heads_d, next2_d, always_d, pair_cap,
        ) = packed
        out = []
        profile_chunks = []
        stat_chunks = []
        dispatch_t0 = time.perf_counter()
        n_samples = int(self.active_sample_count)

        for chunk_start in range(0, total, self.result_chunk):
            chunk_count = min(self.result_chunk, total - chunk_start)
            # Kernels pack active samples tightly using n_samples as the stride.
            # Clear/read the matching flat prefix rather than the full training
            # sample dimension.
            self.feature_stage.reshape(-1)[:chunk_count * n_samples * self.rule_count].zero_()
            self.stat_stage.reshape(-1)[:chunk_count * n_samples * 4].zero_()
            self.profile_stage.reshape(-1)[:chunk_count * n_samples * 15].zero_()

            for local_start in range(0, chunk_count, self.genome_batch):
                gcount = min(self.genome_batch, chunk_count - local_start)
                global_start = chunk_start + local_start
                jobs = gcount * n_samples
                self._lib.evaluate_gp_population_v17_i32(
                    self.buffer_a, self.buffer_b,
                    self.raw_inputs, self.raw_lengths,
                    emb_d, tokens_d, meta_d.reshape(-1), rule_ids_d.reshape(-1),
                    head1_d.reshape(-1), next1_d.reshape(-1),
                    pair_keys_d.reshape(-1), pair_heads_d.reshape(-1),
                    next2_d.reshape(-1), always_d.reshape(-1),
                    self.feature_stage.reshape(-1), self.stat_stage.reshape(-1),
                    self.profile_stage.reshape(-1), self.seen,
                    int(self.capacity), int(self.max_output), int(self.raw_stride),
                    int(n_samples), int(self.rule_count), int(gcount),
                    int(global_start), int(local_start), int(self.max_rounds),
                    int(pair_cap), self.literal_positions, self.wildcard_aux,
                    self._anchors_d[:total].reshape(-1),
                    threads=[jobs * self.threadgroup_size, 1, 1],
                    group_size=[self.threadgroup_size, 1, 1],
                )

            # One synchronization per result chunk.  Kernel work before this is
            # queued asynchronously; the copy waits only after many threadgroups.
            fh = (self.feature_stage.reshape(-1)[:chunk_count * n_samples * self.rule_count]
                  .reshape(chunk_count, n_samples, self.rule_count).cpu().numpy())
            sh = (self.stat_stage.reshape(-1)[:chunk_count * n_samples * 4]
                  .reshape(chunk_count, n_samples, 4).cpu().numpy())
            ph = (self.profile_stage.reshape(-1)[:chunk_count * n_samples * 15]
                  .reshape(chunk_count, n_samples, 15).cpu().numpy())
            if self.unique_input_count < n_samples:
                dup = np.flatnonzero(self._sample_alias != np.arange(n_samples, dtype=np.int32))
                if dup.size:
                    reps = self._sample_alias[dup]
                    fh[:, dup, :] = fh[:, reps, :]
                    sh[:, dup, :] = sh[:, reps, :]
                    ph[:, dup, :] = ph[:, reps, :]
            profile_chunks.append(ph.copy())
            stat_chunks.append(sh.copy())
            for gi in range(chunk_count):
                x = fh[gi].astype(np.float32, copy=False)
                st = sh[gi].astype(np.int32, copy=False)
                baseline = -st[:, 2].astype(np.float64)
                out.append((x, baseline, st))
            if self.progress:
                done = min(total, chunk_start + chunk_count)
                print(f"  mps cooperative eval {done}/{total}", end="\r", flush=True)
        self.last_dispatch_seconds = time.perf_counter() - dispatch_t0
        if profile_chunks:
            prof = np.concatenate(profile_chunks, axis=0).reshape(-1, 15).astype(np.float64)
            sts = np.concatenate(stat_chunks, axis=0).reshape(-1, 4).astype(np.float64)
            # Candidate attempts are accumulated across candidate-index rebuilds,
            # not merely across rewrite rounds.  The old rounds*rules denominator
            # could report nonsensical percentages and hide the real bottleneck.
            index_passes = prof[:, 0] + prof[:, 10]
            denom = np.maximum(1.0, index_passes * float(self.rule_count))
            rew = np.maximum(1.0, prof[:, 3])
            self.last_profile = {
                "rounds_mean": float(np.mean(sts[:, 0])),
                "rounds_p95": float(np.percentile(sts[:, 0], 95)),
                "candidate_rebuilds_mean": float(np.mean(prof[:, 0])),
                "candidate_extensions_mean": float(np.mean(prof[:, 10])),
                "candidate_extension_tokens_mean": float(np.mean(prof[:, 11])),
                "candidate_scan_tokens_mean": float(np.mean(prof[:, 1])),
                "candidate_rules_mean": float(np.mean(prof[:, 2])),
                "candidate_rules_p95": float(np.percentile(prof[:, 2], 95)),
                "candidate_fraction_mean": float(np.mean(prof[:, 2] / denom)),
                "candidate_scan_tokens_p95": float(np.percentile(prof[:, 1], 95)),
                "rewrites_mean": float(np.mean(prof[:, 3])),
                "literal_fast_rewrites_mean": float(np.mean(prof[:, 4])),
                "generic_rewrites_mean": float(np.mean(prof[:, 5])),
                "wc1_literal_fast_rewrites_mean": float(np.mean(prof[:, 12])),
                "inplace_literal_rewrites_mean": float(np.mean(prof[:, 13])),
                "secondary_filter_skips_mean": float(np.mean(prof[:, 14])),
                "secondary_filter_skip_fraction": float(np.mean(prof[:, 14] / np.maximum(1.0, prof[:, 2]))),
                "same_length_literal_rewrites_mean": float(np.mean(prof[:, 6])),
                "pattern_len1_literal_rewrites_mean": float(np.mean(prof[:, 7])),
                "literal_fast_fraction": float(np.mean(prof[:, 4] / rew)),
                "wc1_literal_fast_fraction": float(np.mean(prof[:, 12] / rew)),
                "inplace_literal_fraction": float(np.mean(prof[:, 13] / rew)),
                "same_length_literal_fraction": float(np.mean(prof[:, 6] / rew)),
                "pattern_len1_literal_fraction": float(np.mean(prof[:, 7] / rew)),
                "matches_per_rewrite": float(np.mean(prof[:, 8] / rew)),
                "state_len_per_rewrite": float(np.mean(prof[:, 9] / rew)),
                "cycle_rate": float(np.mean(sts[:, 3] > 0.0)),
                "overflow_rate": float(np.mean(sts[:, 2] > 0.0)),
            }
        else:
            self.last_profile = {}
        if self.progress:
            print(" " * 56, end="\r", flush=True)
        return out


# ---------------------------------------------------------------------------
# Rank-aware readout + Spearman fitness
# ---------------------------------------------------------------------------


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks for ties, 1..N like the Nim Spearman helper."""
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    if n == 0:
        return ranks
    values = x[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    ranks[order] = np.repeat((starts + ends + 1) * 0.5, ends - starts)
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    rx, ry = average_ranks(x), average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    if den <= 1e-12:
        return 0.0
    return float((rx @ ry) / den)


def fit_dual_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Exact dual ridge retained for legacy/A-B readout mode and tests."""
    n, p = x.shape
    w = np.zeros(p, dtype=np.float64)
    if n < 2 or p == 0:
        return w
    xd = x.astype(np.float64, copy=False)
    ss = np.sum(xd * xd, axis=0)
    active = ss > 1e-12
    if not np.any(active):
        return w
    xd = xd[:, active]
    scale = n / ss[active]  # invRMS^2; inactive columns are exactly zero
    k = (xd * scale[None, :]) @ xd.T
    k.flat[:: n + 1] += lam
    try:
        # K is SPD for lam>0.  Cholesky is both cheaper and more stable than
        # a generic LU solve, while preserving the exact legacy objective.
        chol = np.linalg.cholesky(k)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(k, y, rcond=None)[0]
    w[active] = scale * (xd.T @ alpha)
    np.clip(w, -1e6, 1e6, out=w)
    return w


def case_adjacent_rank_pairs(
    cleanliness: np.ndarray,
    case_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a sparse all-sample rank graph inside each case.

    For a case with m trajectories, sort all m by target rank and connect the
    chain with m-1 adjacent constraints.  Thus every trajectory participates in
    pairwise fitting, but we use O(m), not O(m^2), pairs.  Ties receive average
    ranks and therefore a zero desired margin when adjacent within the tie.

    Returns (higher_rank_index, lower_rank_index, normalized_rank_margin).
    """
    cleanliness = np.asarray(cleanliness, dtype=np.float64)
    case_ids = np.asarray(case_ids, dtype=np.int32)
    hi_all: List[int] = []
    lo_all: List[int] = []
    margin_all: List[float] = []
    for ci in np.unique(case_ids):
        ids = np.flatnonzero(case_ids == ci)
        m = int(ids.size)
        if m < 2:
            continue
        vals = cleanliness[ids]
        ranks = average_ranks(vals)
        order = np.argsort(ranks, kind="mergesort")
        norm_rank = (ranks - 1.0) / max(1.0, float(m - 1))
        lo_local = order[:-1]
        hi_local = order[1:]
        lo_ids = ids[lo_local]
        hi_ids = ids[hi_local]
        margins = norm_rank[hi_local] - norm_rank[lo_local]
        hi_all.extend(map(int, hi_ids))
        lo_all.extend(map(int, lo_ids))
        margin_all.extend(map(float, margins))
    return (
        np.asarray(hi_all, dtype=np.int32),
        np.asarray(lo_all, dtype=np.int32),
        np.asarray(margin_all, dtype=np.float64),
    )


def fit_pairwise_rank_ridge_all(
    x: np.ndarray,
    baseline: np.ndarray,
    cleanliness: np.ndarray,
    case_ids: np.ndarray,
    ridge_lambda: float,
    max_features: int = PAIRWISE_RANK_MAX_FEATURES,
) -> np.ndarray:
    """Fit sparse pairwise Rank Ridge from *all* trajectories.

    Every trajectory participates in the case-local adjacent-pair graph.  To keep
    host time close to the former 36-row readout, score all rule columns once by
    normalized pairwise covariance and solve the exact dual ridge only on the
    strongest ``max_features`` columns.  This is supervised sure-independence
    screening: it changes feature sparsity, not which trajectories participate.
    """
    hi, lo, margin = case_adjacent_rank_pairs(cleanliness, case_ids)
    p = int(x.shape[1])
    if hi.size == 0 or p == 0:
        return np.zeros(p, dtype=np.float64)

    # pred = baseline + Xw. For each rank edge hi>lo fit
    #   (X_hi-X_lo)w ~= desired_rank_margin - (b_hi-b_lo).
    pair_x = x[hi].astype(np.float32, copy=False) - x[lo].astype(np.float32, copy=False)
    pair_y = margin - (baseline[hi].astype(np.float64) - baseline[lo].astype(np.float64))

    pd = pair_x.astype(np.float64, copy=False)
    ss = np.sum(pd * pd, axis=0)
    usable = np.flatnonzero(ss > 1.0e-12)
    if usable.size == 0:
        return np.zeros(p, dtype=np.float64)

    cap = max(1, int(max_features))
    if usable.size > cap:
        # Scale-free univariate rank relevance.  All 105 default pair rows
        # contribute to this score before any feature is dropped.
        rel = np.abs(pd[:, usable].T @ pair_y) / np.sqrt(ss[usable])
        keep_local = np.argpartition(rel, -cap)[-cap:]
        selected = usable[keep_local]
    else:
        selected = usable

    selected_w = fit_dual_ridge(pd[:, selected], pair_y, ridge_lambda)
    w = np.zeros(p, dtype=np.float64)
    w[selected] = selected_w
    return w


def score_genome_features(
    genome: Genome,
    x: np.ndarray,
    baseline: np.ndarray,
    cleanliness: np.ndarray,
    case_ids: np.ndarray,
    sample_ids: np.ndarray,
    ridge_lambda: float,
    readout_mode: str = "pairwise-all",
    rank_max_features: int = PAIRWISE_RANK_MAX_FEATURES,
) -> float:
    """Fit the readout and score per-case Spearman.

    pairwise-all (default): all trajectories participate in case-local adjacent
    pairwise rank fitting, and those same trajectories are all used for the
    per-case Spearman fitness requested for v59.

    legacy-holdout: the previous 1/3 raw-target ridge + 2/3 held-out Spearman,
    kept only for controlled A/B comparisons and checkpoint compatibility.
    """
    if readout_mode == "legacy-holdout":
        train = (sample_ids % TRAIN_MOD) == 0
        if int(train.sum()) < 2:
            train[:] = True
        residual = cleanliness[train] - baseline[train]
        w = fit_dual_ridge(x[train], residual, ridge_lambda)
    else:
        w = fit_pairwise_rank_ridge_all(
            x, baseline, cleanliness, case_ids, ridge_lambda, rank_max_features
        )

    genome.readout_weights = w.astype(np.float64, copy=False).tolist()
    active = np.flatnonzero(w)
    pred = baseline.astype(np.float64, copy=True)
    if active.size:
        pred += x[:, active].astype(np.float64, copy=False) @ w[active]

    scores: List[float] = []
    for ci in np.unique(case_ids):
        if readout_mode == "legacy-holdout":
            use = (case_ids == ci) & ((sample_ids % TRAIN_MOD) != 0)
        else:
            # v59: every one of the 36 default trajectories per case contributes
            # to Spearman, including every trajectory used by pairwise fitting.
            use = case_ids == ci
        if int(use.sum()) >= 2:
            scores.append(spearman(pred[use], cleanliness[use]))
    fitness = float(np.mean(scores)) if scores else 0.0
    if not math.isfinite(fitness):
        fitness = -1.0
    genome.fitness = fitness
    genome.case_scores = scores
    return fitness


def evaluate_genome(
    genome: Genome,
    inputs: Sequence[Sequence[int]],
    cleanliness: np.ndarray,
    case_ids: np.ndarray,
    sample_ids: np.ndarray,
    backend: str,
    ridge_lambda: float,
    max_output: int,
    readout_mode: str = "pairwise-all",
    rank_max_features: int = PAIRWISE_RANK_MAX_FEATURES,
) -> float:
    if backend == "mps":
        # Compatibility path. evolve() uses one persistent evaluator for the
        # entire population instead of constructing this per genome.
        ev = MpsPopulationEvaluator(
            len(inputs), len(genome.rules), max(len(x) for x in inputs),
            max_output, genome_batch=1,
        )
        ev.set_inputs(inputs)
        x, baseline, _ = ev.evaluate_population([genome])[0]
    else:
        x, baseline, _, _ = trajectory_features_cpu(inputs, genome, max_output)
    return score_genome_features(
        genome, x, baseline, cleanliness, case_ids, sample_ids, ridge_lambda, readout_mode, rank_max_features
    )


# ---------------------------------------------------------------------------
# v60 string-side inference objective + Pareto selection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InferenceCase:
    clean: List[int]
    noisy: List[int]
    mutable_positions: Tuple[int, ...]
    source_index: int
    serial: int


class RollingInferenceSet:
    """Small held-apart corruption set used by the inner string GA.

    Only the positions deliberately corrupted here may be changed by inference.
    Cases rotate independently from the Spearman trajectories so the second
    objective cannot merely memorize the current rank-fitting batch.
    """
    def __init__(self, corpus: Sequence[bytes], sampler: CorpusSampler,
                 cases: int, span: int, noise_rate: float,
                 excluded_sources: Sequence[int] | None = None):
        if not corpus:
            raise ValueError("empty corpus")
        self.corpus = corpus
        self.sampler = sampler
        self.case_count = max(1, min(int(cases), len(corpus)))
        # Respect the evaluator's raw stride even in tiny smoke configurations;
        # forcing a 16-byte minimum could create inference rows longer than an
        # explicitly smaller --trajectory-len/--max-chunk MPS allocation.
        self.span = max(1, int(span))
        self.noise_rate = min(0.50, max(1.0 / self.span, float(noise_rate)))
        self._serial = 0
        excluded = set(map(int, excluded_sources or ()))
        eligible = [i for i in range(len(corpus)) if i not in excluded]
        source = eligible if len(eligible) >= self.case_count else list(range(len(corpus)))
        chosen = random.sample(source, self.case_count)
        self.slots = [self._make_case(i) for i in chosen]

    def _make_case(self, source_index: int) -> InferenceCase:
        base = bytes(self.corpus[int(source_index)])
        if len(base) > self.span:
            start = random.randrange(0, len(base) - self.span + 1)
            base = base[start:start + self.span]
        clean = list(base)
        n = len(clean)
        if n <= 0:
            clean = [0]
            n = 1
        k = max(1, min(n, int(round(self.noise_rate * n))))
        positions = tuple(sorted(random.sample(range(n), k)))
        noisy = clean.copy()
        for pos in positions:
            old = noisy[pos]
            v = self.sampler.byte()
            tries = 0
            while v == old and tries < 16:
                v = self.sampler.byte()
                tries += 1
            if v == old:
                v = (old + 1) & 255
            noisy[pos] = int(v)
        serial = self._serial
        self._serial += 1
        return InferenceCase(clean, noisy, positions, int(source_index), int(serial))

    def rotate(self, slot: int, excluded_sources: Sequence[int] | None = None) -> Tuple[int, int]:
        slot = int(slot) % self.case_count
        occupied = {c.source_index for i, c in enumerate(self.slots) if i != slot}
        excluded = set(map(int, excluded_sources or ()))
        candidates = [i for i in range(len(self.corpus))
                      if i not in occupied and i not in excluded]
        if not candidates:
            candidates = [i for i in range(len(self.corpus)) if i not in occupied]
        if not candidates:
            candidates = list(range(len(self.corpus)))
        old = self.slots[slot].serial
        self.slots[slot] = self._make_case(random.choice(candidates))
        return old, self.slots[slot].serial


def resolve_inference_budget(sample_count: int, requested_cases: int,
                             requested_population: int, requested_generations: int,
                             ratio: float, genome_count: int | None = None,
                             outer_evaluated_genomes: int | None = None,
                             ensemble_size: int | None = None) -> Tuple[int, int, int, int]:
    """Bound inner inference work in *full-population trajectory equivalents*.

    v60 originally budgeted only ``cases * population * generations`` and then
    evaluated every outer genome on every inner generation.  That understated
    wall-clock risk when outer score reuse made only a fraction of genomes need
    normal evaluation.  The checked version searches with only the voting
    ensemble and performs one full-population scoring pass on the final candidate
    pool.  Its approximate equivalent work is therefore

        cases * population * (1 + generations * ensemble / genome_count).

    ``outer_evaluated_genomes`` further scales the allowance when the normal
    evaluator reused many genomes.  The 0.85 hard ratio is retained as a safety
    ceiling; tiny smoke configurations keep a small floor so inference remains
    executable at all.  The returned ``work`` is the ceiling of this equivalent
    trajectory count, not raw model×trajectory jobs.
    """
    sample_count = max(1, int(sample_count))
    n_genomes = max(1, int(genome_count) if genome_count is not None else 1)
    ensemble = n_genomes if ensemble_size is None else max(1, min(n_genomes, int(ensemble_size)))
    outer_n = n_genomes if outer_evaluated_genomes is None else max(1, min(n_genomes, int(outer_evaluated_genomes)))

    hard_ratio = min(0.85, max(0.05, float(ratio)))
    outer_fraction = outer_n / float(n_genomes)
    budget = max(4.0, sample_count * hard_ratio * outer_fraction)

    # At least two candidate strings are needed for an actual GA population.
    cases = max(1, min(int(requested_cases), max(1, sample_count // 2)))
    generations = max(1, int(requested_generations))
    population = max(2, int(requested_population))

    def equivalent_work(c: int, p: int, g: int) -> float:
        return float(c * p) * (1.0 + float(g) * ensemble / float(n_genomes))

    # Final all-genome scoring evaluates cases*population strings in one active
    # prefix, so it must fit in the evaluator's training sample allocation.
    population = min(population, max(2, sample_count // cases))

    # Prefer preserving several independent cases.  First reduce candidate
    # population, then inner generations, and only then the number of cases.
    # Recompute after each structural reduction because the active-sample bound
    # changes with ``cases``.
    while equivalent_work(cases, population, generations) > budget:
        denom = cases * (1.0 + float(generations) * ensemble / float(n_genomes))
        max_pop = int(math.floor(budget / max(1.0e-12, denom)))
        if max_pop >= 2 and population > max_pop:
            population = max(2, max_pop)
            continue
        if generations > 1:
            generations -= 1
            continue
        if cases > 1:
            cases -= 1
            population = min(population, max(2, sample_count // cases))
            continue
        # With an extremely cheap/reused outer generation even 1×2×1 may exceed
        # the nominal ratio.  Keep the minimal meaningful inference pass.
        population = 2
        break

    actual = int(math.ceil(equivalent_work(cases, population, generations)))
    return cases, population, generations, actual


def _inference_recovery_accuracy(candidate: Sequence[int], case: InferenceCase) -> float:
    pos = case.mutable_positions
    if not pos:
        return 1.0
    return sum(int(candidate[i]) == int(case.clean[i]) for i in pos) / float(len(pos))


def _inference_full_accuracy(candidate: Sequence[int], case: InferenceCase) -> float:
    n = min(len(candidate), len(case.clean))
    if n <= 0:
        return 1.0
    return sum(int(candidate[i]) == int(case.clean[i]) for i in range(n)) / float(n)


def _propose_inference_byte(candidate: Sequence[int], pos: int, sampler: CorpusSampler) -> int:
    """Proposal distribution uses corpus/context only; never the hidden clean byte.

    v60.1 strongly prefers local corpus transitions.  The previous 72% global
    histogram proposal effectively searched a 256-way alphabet almost blindly,
    which made the exact-match Pareto objective stay identically zero for long
    runs even when the scoring code itself was correct.
    """
    old = int(candidate[pos])
    r = random.random()
    if r < 0.68:
        v = sampler.contextual_byte(candidate, int(pos), avoid=old)
    elif r < 0.88:
        v = sampler.byte()
    elif r < 0.97 and len(candidate) > 1:
        lo = max(0, int(pos) - 12)
        hi = min(len(candidate), int(pos) + 13)
        pool = [int(candidate[j]) for j in range(lo, hi) if j != pos and int(candidate[j]) != old]
        v = random.choice(pool) if pool else sampler.byte()
    else:
        v = random.randrange(INPUT_BYTE_COUNT)
    if v == old:
        # Retry context first, then guarantee a real mutation.
        v = sampler.contextual_byte(candidate, int(pos), avoid=old)
        if v == old:
            v = (old + 1 + random.randrange(255)) & 255
    return int(v)


def sample_inference_mutation_count(item_count: int) -> int:
    """Log-uniform number of corrupted loci to mutate, in ``[1, item_count]``.

    The draw is exactly the requested geometry before integer rounding::

        exp(uniform(0, log(item_count)))

    This keeps one/few-locus edits common while giving every scale up to a
    whole-corruption rewrite a non-negligible chance.
    """
    n = max(0, int(item_count))
    if n <= 0:
        return 0
    if n == 1:
        return 1
    draw = math.exp(random.uniform(0.0, math.log(float(n))))
    return max(1, min(n, int(round(draw))))


def mutate_inference_candidate(candidate: Sequence[int], positions: Sequence[int],
                               sampler: CorpusSampler, force_changes: int = 0,
                               focus_position: int | None = None) -> List[int]:
    out = list(map(int, candidate))
    if not positions:
        return out
    pos_list = list(map(int, positions))

    if force_changes > 0:
        count = max(1, min(len(pos_list), int(force_changes)))
    else:
        count = sample_inference_mutation_count(len(pos_list))

    # ``focus_position`` is an exploration-scheduling hint, not a request to
    # collapse the mutation radius to one locus.  When present, guarantee that
    # locus is among the log-uniformly many changed genes and draw the remainder
    # without replacement from the other known-corrupted positions.
    focus = int(focus_position) if focus_position is not None else None
    if focus is not None and focus in pos_list:
        others = [p for p in pos_list if p != focus]
        chosen = [focus]
        if count > 1:
            chosen.extend(random.sample(others, min(count - 1, len(others))))
    else:
        chosen = random.sample(pos_list, count)

    for pos in chosen:
        out[int(pos)] = _propose_inference_byte(out, int(pos), sampler)
    return out


def crossover_inference_candidates(a: Sequence[int], b: Sequence[int],
                                   positions: Sequence[int]) -> List[int]:
    out = list(map(int, a))
    if not positions:
        return out
    # Coordinate-wise uniform crossover is natural because only known corrupted
    # loci are genes; context bytes are immutable by construction.
    for pos in positions:
        if random.random() < 0.35:
            out[int(pos)] = int(b[int(pos)])
    return out


def _dedupe_inference_genomes(genomes: Sequence[Genome], owner: int | None = None):
    """Return structural representatives and a row map without changing semantics."""
    if owner is None:
        return list(genomes), np.arange(len(genomes), dtype=np.int32)
    reps: List[Genome] = []
    row_map = np.empty(len(genomes), dtype=np.int32)
    buckets: dict[tuple, List[int]] = {}
    for i, g in enumerate(genomes):
        sig = genome_eval_signature(g, owner)
        found = -1
        for ri in buckets.get(sig, []):
            if (same_genome_structure(g, reps[ri])
                    and g.readout_weights == reps[ri].readout_weights):
                found = ri
                break
        if found < 0:
            found = len(reps)
            reps.append(g)
            buckets.setdefault(sig, []).append(found)
        row_map[i] = found
    return reps, row_map


def _prediction_scores_from_features(genome: Genome, x: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    pred = baseline.astype(np.float64, copy=True)
    if len(genome.readout_weights) == len(genome.rules):
        w = np.asarray(genome.readout_weights, dtype=np.float64)
        active = np.flatnonzero(np.abs(w) > 1.0e-12)
        if active.size:
            pred += x[:, active].astype(np.float64, copy=False) @ w[active]
    return pred


def _prepare_inference_mps_context(genomes: Sequence[Genome],
                                   mps_evaluator: MpsPopulationEvaluator,
                                   pool_live_genomes: Sequence[Genome] | None = None):
    """Pack one immutable genome set once for repeated candidate evaluations."""
    reps, row_map = _dedupe_inference_genomes(genomes, id(mps_evaluator))
    roots = genomes if pool_live_genomes is None else pool_live_genomes
    packer = getattr(mps_evaluator, "_pack_population", None)
    packed = packer(reps, pool_live_genomes=roots) if callable(packer) else None
    return reps, row_map, packed


def evaluate_inference_candidates(genomes: Sequence[Genome], candidates: Sequence[Sequence[int]],
                                  backend: str, max_output: int,
                                  mps_evaluator: MpsPopulationEvaluator | None = None,
                                  mps_context=None,
                                  pool_live_genomes: Sequence[Genome] | None = None) -> np.ndarray:
    """Score the same candidate strings under every genome's fitted cleanliness readout.

    ``mps_context`` is a prepacked ``(representatives, row_map, packed)`` tuple.
    Reusing it is important for the inner GA: model rules do not change between
    candidate generations, so rebuilding the candidate index every round is a
    pure host/GPU setup bottleneck.
    """
    if not genomes or not candidates:
        return np.empty((len(genomes), len(candidates)), dtype=np.float64)
    if backend == "mps":
        if mps_evaluator is None:
            raise ValueError("MPS inference requires the persistent evaluator")
        if mps_context is None:
            reps, row_map = _dedupe_inference_genomes(genomes, id(mps_evaluator))
            packed = None
        else:
            reps, row_map, packed = mps_context
            if len(row_map) != len(genomes):
                raise ValueError("inference MPS context does not match genome count")
        mps_evaluator.set_inputs(candidates)
        feats = mps_evaluator.evaluate_population(
            reps,
            pool_live_genomes=(genomes if pool_live_genomes is None else pool_live_genomes),
            prepacked=packed,
        )
        rep_scores = np.empty((len(reps), len(candidates)), dtype=np.float64)
        for i, (g, (x, baseline, _stats)) in enumerate(zip(reps, feats)):
            rep_scores[i] = _prediction_scores_from_features(g, x, baseline)
        return rep_scores[np.asarray(row_map, dtype=np.int32)]

    scores = np.empty((len(genomes), len(candidates)), dtype=np.float64)
    for gi, g in enumerate(genomes):
        x, baseline, _stats, _states = trajectory_features_cpu(candidates, g, max_output)
        scores[gi] = _prediction_scores_from_features(g, x, baseline)
    return scores


def _row_rank01(values: np.ndarray) -> np.ndarray:
    """Per-row average-rank normalization in [0,1], with neutral ties.

    Exact ties are common early in training when no active rule distinguishes two
    candidate strings.  Assigning ordinal ranks to tied values leaked candidate
    list order into GA selection.  Average ranks make an all-tied row contribute
    exactly 0.5 to every candidate instead.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("rank normalization expects a 2-D matrix")
    m = values.shape[1]
    if m <= 1:
        return np.ones_like(values, dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)
    den = float(m - 1)
    for i in range(values.shape[0]):
        out[i] = (average_ranks(values[i]) - 1.0) / den
    return out


def _inference_elite_ids(ensemble_score: Sequence[float], elite_count: int) -> np.ndarray:
    """Return candidate ids copied byte-for-byte into the next inner-GA generation.

    Keep at least one true elite whenever a candidate population exists, but for
    width>1 reserve at least one slot for offspring.  Stable sorting makes exact
    score ties deterministic instead of allowing NumPy's default quicksort order
    to change which candidate survives.
    """
    score = np.asarray(ensemble_score, dtype=np.float64)
    width = int(score.size)
    if width <= 0:
        return np.empty(0, dtype=np.int32)
    max_elites = 1 if width == 1 else width - 1
    keep = min(max_elites, max(1, int(elite_count)))
    return np.argsort(-score, kind="stable")[:keep].astype(np.int32, copy=False)


def run_string_inference_ga(genomes: Sequence[Genome], inference_set: RollingInferenceSet,
                            sampler: CorpusSampler, population_size: int, generations: int,
                            ensemble_size: int, backend: str, max_output: int,
                            mps_evaluator: MpsPopulationEvaluator | None = None,
                            restore_inputs: Sequence[Sequence[int]] | None = None,
                            case_count: int | None = None,
                            pool_live_genomes: Sequence[Genome] | None = None,
                            elite_count: int = 3) -> dict:
    """Inner GA over corrupted bytes; final recovery is objective #2.

    Search is shared by the strongest Spearman ensemble.  Earlier v60 code paid
    to score *all* ~450 genomes on every inner generation even though only the
    top ensemble votes affected candidate evolution.  This version evaluates
    only that ensemble during the search, then performs exactly one full-population
    pass on the final candidate pool.  Besides being substantially cheaper, this
    matches the requested semantics more closely: the second objective is the
    recovery accuracy of the GA's *result*, not the best accidental intermediate
    candidate seen at any prior inner generation.
    """
    t0 = time.perf_counter()
    n_genomes = len(genomes)
    max_cases = len(inference_set.slots)
    use_cases = max_cases if case_count is None else max(1, min(max_cases, int(case_count)))
    cases = inference_set.slots[:use_cases]
    psize = max(2, int(population_size))
    rounds = max(1, int(generations))
    if n_genomes == 0 or not cases:
        return {
            "seconds": 0.0, "evaluations": 0, "ensemble_accuracy": float("nan"),
            "model_jobs": 0, "kernel_jobs": 0, "search_rounds": 0,
            "equivalent_work": 0.0,
        }

    # Preserve training diagnostics: inference intentionally reuses the same MPS
    # evaluator/rule pool for speed, but profiler history should still describe
    # the outer training evaluation.  Rule-pool state itself is not rolled back.
    diag_attrs = (
        "last_pack_seconds", "last_pack_cache_rows", "last_pack_rebuilt_rows",
        "last_rule_pack_memo_hits", "last_rule_pack_memo_misses",
        "last_dispatch_seconds", "last_pool_upload_seconds", "last_pool_new_rules",
        "last_pool_new_tokens", "last_pool_gc_seconds", "last_pool_gc_triggered",
        "last_pool_gc_reason", "last_pool_gc_rules_before", "last_pool_gc_rules_after",
        "last_pool_gc_tokens_before", "last_pool_gc_tokens_after",
        "last_pool_gc_alloc_before", "last_pool_gc_alloc_after", "last_profile",
    )
    saved_diag = None
    if backend == "mps" and mps_evaluator is not None:
        saved_diag = {k: (dict(getattr(mps_evaluator, k)) if k == "last_profile"
                          else getattr(mps_evaluator, k)) for k in diag_attrs}

    candidate_pops: List[List[List[int]]] = []
    for case in cases:
        pop = [list(case.noisy)]
        seen = {tuple(case.noisy)}
        attempts = 0
        positions = list(case.mutable_positions)
        while len(pop) < psize:
            # Cover corrupted loci round-robin before spending proposals on
            # duplicates.  This turns the first population into a set of local
            # counterfactuals instead of 11 unrelated multi-byte perturbations.
            focus = positions[(len(pop) - 1) % len(positions)] if positions else None
            cand = mutate_inference_candidate(
                case.noisy, case.mutable_positions, sampler, focus_position=focus
            )
            key = tuple(cand)
            if key in seen:
                cand = mutate_inference_candidate(
                    cand, case.mutable_positions, sampler, force_changes=1,
                    focus_position=focus,
                )
                key = tuple(cand)
            attempts += 1
            # The gene space can be tiny in smoke tests (e.g. one corrupted
            # byte).  Do not spin forever trying to force uniqueness.
            if key in seen and attempts < psize * 16:
                continue
            seen.add(key)
            pop.append(cand)
        candidate_pops.append(pop)

    top_ids = np.argsort(np.asarray([float(g.fitness) for g in genomes], dtype=np.float64))[::-1]
    top_ids = top_ids[:max(1, min(len(top_ids), int(ensemble_size)))]
    search_genomes = [genomes[int(i)] for i in top_ids]
    roots = genomes if pool_live_genomes is None else pool_live_genomes

    search_context = None
    total_candidate_evaluations = 0
    total_model_jobs = 0
    total_kernel_jobs = 0
    executed_rounds = 0
    final_ensemble_accuracy = float("nan")
    oracle_accuracy = float("nan")
    selected_change_rate = float("nan")

    try:
        if backend == "mps":
            if mps_evaluator is None:
                raise ValueError("MPS inference requires the persistent evaluator")
            search_context = _prepare_inference_mps_context(search_genomes, mps_evaluator, roots)

        # Search phase: only the voting ensemble is required here.
        for step in range(rounds):
            flat = [cand for pop in candidate_pops for cand in pop]
            search_scores = evaluate_inference_candidates(
                search_genomes, flat, backend, max_output,
                mps_evaluator=mps_evaluator, mps_context=search_context,
                pool_live_genomes=roots,
            )
            executed_rounds += 1
            total_candidate_evaluations += len(flat)
            total_model_jobs += len(search_genomes) * len(flat)
            if backend == "mps" and search_context is not None:
                total_kernel_jobs += len(search_context[0]) * len(flat)
            else:
                total_kernel_jobs += len(search_genomes) * len(flat)

            next_pops: List[List[List[int]]] = []
            offset = 0
            for case, pop in zip(cases, candidate_pops):
                width = len(pop)
                local = search_scores[:, offset:offset + width]
                ensemble_score = np.mean(_row_rank01(local), axis=0)

                if step + 1 < rounds:
                    # Explicit elitist preservation for the string-side GA.
                    # These candidates are copied unchanged; mutation/crossover
                    # is applied only to the remaining slots.  The helper also
                    # guarantees at least one offspring slot when width>1.
                    elite_ids = _inference_elite_ids(ensemble_score, elite_count)
                    elite_n = int(elite_ids.size)
                    elites = [list(pop[int(i)]) for i in elite_ids]
                    new_pop = [list(x) for x in elites]
                    seen = {tuple(x) for x in new_pop}
                    ranked_ids = np.argsort(-ensemble_score, kind="stable")
                    parent_pool = ranked_ids[:max(elite_n, width // 2)]
                    attempts = 0
                    while len(new_pop) < psize:
                        # Bias reproduction toward exact elites without ever
                        # mutating the elite copies already stored in new_pop.
                        if elite_n and random.random() < 0.70:
                            p1 = pop[int(random.choice(elite_ids))]
                        else:
                            p1 = pop[int(random.choice(parent_pool))]
                        if len(parent_pool) > 1 and random.random() < 0.65:
                            p2 = pop[int(random.choice(parent_pool))]
                            child = crossover_inference_candidates(p1, p2, case.mutable_positions)
                        else:
                            child = list(p1)
                        # Systematically revisit every corrupted locus.  A small
                        # GA population otherwise spends many generations touching
                        # only a subset of positions, making exact recovery zero
                        # for purely combinatorial reasons.
                        positions = list(case.mutable_positions)
                        child_slot = max(0, len(new_pop) - elite_n)
                        focus = (positions[(step * max(1, psize - elite_n) + child_slot) % len(positions)]
                                 if positions else None)
                        child = mutate_inference_candidate(
                            child, case.mutable_positions, sampler,
                            focus_position=focus if random.random() < 0.88 else None,
                        )
                        key = tuple(child)
                        attempts += 1
                        if key in seen and attempts < psize * 8:
                            continue
                        seen.add(key)
                        new_pop.append(child)
                    next_pops.append(new_pop[:psize])
                offset += width

            if step + 1 >= rounds:
                break
            candidate_pops = next_pops

        # Objective phase: one full-population evaluation of the *final* candidate
        # pool.  Hidden clean bytes are still used only after model scoring.
        final_flat = [cand for pop in candidate_pops for cand in pop]
        final_context = None
        if backend == "mps":
            assert mps_evaluator is not None
            # A search-pack GC can invalidate cached pack rows for genomes outside
            # the small voting ensemble.  Repacking those rows while the evaluator
            # still contains inference candidates would choose semantically-valid
            # but poorly tuned anchors for future training.  Restore the outer
            # training inputs before the one full-population pack so any rebuilt
            # anchor/filter heuristics are selected on the real training batch.
            if restore_inputs is not None:
                mps_evaluator.set_inputs(restore_inputs)
            final_context = _prepare_inference_mps_context(genomes, mps_evaluator, roots)
        final_scores = evaluate_inference_candidates(
            genomes, final_flat, backend, max_output,
            mps_evaluator=mps_evaluator, mps_context=final_context,
            pool_live_genomes=roots,
        )
        total_candidate_evaluations += len(final_flat)
        total_model_jobs += n_genomes * len(final_flat)
        if backend == "mps" and final_context is not None:
            total_kernel_jobs += len(final_context[0]) * len(final_flat)
        else:
            total_kernel_jobs += n_genomes * len(final_flat)

        per_case_acc = np.zeros((n_genomes, len(cases)), dtype=np.float64)
        per_case_full = np.zeros((n_genomes, len(cases)), dtype=np.float64)
        ensemble_case_acc: List[float] = []
        oracle_case_acc: List[float] = []
        selected_changed = 0.0
        selected_total = 0
        offset = 0
        for ci, (case, pop) in enumerate(zip(cases, candidate_pops)):
            width = len(pop)
            local = final_scores[:, offset:offset + width]
            acc = np.asarray([_inference_recovery_accuracy(c, case) for c in pop], dtype=np.float64)
            full = np.asarray([_inference_full_accuracy(c, case) for c in pop], dtype=np.float64)

            # v60.1: score *all* exactly top-rated candidates uniformly instead
            # of np.argmax's hidden index-order tie break.  Candidate 0 is the
            # untouched noisy string, so np.argmax made a flat/quantized readout
            # choose "do nothing" for every genome and could pin inferBest at
            # exactly 0.000 even when recoveries were present in the candidate
            # pool.  Averaging over the max-score tie set is the expected accuracy
            # of an unbiased uniform tie-break; the clean target never changes
            # which candidates belong to that set.
            maxima = np.max(local, axis=1, keepdims=True)
            tie = np.isclose(local, maxima, rtol=1.0e-12, atol=1.0e-12)
            tie_n = np.maximum(1, tie.sum(axis=1)).astype(np.float64)
            per_case_acc[:, ci] = (tie.astype(np.float64) @ acc) / tie_n
            per_case_full[:, ci] = (tie.astype(np.float64) @ full) / tie_n
            oracle_case_acc.append(float(np.max(acc)) if acc.size else 0.0)
            if width:
                noisy_tuple = tuple(case.noisy)
                changed_mask = np.asarray([tuple(c) != noisy_tuple for c in pop], dtype=np.float64)
                selected_changed += float(np.sum((tie.astype(np.float64) @ changed_mask) / tie_n))
                selected_total += int(tie.shape[0])

            ensemble_local = local[top_ids]
            ensemble_score = np.mean(_row_rank01(ensemble_local), axis=0)
            ens_max = float(np.max(ensemble_score))
            ens_tie = np.isclose(ensemble_score, ens_max, rtol=1.0e-12, atol=1.0e-12)
            ensemble_case_acc.append(float(np.mean(acc[ens_tie])) if np.any(ens_tie) else 0.0)
            offset += width

        per_genome = np.mean(per_case_acc, axis=1)
        per_genome_full = np.mean(per_case_full, axis=1)
        for g, a, fa in zip(genomes, per_genome.tolist(), per_genome_full.tolist()):
            g.inference_accuracy = float(a)
            g.inference_full_accuracy = float(fa)
        final_ensemble_accuracy = (
            float(statistics.fmean(ensemble_case_acc)) if ensemble_case_acc else float("nan")
        )
        oracle_accuracy = (
            float(statistics.fmean(oracle_case_acc)) if oracle_case_acc else float("nan")
        )
        selected_change_rate = (
            float(selected_changed) / float(selected_total) if selected_total else float("nan")
        )

    finally:
        if backend == "mps" and mps_evaluator is not None:
            if restore_inputs is not None:
                mps_evaluator.set_inputs(restore_inputs)
            if saved_diag is not None:
                for k, v in saved_diag.items():
                    setattr(mps_evaluator, k, v)

    elapsed = float(time.perf_counter() - t0)
    equiv = total_model_jobs / float(max(1, n_genomes))
    return {
        "seconds": elapsed,
        "evaluations": int(total_candidate_evaluations),
        "ensemble_accuracy": final_ensemble_accuracy,
        # Diagnostic only: best truth-match present anywhere in the final
        # candidate pool.  It never participates in search or selection.  If
        # oracle>0 while inferBest==0, the models saw a recoverable candidate but
        # ranked it incorrectly; if both are 0, proposal/search is the bottleneck.
        "oracle_accuracy": oracle_accuracy,
        "selected_change_rate": selected_change_rate,
        "mean_accuracy": float(np.mean(per_genome)) if per_genome.size else float("nan"),
        "best_accuracy": float(np.max(per_genome)) if per_genome.size else float("nan"),
        "model_jobs": int(total_model_jobs),
        "kernel_jobs": int(total_kernel_jobs),
        "search_rounds": int(executed_rounds),
        "equivalent_work": float(equiv),
        "active_cases": int(len(cases)),
        "candidate_population": int(psize),
    }


def assign_pareto_metrics(population: Sequence[Genome]) -> List[List[int]]:
    """NSGA-II nondominated rank + crowding for (Spearman, inference accuracy).

    The first v60 implementation used nested Python loops and tiny NumPy calls for
    every genome pair.  At population=450 that alone cost ~0.8 s/generation on
    the audit machine.  Two objectives fit comfortably in a 450x450 boolean
    domination matrix, so compute all pair relations in vectorized NumPy and peel
    fronts from the resulting counts.  Semantics are identical but host overhead
    falls by roughly two orders of magnitude.
    """
    n = len(population)
    if n == 0:
        return []
    obj = np.asarray([
        [float(g.fitness) if math.isfinite(float(g.fitness)) else -1.0,
         float(g.inference_accuracy) if math.isfinite(float(g.inference_accuracy)) else -1.0]
        for g in population
    ], dtype=np.float64)

    # dominates[i,j] iff i is no worse on every objective and strictly better on
    # at least one.  n=450 -> ~0.2M pair cells, tiny compared with model features.
    ge = obj[:, None, :] >= obj[None, :, :]
    gt = obj[:, None, :] > obj[None, :, :]
    dominates = np.all(ge, axis=2) & np.any(gt, axis=2)
    dominated_by = dominates.sum(axis=0, dtype=np.int32)
    remaining = np.ones(n, dtype=bool)

    fronts: List[List[int]] = []
    rank = 0
    current = np.flatnonzero(remaining & (dominated_by == 0))
    while current.size:
        front = current.astype(np.int32, copy=False).tolist()
        fronts.append(front)
        for i in front:
            population[i].pareto_rank = rank
        remaining[current] = False
        # Remove every outgoing domination edge from this whole front at once.
        dominated_by -= dominates[current].sum(axis=0, dtype=np.int32)
        current = np.flatnonzero(remaining & (dominated_by == 0))
        rank += 1

    # Defensive fallback: finite two-objective inputs should always be exhausted.
    # If a future caller feeds NaNs despite the sanitization above, do not leave
    # stale Pareto metadata on unassigned genomes.
    if np.any(remaining):
        tail = np.flatnonzero(remaining).astype(np.int32, copy=False).tolist()
        fronts.append(tail)
        for i in tail:
            population[i].pareto_rank = rank

    for front in fronts:
        for i in front:
            population[i].pareto_crowding = 0.0
        if len(front) <= 2:
            for i in front:
                population[i].pareto_crowding = float("inf")
            continue
        for axis in (0, 1):
            ids = sorted(front, key=lambda i: obj[i, axis])
            lo, hi = float(obj[ids[0], axis]), float(obj[ids[-1], axis])
            span = hi - lo
            # A constant objective carries no diversity information.  Marking
            # arbitrary first/last rows as infinite crowding here used to bias
            # selection whenever inference accuracy tied (very common because it
            # is a discrete recovery fraction, and always true under
            # --no-inference).  Skip the axis entirely when its span is zero.
            if span <= 1.0e-15:
                continue

            # Recovery accuracy is intentionally discrete, so several genomes can
            # share an exact boundary value.  Give every true min/max boundary
            # point the same NSGA-II protection instead of whichever equal-valued
            # row happened to sort first or last.
            eps = max(1.0e-15, abs(span) * 1.0e-12)
            for i in ids:
                v = float(obj[i, axis])
                if abs(v - lo) <= eps or abs(v - hi) <= eps:
                    population[i].pareto_crowding = float("inf")

            for k in range(1, len(ids) - 1):
                gi = population[ids[k]]
                if not math.isinf(gi.pareto_crowding):
                    gi.pareto_crowding += float((obj[ids[k + 1], axis] - obj[ids[k - 1], axis]) / span)
    return fronts


def pareto_sort_key(g: Genome):
    inf_acc = float(g.inference_accuracy) if math.isfinite(float(g.inference_accuracy)) else -1.0
    crowd = float(g.pareto_crowding)
    return (int(g.pareto_rank), -crowd, -float(g.fitness), -inf_acc)


def pareto_better(a: Genome, b: Genome) -> bool:
    return pareto_sort_key(a) < pareto_sort_key(b)


def choose_pareto_knee(population: Sequence[Genome]) -> Genome:
    front = [g for g in population if int(g.pareto_rank) == 0]
    if not front:
        return max(population, key=lambda g: g.fitness)
    f = np.asarray([g.fitness for g in front], dtype=np.float64)
    a = np.asarray([g.inference_accuracy for g in front], dtype=np.float64)
    fr = max(1.0e-12, float(f.max() - f.min()))
    ar = max(1.0e-12, float(a.max() - a.min()))
    fn = (f - f.min()) / fr
    an = (a - a.min()) / ar
    dist = (1.0 - fn) ** 2 + (1.0 - an) ** 2
    return front[int(np.argmin(dist))]


def select_outer_elites(population: Sequence[Genome], elite_count: int) -> List[Genome]:
    """Choose genomes copied structurally unchanged into the next generation.

    NSGA-II crowding normally protects objective extremes, but make that guarantee
    explicit: with enough elite slots, preserve the Pareto knee plus the best
    Spearman and best inference individuals, then fill the remaining slots in
    normal Pareto order.  This is *survivor elitism*, independent of HoF/QD.
    """
    if not population:
        return []
    keep = max(1, min(int(elite_count), len(population)))
    ordered = sorted(population, key=pareto_sort_key)
    inference_best = max(
        population,
        key=lambda g: float(g.inference_accuracy)
        if math.isfinite(float(g.inference_accuracy)) else -1.0,
    )
    anchors = [choose_pareto_knee(population), max(population, key=lambda g: g.fitness), inference_best]
    out: List[Genome] = []
    seen: set[int] = set()
    for g in anchors + ordered:
        ident = id(g)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(g)
        if len(out) >= keep:
            break
    return out


# ---------------------------------------------------------------------------
# Evolution loop
# ---------------------------------------------------------------------------


def save_genome(path: str, genome: Genome, generation: int) -> None:
    payload = {
        "generation": generation,
        "fitness": genome.fitness,
        "case_scores": genome.case_scores,
        "inference_accuracy": genome.inference_accuracy,
        "inference_full_accuracy": genome.inference_full_accuracy,
        "pareto_rank": genome.pareto_rank,
        "pareto_crowding": genome.pareto_crowding,
        "embedding": genome.embedding,
        "rules": [
            {
                "a": r.pattern,
                "b": r.replacement,
                "weight": (genome.readout_weights[i] if len(genome.readout_weights) == len(genome.rules) else r.weight),
            }
            for i, r in enumerate(genome.rules)
        ],
    }
    tmp = path + ".tmp"
    Path(tmp).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _pack_genome_group(genomes: Sequence[Genome], prefix: str, arrays: dict) -> None:
    if not genomes:
        arrays[prefix + "count"] = np.asarray([0], dtype=np.int32)
        return
    pcount = len(genomes)
    rcount = len(genomes[0].rules)
    pat_offsets = [0]
    rep_offsets = [0]
    pat_tokens: List[int] = []
    rep_tokens: List[int] = []
    weights = np.empty((pcount, rcount), dtype=np.float32)
    embeddings = np.empty((pcount, EMBEDDING_ENTRY_COUNT), dtype=np.int16)
    fitness = np.empty(pcount, dtype=np.float64)
    inference_accuracy = np.empty(pcount, dtype=np.float64)
    inference_full_accuracy = np.empty(pcount, dtype=np.float64)
    pareto_rank = np.empty(pcount, dtype=np.int32)
    pareto_crowding = np.empty(pcount, dtype=np.float64)
    score_offsets = [0]
    score_values: List[float] = []
    for gi, g in enumerate(genomes):
        if len(g.rules) != rcount:
            raise ValueError("checkpoint requires fixed rule count")
        embeddings[gi] = np.asarray(g.embedding, dtype=np.int16)
        fitness[gi] = float(g.fitness)
        inference_accuracy[gi] = float(g.inference_accuracy)
        inference_full_accuracy[gi] = float(g.inference_full_accuracy)
        pareto_rank[gi] = int(g.pareto_rank)
        pareto_crowding[gi] = float(g.pareto_crowding)
        score_values.extend(map(float, g.case_scores))
        score_offsets.append(len(score_values))
        for ri, rule in enumerate(g.rules):
            pat_tokens.extend(map(int, rule.pattern)); pat_offsets.append(len(pat_tokens))
            rep_tokens.extend(map(int, rule.replacement)); rep_offsets.append(len(rep_tokens))
            if len(g.readout_weights) == rcount:
                weights[gi, ri] = float(g.readout_weights[ri])
            else:
                weights[gi, ri] = float(rule.weight)
    arrays[prefix + "count"] = np.asarray([pcount], dtype=np.int32)
    arrays[prefix + "rules"] = np.asarray([rcount], dtype=np.int32)
    arrays[prefix + "pat_offsets"] = np.asarray(pat_offsets, dtype=np.int64)
    arrays[prefix + "rep_offsets"] = np.asarray(rep_offsets, dtype=np.int64)
    arrays[prefix + "pat_tokens"] = np.asarray(pat_tokens, dtype=np.int16)
    arrays[prefix + "rep_tokens"] = np.asarray(rep_tokens, dtype=np.int16)
    arrays[prefix + "weights"] = weights
    arrays[prefix + "embeddings"] = embeddings
    arrays[prefix + "fitness"] = fitness
    arrays[prefix + "inference_accuracy"] = inference_accuracy
    arrays[prefix + "inference_full_accuracy"] = inference_full_accuracy
    arrays[prefix + "pareto_rank"] = pareto_rank
    arrays[prefix + "pareto_crowding"] = pareto_crowding
    arrays[prefix + "score_offsets"] = np.asarray(score_offsets, dtype=np.int32)
    arrays[prefix + "score_values"] = np.asarray(score_values, dtype=np.float64)


def _unpack_genome_group(z, prefix: str) -> List[Genome]:
    count = int(z[prefix + "count"][0])
    if count == 0:
        return []
    rcount = int(z[prefix + "rules"][0])
    po = z[prefix + "pat_offsets"]
    ro = z[prefix + "rep_offsets"]
    pt = z[prefix + "pat_tokens"]
    rt = z[prefix + "rep_tokens"]
    weights = z[prefix + "weights"]
    embeddings = z[prefix + "embeddings"]
    fitness = z[prefix + "fitness"]
    inference_accuracy = (z[prefix + "inference_accuracy"]
                          if prefix + "inference_accuracy" in z.files
                          else np.full(count, np.nan, dtype=np.float64))
    inference_full_accuracy = (z[prefix + "inference_full_accuracy"]
                               if prefix + "inference_full_accuracy" in z.files
                               else np.full(count, np.nan, dtype=np.float64))
    pareto_rank = (z[prefix + "pareto_rank"]
                   if prefix + "pareto_rank" in z.files
                   else np.full(count, 2_147_483_647, dtype=np.int32))
    pareto_crowding = (z[prefix + "pareto_crowding"]
                       if prefix + "pareto_crowding" in z.files
                       else np.zeros(count, dtype=np.float64))
    so = z[prefix + "score_offsets"]
    sv = z[prefix + "score_values"]
    out: List[Genome] = []
    q = 0
    for gi in range(count):
        rules: List[Rule] = []
        for ri in range(rcount):
            a = pt[int(po[q]):int(po[q + 1])].astype(np.int32).tolist()
            b = rt[int(ro[q]):int(ro[q + 1])].astype(np.int32).tolist()
            rules.append(Rule(a, b, float(weights[gi, ri])))
            q += 1
        scores = sv[int(so[gi]):int(so[gi + 1])].astype(np.float64).tolist()
        out.append(Genome(
            rules=rules,
            embedding=embeddings[gi].astype(np.int32).tolist(),
            fitness=float(fitness[gi]),
            case_scores=scores,
            readout_weights=weights[gi].astype(np.float64).tolist(),
            inference_accuracy=float(inference_accuracy[gi]),
            inference_full_accuracy=float(inference_full_accuracy[gi]),
            pareto_rank=int(pareto_rank[gi]),
            pareto_crowding=float(pareto_crowding[gi]),
        ))
    return out


def _bytes_array(obj) -> np.ndarray:
    return np.frombuffer(pickle.dumps(obj, protocol=5), dtype=np.uint8)


def _array_object(a: np.ndarray):
    return pickle.loads(np.asarray(a, dtype=np.uint8).tobytes())


def save_checkpoint(path: str, next_generation: int, population: Sequence[Genome],
                    best_ever: Genome | None, history: Sequence[dict],
                    sampler: CorpusSampler, args,
                    hof_archive: Sequence[HallOfFameEntry] | None = None) -> None:
    if not path:
        return
    arrays: dict = {
        "version": np.asarray([1], dtype=np.int32),
        "next_generation": np.asarray([int(next_generation)], dtype=np.int64),
        "history": _bytes_array(list(history)),
        "py_rng": _bytes_array(random.getstate()),
        "np_rng": _bytes_array(np.random.get_state()),
        "torch_rng": (torch.get_rng_state().cpu().numpy().astype(np.uint8, copy=False)
                      if torch is not None else np.empty(0, dtype=np.uint8)),
        "sampler_ngrams": _bytes_array(sampler.ngrams),
        "sampler_byte_buffer": np.asarray(sampler._byte_buffer, dtype=np.uint16),
        "sampler_byte_pos": np.asarray([sampler._byte_pos], dtype=np.int64),
        "config": _bytes_array({
            "population": len(population),
            "rules": len(population[0].rules) if population else 0,
            "cases": args.cases,
            "samples": args.samples,
            "max_noise": args.max_noise,
            "max_output": args.max_output,
        }),
    }
    _pack_genome_group(population, "pop_", arrays)
    _pack_genome_group([] if best_ever is None else [best_ever], "best_", arrays)
    # v22 persists HoF structures, but not rolling score-cache serials.  The
    # evaluation set intentionally restarts after resume, so stale serial ids
    # would be meaningless and could collide with the new rolling set.
    _pack_genome_group(
        [] if hof_archive is None else [e.genome for e in hof_archive],
        "hof_", arrays,
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def load_checkpoint(path: str, sampler: CorpusSampler):
    with np.load(path, allow_pickle=False) as z:
        version = int(z["version"][0])
        if version != 1:
            raise ValueError(f"unsupported checkpoint version {version}")
        next_generation = int(z["next_generation"][0])
        population = _unpack_genome_group(z, "pop_")
        best_group = _unpack_genome_group(z, "best_")
        best_ever = best_group[0] if best_group else None
        hof_group = _unpack_genome_group(z, "hof_") if "hof_count" in z.files else []
        history = list(_array_object(z["history"]))
        py_rng = _array_object(z["py_rng"])
        np_rng = _array_object(z["np_rng"])
        torch_rng = np.asarray(z["torch_rng"], dtype=np.uint8).copy()
        sampler.ngrams = list(_array_object(z["sampler_ngrams"]))
        # Contextual inference proposals are derived from the persisted n-gram
        # pool.  Rebuild after replacing the constructor's temporary pool so a
        # resumed run uses exactly the checkpointed proposal model.
        sampler._rebuild_context_model()
        sampler._byte_buffer = np.asarray(z["sampler_byte_buffer"], dtype=np.uint16).copy()
        sampler._byte_pos = int(z["sampler_byte_pos"][0])
        config = dict(_array_object(z["config"]))
    random.setstate(py_rng)
    np.random.set_state(np_rng)
    if torch is not None and torch_rng.size:
        torch.set_rng_state(torch.from_numpy(torch_rng))
    return next_generation, population, best_ever, history, config, hof_group


def _moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """Trailing moving average with an expanding prefix.

    The old implementation returned NaN for the first window-1 generations,
    which made a 75-generation MA appear broken during early training.  Here
    generation i averages the last min(window, i+1) finite values, so the line
    is defined from generation 0 and becomes a conventional fixed-width MA once
    enough history exists.
    """
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return a
    w = max(1, int(window))
    out = np.full_like(a, np.nan, dtype=np.float64)
    finite = np.isfinite(a)
    vals = np.where(finite, a, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(vals, dtype=np.float64)))
    ccount = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    for i in range(a.size):
        lo = max(0, i + 1 - w)
        count = int(ccount[i + 1] - ccount[lo])
        if count > 0:
            out[i] = (csum[i + 1] - csum[lo]) / float(count)
    return out


def append_history_csv(path: str, row: dict) -> None:
    """Append history while safely extending the CSV schema.

    Newer versions add pool/profiler/GC columns. If an older fitness_history.csv exists,
    rewrite it once with the union of old and new headers instead of silently
    producing rows with a different column count.
    """
    p = Path(path)
    fields = list(row.keys())
    if p.exists() and p.stat().st_size > 0:
        with p.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            if any(k not in old_fields for k in fields):
                old_rows = list(reader)
                merged = old_fields + [k for k in fields if k not in old_fields]
                tmp = p.with_suffix(p.suffix + ".tmp")
                with tmp.open("w", newline="", encoding="utf-8") as out:
                    writer = csv.DictWriter(out, fieldnames=merged)
                    writer.writeheader()
                    writer.writerows(old_rows)
                os.replace(tmp, p)
                fields = merged
            else:
                fields = old_fields
    exists = p.exists() and p.stat().st_size > 0
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def write_plots(history: List[dict], prefix: str, ma_window: int) -> None:
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot: matplotlib unavailable; install with `pip install matplotlib`", file=sys.stderr)
        return

    gen = np.asarray([h["generation"] for h in history])
    best = np.asarray([h["best"] for h in history], dtype=float)
    mean = np.asarray([h["mean"] for h in history], dtype=float)
    median = np.asarray([h["median"] for h in history], dtype=float)
    best_ever = np.asarray([h["best_ever"] for h in history], dtype=float)
    ma = _moving_average(best, ma_window)

    base = str(Path(prefix))
    Path(base).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    ax.plot(gen, best, label="best", linewidth=1.2)
    ax.plot(gen, mean, label="mean", linewidth=1.0)
    ax.plot(gen, median, label="median", linewidth=1.0)
    ax.plot(gen, best_ever, label="best-ever", linewidth=1.2)
    ax.plot(gen, ma, label=f"best MA({ma_window})", linewidth=1.5)
    ax.set_xlabel("generation")
    ax.set_ylabel("Spearman fitness")
    ax.set_ylim(-1.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title("Replacement GP fitness")
    fig.tight_layout()
    tmp = base + "_fitness.tmp.png"
    out = base + "_fitness.png"
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    os.replace(tmp, out)

    # Old Nim-style saturation view plus v60 inference recovery on a second axis.
    def sat(v):
        return -np.log2(np.maximum(1e-9, 1.0 - np.minimum(v, 1.0 - 1e-9)))
    inference_raw = np.asarray([float(h.get("inference_raw", np.nan)) for h in history], dtype=float)
    inference_best = np.asarray([float(h.get("inference_best", np.nan)) for h in history], dtype=float)
    inference_ma = _moving_average(inference_raw, ma_window)
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    ax.plot(gen, sat(best), label="best", color="lightgray")
    ax.plot(gen, sat(best_ever), label="best-ever")
    ax.plot(gen, sat(np.where(np.isfinite(ma), ma, best)), label=f"best MA({ma_window})")
    ax.set_xlabel("generation")
    ax.set_ylabel("-log2(1 - Spearman)")
    ax.grid(True, alpha=0.25)

    have_inference = np.any(np.isfinite(inference_raw))
    if have_inference:
        ax2 = ax.twinx()
        ax2.plot(gen, inference_raw, label="inference raw", color="gray", alpha=0.38, linewidth=1.0)
        ax2.plot(gen, inference_best, label="inference best", linestyle="--", linewidth=1.25)
        ax2.plot(gen, inference_ma, label=f"inference MA({ma_window})", linewidth=1.5)
        ax2.set_ylabel("corrupted-byte recovery accuracy")
        ax2.set_ylim(-0.02, 1.02)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], loc="best")
    else:
        ax.legend()
    ax.set_title("Fitness near saturation + string-side inference recovery")
    fig.tight_layout()
    tmp = base + "_saturation.tmp.png"
    out = base + "_saturation.png"
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    os.replace(tmp, out)

    moved = np.asarray([h["embedding_moved"] for h in history], dtype=float)
    latent = np.asarray([h["embedding_latent"] for h in history], dtype=float)
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    ax.plot(gen, moved, label="bytes moved from identity")
    ax.plot(gen, latent, label="bytes mapped into latent tokens")
    ax.set_xlabel("generation")
    ax.set_ylabel("embedding entries")
    ax.set_ylim(0, EMBEDDING_ENTRY_COUNT)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title("Champion embedding evolution")
    fig.tight_layout()
    tmp = base + "_embedding.tmp.png"
    out = base + "_embedding.png"
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    os.replace(tmp, out)

    # Trajectory profiler / rule-pool diagnostics (available on v14 MPS runs).
    cand = np.asarray([float(h.get("prof_candidates_mean", np.nan)) for h in history])
    rounds = np.asarray([float(h.get("prof_rounds_mean", np.nan)) for h in history])
    scan = np.asarray([float(h.get("prof_scan_tokens_mean", np.nan)) for h in history])
    new_rules = np.asarray([float(h.get("pool_new_rules", np.nan)) for h in history])
    have_profiler = any(float(h.get("mps_seconds", 0.0) or 0.0) > 0.0 for h in history)
    if have_profiler and np.any(np.isfinite(cand)):
        fig = plt.figure(figsize=(11, 6))
        ax = fig.add_subplot(111)
        ax.plot(gen, cand, label="candidate rules / trajectory")
        ax.plot(gen, scan, label="candidate-scan tokens / trajectory")
        ax.set_xlabel("generation")
        ax.set_ylabel("work")
        ax.grid(True, alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(gen, rounds, label="rounds / trajectory", linestyle="--")
        ax2.plot(gen, new_rules, label="new pooled rules / generation", linestyle=":")
        ax2.set_ylabel("rounds / new rules")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], loc="best")
        ax.set_title("GPU trajectory profiler and global rule pool")
        fig.tight_layout()
        tmp = base + "_profiler.tmp.png"
        out = base + "_profiler.png"
        fig.savefig(tmp, dpi=140)
        plt.close(fig)
        os.replace(tmp, out)


def evolve(args) -> Genome:
    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    if torch is not None:
        torch.manual_seed(args.seed)

    if args.backend == "mps":
        if not (torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")

    corpus = load_corpus(args)
    t_stage = time.perf_counter()
    print("init: building corpus sampler...", flush=True)
    sampler = CorpusSampler(corpus)
    print(f"init: corpus sampler ready ({time.perf_counter() - t_stage:.2f}s)", flush=True)

    if args.backend == "mps":
        t_stage = time.perf_counter()
        print("init: compiling Metal trajectory shader...", flush=True)
        _get_mps_trajectory_lib()
        print(f"init: Metal shader library ready ({time.perf_counter() - t_stage:.2f}s)", flush=True)

    start_generation = 0
    best_ever: Genome | None = None
    history: List[dict] = []
    loaded_hof_genomes: List[Genome] = []

    if args.load:
        t_stage = time.perf_counter()
        print(f"init: loading checkpoint {args.load}...", flush=True)
        start_generation, population, best_ever, history, cfg, loaded_hof_genomes = load_checkpoint(args.load, sampler)
        if not population:
            raise ValueError("checkpoint contains an empty population")
        args.population = len(population)
        args.rules = len(population[0].rules)
        migrated = sum(stabilize_genome_nonexpanding(g, sampler) for g in population)
        migrated_best = 0
        if best_ever is not None:
            migrated_best = stabilize_genome_nonexpanding(best_ever, sampler)
        print(
            f"init: checkpoint loaded at generation {start_generation}; "
            f"population={args.population}, rules={args.rules}, "
            f"nonexpanding_migration={migrated} rules"
            + (f" (+{migrated_best} best-ever)" if migrated_best else "")
            + (f", hof={len(loaded_hof_genomes)}" if loaded_hof_genomes else "")
            + f" ({time.perf_counter() - t_stage:.2f}s)", flush=True,
        )
    else:
        t_stage = time.perf_counter()
        print(
            f"init: creating population ({args.population} genomes x {args.rules} rules = "
            f"{args.population * args.rules:,} rules)...", flush=True,
        )
        population: List[Genome] = []
        init_tick = max(1, args.population // 20)
        for i in range(args.population):
            population.append(random_genome(args.rules, sampler, not args.no_embedding))
            if args.progress and ((i + 1) % init_tick == 0 or i + 1 == args.population):
                print(f"  population {i + 1}/{args.population}", end="\r", flush=True)
        if args.progress:
            print(" " * 48, end="\r")
        print(f"init: population ready ({time.perf_counter() - t_stage:.2f}s)", flush=True)

    # Runtime-only mutation pressure. v22 uses a rolling evaluation set rather
    # than replacing the whole test batch at once: by default 8 cases x 20
    # trajectories, with exactly one case slot replaced every 2 generations.
    # Stagnation is measured only on slots whose serial identity exists in both
    # adjacent generations, so a newly sampled hard/easy case cannot fake a
    # regression/improvement. Runtime rolling state intentionally restarts on
    # checkpoint resume; the model population/checkpoint format stays compatible.
    stagnation = 0
    rolling_dataset: RollingEvaluationSet | None = None
    dataset_epoch = 0
    previous_case_scores: List[float] | None = None
    previous_case_serials: Tuple[int, ...] | None = None
    overlap_delta = float("nan")
    overlap_cases = 0

    # v60 second-objective state. Inference cases are independent from the
    # Spearman rolling set and rotate slowly; the inner GA itself reruns every
    # generation so each new genome must earn its recovery score.
    inference_set: RollingInferenceSet | None = None
    inference_epoch = 0
    inference_best_ever = max(
        [float(h.get("inference_best_ever", h.get("inference_best", float("nan"))))
         for h in history if math.isfinite(float(h.get("inference_best_ever", h.get("inference_best", float("nan"))) or float("nan")))]
        or [0.0]
    )

    # Persistent rolling elite-of-elites.  Structures survive checkpoints; their
    # case-score history restarts because the rolling evaluation serials restart.
    args.hof_size = max(args.hof_size, len(loaded_hof_genomes))
    hof_archive: List[HallOfFameEntry] = [
        HallOfFameEntry(clone_genome_deep(g), born_generation=start_generation - 1)
        for g in loaded_hof_genomes[:max(0, int(args.hof_size))]
    ]

    for entry in hof_archive:
        entry.archive_score = float(entry.genome.fitness)
    plateau = PlateauTracker(args.plateau_window, args.plateau_min_gain)
    for row in history[-2 * args.plateau_window:]:
        plateau.update(float(row["best"]))

    # v58 controllers are runtime-only so v57 checkpoints remain directly
    # loadable. Bandit cumulative statistics are restored from the last history
    # row, while the sparse linkage model is relearned from the first evaluated
    # population after resume.
    operator_arms = ("mix_local", "mix_balanced", "local", "balanced", "explore", "legacy")
    op_bandit = AdaptiveOperatorBandit(operator_arms, args.operator_ucb, args.operator_epsilon)
    op_bandit.restore_from_history(history)
    linkage_model = SparseLinkageModel()

    mps_evaluator = None
    if args.backend == "mps":
        # Match RollingEvaluationSet's coercions exactly.  Using raw CLI values
        # under-allocated the MPS sample buffers for --samples 1/2 (the rolling
        # set enforces at least 3) and for --cases <= 0 (it enforces at least 1).
        sample_count = max(1, min(int(args.cases), len(corpus))) * max(3, int(args.samples))
        t_stage = time.perf_counter()
        print(
            f"init: allocating v33 dual-anchor-bloom v31 narrow-state wc1-inplace fused-hash incremental-pack inplace-literal anchor-guided wc1-fast score-reuse direct-diff local-generic-candidate variable-length GPU-index rolling-HoF bounded-workload compacting-GC precise-mutation nonexpanding pooled MPS evaluator "
            f"(genome_batch={args.mps_genome_batch}, result_chunk={args.mps_result_chunk}, "
            f"threadgroup={args.mps_threadgroup}, samples={sample_count})...", flush=True,
        )
        effective_raw_len = effective_trajectory_span(args, corpus)
        mps_evaluator = MpsPopulationEvaluator(
            sample_count=sample_count,
            rule_count=args.rules,
            max_raw_len=effective_raw_len,
            max_output=args.max_output,
            genome_batch=args.mps_genome_batch,
            result_chunk=args.mps_result_chunk,
            progress=args.progress,
            threadgroup_size=args.mps_threadgroup,
            pool_max_rules=args.mps_rule_pool_max,
            pool_gc_ratio=args.mps_rule_pool_gc_ratio,
            pool_gc_min_dead=args.mps_rule_pool_gc_min_dead,
        )
        print(f"init: MPS evaluator ready ({time.perf_counter() - t_stage:.2f}s)", flush=True)

    for generation in range(start_generation, args.generations):
        t0 = time.perf_counter()
        rel_generation = generation - start_generation
        dataset_changed = False
        rotated_slot = -1
        if rolling_dataset is None:
            if args.progress:
                print(f"gen={generation}: building initial rolling noise trajectories...", flush=True)
            rolling_dataset = RollingEvaluationSet(
                corpus, sampler, args.cases, args.samples, args.max_noise, args.trajectory_len
            )
            dataset_changed = True
        else:
            rotate_every = max(1, int(args.case_rotate_every))
            if rel_generation > 0 and (rel_generation % rotate_every) == 0:
                rotated_slot = ((rel_generation // rotate_every) - 1) % rolling_dataset.case_count
                old_serial, new_serial = rolling_dataset.rotate(rotated_slot)
                dataset_epoch += 1
                dataset_changed = True
                if args.progress:
                    print(
                        f"gen={generation}: rolling case slot {rotated_slot} "
                        f"serial {old_serial}->{new_serial}...", flush=True,
                    )
        inputs, target, case_ids, sample_ids = rolling_dataset.flatten()
        current_case_serials = rolling_dataset.serials
        evaluated_hof = random.sample(hof_archive, min(len(hof_archive), max(0, args.hof_eval)))
        hof_genomes = [e.genome for e in evaluated_hof]
        eval_genomes = list(population) + hof_genomes
        if args.progress:
            lens = np.asarray([len(x) for x in inputs], dtype=np.int32)
            print(
                f"gen={generation}: evaluating {args.population} population + {len(hof_genomes)} hof genomes "
                f"on {len(inputs)} trajectories (roll={dataset_epoch}, changed={int(dataset_changed)}, "
                f"len={int(lens.min())}..{int(lens.max())})...",
                flush=True,
            )

        outer_eval_t0 = time.perf_counter()
        mps_kernel_seconds = 0.0
        mps_pack_seconds = 0.0
        readout_seconds = 0.0
        eval_reused = 0
        eval_deduped = 0
        eval_unique = len(eval_genomes)
        if args.backend == "mps":
            assert mps_evaluator is not None
            if dataset_changed:
                mps_evaluator.set_inputs(inputs)

            # A sampled archive entry can be many epochs old: only an exact
            # match of case identities permits reuse, even on non-rotation steps.
            reusable: List[int] = []
            pending: List[int] = []
            expected_cases = rolling_dataset.case_count if rolling_dataset is not None else 0
            for i, g in enumerate(eval_genomes):
                can_reuse = (
                    g._eval_case_serials == tuple(current_case_serials)
                    and math.isfinite(g.fitness)
                    and len(g.case_scores) == expected_cases
                    and len(g.readout_weights) == len(g.rules)
                )
                if can_reuse:
                    reusable.append(i)
                else:
                    pending.append(i)
            eval_reused = len(reusable)

            # Exact structural dedupe for the remaining jobs. Reusable genomes
            # seed the map too, so a nominally-new clone/crossover that is
            # structurally identical to a survivor gets its exact score for free.
            owner = id(mps_evaluator)
            sig_to_reps: dict[tuple, List[int]] = {}
            for i in reusable:
                sig = genome_eval_signature(eval_genomes[i], owner)
                sig_to_reps.setdefault(sig, []).append(i)

            unique_pending: List[int] = []
            duplicate_of: dict[int, int] = {}
            for i in pending:
                g = eval_genomes[i]
                sig = genome_eval_signature(g, owner)
                match = -1
                for j in sig_to_reps.get(sig, []):
                    if same_genome_structure(g, eval_genomes[j]):
                        match = j
                        break
                if match >= 0:
                    duplicate_of[i] = match
                    eval_deduped += 1
                else:
                    unique_pending.append(i)
                    sig_to_reps.setdefault(sig, []).append(i)

            eval_unique = len(unique_pending)
            tb = time.perf_counter()
            if unique_pending:
                eval_batch = [eval_genomes[i] for i in unique_pending]
                all_features = mps_evaluator.evaluate_population(
                    eval_batch, pool_live_genomes=list(population) + [e.genome for e in hof_archive]
                )
                mps_kernel_seconds = time.perf_counter() - tb
                mps_pack_seconds = mps_evaluator.last_pack_seconds
                tr = time.perf_counter()
                total_eval = len(eval_batch)
                for pos, (idx, (x, baseline, _stats)) in enumerate(zip(unique_pending, all_features)):
                    genome = eval_genomes[idx]
                    score_genome_features(
                        genome, x, baseline, target, case_ids, sample_ids, args.ridge_lambda,
                        args.readout_mode, args.rank_max_features,
                    )
                    genome._eval_case_serials = tuple(current_case_serials)
                    if args.progress and ((pos + 1) % max(1, total_eval // 20) == 0 or pos + 1 == total_eval):
                        print(f"  readout {pos+1}/{total_eval}", end="\r", flush=True)
                readout_seconds = time.perf_counter() - tr
                del all_features
            else:
                # Avoid reporting stale timing/profile data from the previous gen.
                mps_evaluator.last_pack_seconds = 0.0
                mps_evaluator.last_pack_cache_rows = 0
                mps_evaluator.last_pack_rebuilt_rows = 0
                mps_evaluator.last_dispatch_seconds = 0.0
                mps_evaluator.last_pool_upload_seconds = 0.0
                mps_evaluator.last_pool_new_rules = 0
                mps_evaluator.last_pool_new_tokens = 0
                mps_evaluator.last_profile = {}
                mps_kernel_seconds = time.perf_counter() - tb
                mps_pack_seconds = 0.0

            # Copy only genome-local readout/score state. Structural rules remain
            # shared/independent exactly as bred; no Rule.weight mutation occurs.
            for i, j in duplicate_of.items():
                dst = eval_genomes[i]
                src = eval_genomes[j]
                dst.fitness = float(src.fitness)
                dst.case_scores = list(src.case_scores)
                dst.readout_weights = list(src.readout_weights)
                dst._eval_case_serials = src._eval_case_serials
        else:
            total_eval = len(eval_genomes)
            eval_tick = max(1, total_eval // 20)
            for i, genome in enumerate(eval_genomes):
                evaluate_genome(
                    genome, inputs, target, case_ids, sample_ids,
                    args.backend, args.ridge_lambda, args.max_output, args.readout_mode,
                    args.rank_max_features,
                )
                genome._eval_case_serials = tuple(current_case_serials)
                if args.progress and ((i + 1) % eval_tick == 0 or i + 1 == total_eval):
                    print(f"  eval {i+1}/{total_eval}", end="\r", flush=True)
        if args.progress:
            print(" " * 56, end="\r")
        outer_eval_seconds = max(1.0e-9, time.perf_counter() - outer_eval_t0)

        # Only sampled entries have current observations. Never attach a fresh
        # case serial to an unobserved stale score.
        for entry in evaluated_hof:
            refresh_hof_entry(
                entry, current_case_serials, args.hof_history_cases, args.hof_current_weight
            )

        # Learn operator utility from the outer Spearman evaluation we already paid
        # for. The v60 inference cases are a different rolling test, so mixing them
        # into delayed operator credit would compare non-identical tasks.
        operator_updates, operator_reward_mean = op_bandit.observe_population(population)

        # v60 checked: bounded string-side genetic inference.  Candidate search
        # is paid only for the voting ensemble; all genomes are scored once on
        # the final candidate pool.  A hardware-independent equivalent-work cap
        # keeps CPU/MPS evolution deterministic under the same RNG seed while
        # leaving ample headroom below the requested ~2x ceiling.
        inference_stats = {
            "seconds": 0.0, "evaluations": 0, "ensemble_accuracy": float("nan"),
            "oracle_accuracy": float("nan"), "selected_change_rate": float("nan"),
            "mean_accuracy": float("nan"), "best_accuracy": float("nan"),
            "model_jobs": 0, "kernel_jobs": 0, "search_rounds": 0,
            "equivalent_work": 0.0,
        }
        inference_rotated_slot = -1
        inf_cases = 0
        inf_pop = 0
        inf_gens = 0
        inf_work = 0
        if args.no_inference:
            for g in population:
                g.inference_accuracy = 0.0
                g.inference_full_accuracy = 0.0
        else:
            if inference_set is None:
                evaluator_span = effective_trajectory_span(args, corpus)
                max_inf_cases = max(
                    1,
                    min(int(args.inference_cases), len(corpus), max(1, len(inputs) // 2)),
                )
                inference_set = RollingInferenceSet(
                    corpus, sampler, max_inf_cases,
                    min(args.inference_span, evaluator_span), args.inference_noise,
                    excluded_sources=[c.source_index for c in rolling_dataset.slots],
                )
                print(
                    f"init: inference set cases={max_inf_cases} span<={min(args.inference_span, evaluator_span)} "
                    f"noise={args.inference_noise:.3f}", flush=True,
                )
            else:
                rotate_every = max(1, int(args.inference_rotate_every))
                if rel_generation > 0 and (rel_generation % rotate_every) == 0:
                    inference_rotated_slot = ((rel_generation // rotate_every) - 1) % inference_set.case_count
                    inference_set.rotate(
                        inference_rotated_slot,
                        excluded_sources=[c.source_index for c in rolling_dataset.slots],
                    )
                    inference_epoch += 1

            # Budget against one full outer-population evaluation.  Keeping this
            # hardware-independent is intentional: CPU and MPS must consume the
            # same RNG stream and evolve the same population under a fixed seed.
            inf_cases, inf_pop, inf_gens, inf_work = resolve_inference_budget(
                len(inputs), inference_set.case_count, args.inference_population,
                args.inference_generations, args.inference_budget_ratio,
                genome_count=len(population),
                outer_evaluated_genomes=len(population),
                ensemble_size=args.inference_ensemble,
            )
            live_pool_roots = list(population) + [e.genome for e in hof_archive]
            inference_stats = run_string_inference_ga(
                population, inference_set, sampler, inf_pop, inf_gens,
                args.inference_ensemble, args.backend, args.max_output,
                mps_evaluator=mps_evaluator, restore_inputs=inputs,
                case_count=inf_cases, pool_live_genomes=live_pool_roots,
                elite_count=args.inference_elites,
            )
            inference_stats["budget_equivalent"] = int(inf_work)
            inference_stats["budget_cases"] = int(inf_cases)
            inference_stats["budget_population"] = int(inf_pop)
            inference_stats["budget_generations"] = int(inf_gens)

        fronts = assign_pareto_metrics(population)
        population.sort(key=pareto_sort_key)
        spearman_sorted = sorted(population, key=lambda g: g.fitness, reverse=True)
        search_model_t0 = time.perf_counter()
        linkage_model.refresh(
            population, generation, args.linkage_refresh, args.linkage_elites,
            args.linkage_loci, args.linkage_modules, args.linkage_max_module,
        )
        qd_grid = build_qd_grid(population, args.qd_bins)
        qd_elites = sorted(qd_grid.values(), key=pareto_sort_key)
        search_model_seconds = time.perf_counter() - search_model_t0

        # HoF remains a long-term rank-representation archive; feed it the top
        # Spearman candidates while the live population itself is NSGA-II Pareto.
        hof_candidate_n = max(0, min(int(args.hof_candidates), len(population)))
        hof_admitted, hof_replaced = update_hall_of_fame(
            hof_archive, spearman_sorted[:hof_candidate_n], current_case_serials, generation,
            args.hof_size, args.hof_history_cases, args.hof_current_weight, args.hof_min_distance,
        )
        hof_best_score = hof_archive[0].archive_score if hof_archive else float("nan")
        current_hof = [e for e in hof_archive if e.genome._eval_case_serials == tuple(current_case_serials)]
        hof_best_current = max((e.current_fitness for e in current_hof), default=float("nan"))
        champion = spearman_sorted[0]
        pareto_champion = choose_pareto_knee(population)
        expanding_rules = sum(1 for r in champion.rules if not is_rule_nonexpanding(r))
        if expanding_rules:
            raise AssertionError(f"v22 invariant broken: champion has {expanding_rules} expanding rules")
        # Global best remains a convenient "best observed" checkpoint, but it
        # is intentionally NOT used for mutation pressure because different
        # dataset epochs are different tests.
        improved = best_ever is None or champion.fitness > best_ever.fitness + 1.0e-12
        if improved:
            best_ever = clone_genome_shallow(champion)
            save_genome(args.save, best_ever, generation)

        # Compare only case slots that are literally the same evaluation cases
        # in both adjacent generations. On a rotation generation this excludes
        # the one fresh slot (7/8 overlap by default); on the second generation
        # of a pair all 8/8 slots are comparable.
        stagnation = short_stagnation if generation > start_generation else 0
        current_case_scores = list(map(float, champion.case_scores))
        overlap_delta = float("nan")
        overlap_cases = 0
        if previous_case_scores is not None and previous_case_serials is not None:
            common = [
                i for i, serial in enumerate(current_case_serials)
                if i < len(previous_case_serials)
                and i < len(previous_case_scores)
                and i < len(current_case_scores)
                and serial == previous_case_serials[i]
            ]
            overlap_cases = len(common)
            if common:
                prev_overlap = statistics.fmean(previous_case_scores[i] for i in common)
                curr_overlap = statistics.fmean(current_case_scores[i] for i in common)
                overlap_delta = float(curr_overlap - prev_overlap)
                if overlap_delta > float(args.stagnation_epsilon):
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation = 0
        else:
            stagnation = 0
        short_stagnation = stagnation
        long_stagnation, plateau_gain = plateau.update(champion.fitness)
        stagnation = max(short_stagnation, long_stagnation)
        previous_case_scores = current_case_scores
        previous_case_serials = current_case_serials

        inference_values = [float(g.inference_accuracy) for g in population
                            if math.isfinite(float(g.inference_accuracy))]
        inference_best = max(inference_values) if inference_values else float("nan")
        inference_mean = statistics.fmean(inference_values) if inference_values else float("nan")
        if math.isfinite(inference_best):
            inference_best_ever = max(float(inference_best_ever), float(inference_best))
        pareto_front_size = len(fronts[0]) if fronts else 0

        elapsed = time.perf_counter() - t0
        mean_fit = statistics.fmean(g.fitness for g in population)
        med_fit = statistics.median(g.fitness for g in population)
        active_weights = champion.readout_weights if len(champion.readout_weights) == len(champion.rules) else [r.weight for r in champion.rules]
        active = sum(1 for w in active_weights if abs(w) > 1e-12)
        moved, latent = embedding_metrics(champion.embedding)
        bemoved, belatent = embedding_metrics(best_ever.embedding)
        row = {
            "generation": generation,
            "best": champion.fitness,
            "mean": mean_fit,
            "median": med_fit,
            "best_ever": best_ever.fitness,
            "inference_raw": float(pareto_champion.inference_accuracy),
            "inference_best": inference_best,
            "inference_best_ever": inference_best_ever,
            "inference_mean": inference_mean,
            "inference_full_raw": float(pareto_champion.inference_full_accuracy),
            "inference_ensemble": float(inference_stats.get("ensemble_accuracy", float("nan"))),
            "inference_oracle": float(inference_stats.get("oracle_accuracy", float("nan"))),
            "inference_selected_change_rate": float(inference_stats.get("selected_change_rate", float("nan"))),
            "inference_seconds": float(inference_stats.get("seconds", 0.0)),
            "inference_evaluations": int(inference_stats.get("evaluations", 0)),
            "inference_model_jobs": int(inference_stats.get("model_jobs", 0)),
            "inference_kernel_jobs": int(inference_stats.get("kernel_jobs", 0)),
            "inference_equivalent_work": float(inference_stats.get("equivalent_work", 0.0)),
            "inference_budget_equivalent": int(inference_stats.get("budget_equivalent", 0)),
            "inference_budget_cases": int(inference_stats.get("budget_cases", inf_cases)),
            "inference_budget_population": int(inference_stats.get("budget_population", inf_pop)),
            "inference_budget_generations": int(inference_stats.get("budget_generations", inf_gens)),
            "inference_search_rounds": int(inference_stats.get("search_rounds", 0)),
            "inference_outer_eval_seconds": float(outer_eval_seconds),
            "inference_time_ratio": float(inference_stats.get("seconds", 0.0)) / max(1.0e-9, float(outer_eval_seconds)),
            "inference_epoch": inference_epoch,
            "inference_rotated_slot": inference_rotated_slot,
            "pareto_front_size": pareto_front_size,
            "pareto_champion_fitness": float(pareto_champion.fitness),
            "pareto_champion_inference": float(pareto_champion.inference_accuracy),
            "active_rules": active,
            "expanding_rules": expanding_rules,
            "mutation_stagnation": stagnation,
            "plateau_gain": plateau_gain,
            "hof_evaluated": len(evaluated_hof),
            "dataset_epoch": dataset_epoch,
            "rolling_rotated_slot": rotated_slot,
            "rolling_overlap_cases": overlap_cases,
            "rolling_overlap_delta": overlap_delta,
            "hof_size": len(hof_archive),
            "hof_best_score": hof_best_score,
            "hof_best_current": hof_best_current,
            "hof_admitted": hof_admitted,
            "hof_replaced": hof_replaced,
            "qd_cells": len(qd_grid),
            "linkage_modules": len(linkage_model.modules),
            "linkage_loci": linkage_model.loci,
            "operator_updates": operator_updates,
            "operator_reward_mean": operator_reward_mean,
            "search_model_seconds": search_model_seconds,
            "trajectory_min_len": min(map(len, inputs)) if inputs else 0,
            "trajectory_max_len": max(map(len, inputs)) if inputs else 0,
            "embedding_moved": moved,
            "embedding_latent": latent,
            "best_ever_embedding_moved": bemoved,
            "best_ever_embedding_latent": belatent,
            "seconds": elapsed,
            "mps_seconds": mps_kernel_seconds if args.backend == "mps" else 0.0,
            "mps_pack_seconds": mps_pack_seconds if args.backend == "mps" else 0.0,
            "mps_dispatch_seconds": (mps_evaluator.last_dispatch_seconds if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "mps_pool_upload_seconds": (mps_evaluator.last_pool_upload_seconds if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "pool_rules": (len(mps_evaluator._pool_meta_host) if args.backend == "mps" and mps_evaluator is not None else 0),
            "pool_new_rules": (mps_evaluator.last_pool_new_rules if args.backend == "mps" and mps_evaluator is not None else 0),
            "pool_new_tokens": (mps_evaluator.last_pool_new_tokens if args.backend == "mps" and mps_evaluator is not None else 0),
            "pool_gc": (1 if args.backend == "mps" and mps_evaluator is not None and mps_evaluator.last_pool_gc_triggered else 0),
            "pool_gc_count": (mps_evaluator.pool_gc_count if args.backend == "mps" and mps_evaluator is not None else 0),
            "pool_gc_seconds": (mps_evaluator.last_pool_gc_seconds if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "pool_gc_rules_before": (mps_evaluator.last_pool_gc_rules_before if args.backend == "mps" and mps_evaluator is not None else 0),
            "pool_gc_rules_after": (mps_evaluator.last_pool_gc_rules_after if args.backend == "mps" and mps_evaluator is not None else 0),
            "pool_gc_tokens_before": (mps_evaluator.last_pool_gc_tokens_before if args.backend == "mps" and mps_evaluator is not None else 0),
            "pool_gc_tokens_after": (mps_evaluator.last_pool_gc_tokens_after if args.backend == "mps" and mps_evaluator is not None else 0),
            "pool_gc_mps_before": (mps_evaluator.last_pool_gc_alloc_before if args.backend == "mps" and mps_evaluator is not None else 0),
            "pool_gc_mps_after": (mps_evaluator.last_pool_gc_alloc_after if args.backend == "mps" and mps_evaluator is not None else 0),
            "prof_rounds_mean": (mps_evaluator.last_profile.get("rounds_mean", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_candidates_mean": (mps_evaluator.last_profile.get("candidate_rules_mean", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_candidate_fraction": (mps_evaluator.last_profile.get("candidate_fraction_mean", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_rebuilds_mean": (mps_evaluator.last_profile.get("candidate_rebuilds_mean", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_extensions_mean": (mps_evaluator.last_profile.get("candidate_extensions_mean", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_extension_tokens_mean": (mps_evaluator.last_profile.get("candidate_extension_tokens_mean", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_scan_tokens_mean": (mps_evaluator.last_profile.get("candidate_scan_tokens_mean", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_rewrites_mean": (mps_evaluator.last_profile.get("rewrites_mean", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_literal_fast_fraction": (mps_evaluator.last_profile.get("literal_fast_fraction", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_wc1_literal_fast_fraction": (mps_evaluator.last_profile.get("wc1_literal_fast_fraction", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_inplace_literal_fraction": (mps_evaluator.last_profile.get("inplace_literal_fraction", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_same_length_literal_fraction": (mps_evaluator.last_profile.get("same_length_literal_fraction", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_pattern_len1_literal_fraction": (mps_evaluator.last_profile.get("pattern_len1_literal_fraction", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_matches_per_rewrite": (mps_evaluator.last_profile.get("matches_per_rewrite", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_state_len_per_rewrite": (mps_evaluator.last_profile.get("state_len_per_rewrite", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_cycle_rate": (mps_evaluator.last_profile.get("cycle_rate", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "prof_overflow_rate": (mps_evaluator.last_profile.get("overflow_rate", 0.0) if args.backend == "mps" and mps_evaluator is not None else 0.0),
            "readout_seconds": readout_seconds if args.backend == "mps" else 0.0,
            "eval_unique": eval_unique if args.backend == "mps" else len(eval_genomes),
            "eval_reused": eval_reused if args.backend == "mps" else 0,
            "eval_deduped": eval_deduped if args.backend == "mps" else 0,
        }
        op_bandit.add_history(row)
        history.append(row)
        append_history_csv(args.history_csv, row)
        if not args.no_plot and (generation % max(1, args.plot_every) == 0):
            write_plots(history, args.plot_prefix, args.plot_window)
        print(
            f"gen={generation:6d} best={champion.fitness:+.6f} "
            f"mean={mean_fit:+.6f} median={med_fit:+.6f} "
            f"infer={float(pareto_champion.inference_accuracy):.3f} "
            f"inferBest={inference_best:.3f} "
            f"infEns={float(inference_stats.get('ensemble_accuracy', float('nan'))):.3f} "
            f"infOracle={float(inference_stats.get('oracle_accuracy', float('nan'))):.3f} "
            f"infMove={float(inference_stats.get('selected_change_rate', float('nan'))):.2f} "
            f"front={pareto_front_size} "
            f"infGA={float(inference_stats.get('seconds', 0.0)):.2f}s "
            f"infPlan={inf_cases}x{inf_pop}x{inf_gens}/"
            f"{int(inference_stats.get('search_rounds', 0))}r "
            f"infEq={float(inference_stats.get('equivalent_work', 0.0)):.1f}/{inf_work} "
            f"active={active}/{len(champion.rules)} expand={expanding_rules} "
            f"stag={stagnation} roll={dataset_epoch} overlap={overlap_cases}/{len(current_case_serials)} "
            f"d={overlap_delta:+.3e} hof={len(hof_archive)}/{args.hof_size} "
            f"hofScore={hof_best_score:+.6f} hofNow={hof_best_current:+.6f} "
            f"link={len(linkage_model.modules)} qd={len(qd_grid)} search={search_model_seconds:.3f}s "
            f"embed_moved={moved} latent={latent} seconds={elapsed:.2f}"
            + (f" mps={mps_kernel_seconds:.2f} pack={mps_pack_seconds:.2f} packRows={mps_evaluator.last_pack_rebuilt_rows}/{mps_evaluator.last_pack_cache_rows + mps_evaluator.last_pack_rebuilt_rows} ruleMemo={mps_evaluator.last_rule_pack_memo_hits}/{mps_evaluator.last_rule_pack_memo_hits + mps_evaluator.last_rule_pack_memo_misses} dispatch={mps_evaluator.last_dispatch_seconds:.2f} "
               f"pool={len(mps_evaluator._pool_meta_host)}(+{mps_evaluator.last_pool_new_rules}) "
               f"upload={mps_evaluator.last_pool_upload_seconds:.3f} "
               + (f"gc={mps_evaluator.last_pool_gc_rules_before}->{mps_evaluator.last_pool_gc_rules_after}/"
                  f"{mps_evaluator.last_pool_gc_seconds:.2f}s " if mps_evaluator.last_pool_gc_triggered else "")
               + f"readout={readout_seconds:.2f} "
               + f"eval={eval_unique}/{len(eval_genomes)} reuse={eval_reused} dedupe={eval_deduped} "
               + f"inputs={mps_evaluator.unique_input_count}/{mps_evaluator.sample_count}"
               if args.backend == "mps" and mps_evaluator is not None else "")
        )
        if args.backend == "mps" and mps_evaluator is not None and mps_evaluator.last_profile:
            pr = mps_evaluator.last_profile
            print(
                f"  profile rounds={pr.get('rounds_mean', 0.0):.2f} "
                f"cand={pr.get('candidate_rules_mean', 0.0):.1f} "
                f"cand%={100.0*pr.get('candidate_fraction_mean', 0.0):.2f}% "
                f"rebuild={pr.get('candidate_rebuilds_mean', 0.0):.2f} "
                f"extend={pr.get('candidate_extensions_mean', 0.0):.2f} "
                f"scan={pr.get('candidate_scan_tokens_mean', 0.0):.0f} "
                f"rewrite={pr.get('rewrites_mean', 0.0):.2f} "
                f"litfast={100.0*pr.get('literal_fast_fraction', 0.0):.1f}% "
                f"wc1fast={100.0*pr.get('wc1_literal_fast_fraction', 0.0):.1f}% "
                f"inplace={100.0*pr.get('inplace_literal_fraction', 0.0):.1f}% "
                f"filterskip={100.0*pr.get('secondary_filter_skip_fraction', 0.0):.1f}% "
                f"same={100.0*pr.get('same_length_literal_fraction', 0.0):.1f}% "
                f"len1={100.0*pr.get('pattern_len1_literal_fraction', 0.0):.1f}% "
                f"matches/rw={pr.get('matches_per_rewrite', 0.0):.2f} "
                f"n/rw={pr.get('state_len_per_rewrite', 0.0):.0f} "
                f"cycle={100.0*pr.get('cycle_rate', 0.0):.1f}% "
                f"overflow={100.0*pr.get('overflow_rate', 0.0):.1f}%"
            )

        # Offspring construction used to deep-copy 1500 Rule objects for nearly
        # every child, causing a large unreported pause after the generation log.
        # Use copy-on-write instead and time it explicitly.
        tb = time.perf_counter()
        if args.progress:
            print(f"gen={generation}: breeding next population...", flush=True)
        elite_n = max(1, min(args.elites, args.population))
        elite_sources = select_outer_elites(population, elite_n)
        # True elite preservation: these structural genomes are copied unchanged
        # and never enter the mutation/crossover path below.  Copy-on-write Rule
        # sharing keeps this cheap while preserving the exact program/embedding.
        next_pop = [clone_genome_shallow(g) for g in elite_sources]

        # MAP-Elites-style survivor grid: every injected genome was already
        # evaluated in this generation.  Diversity therefore consumes population
        # slots, never additional Metal evaluations.
        qd_injected = 0
        for g in qd_elites:
            if qd_injected >= max(0, int(args.qd_inject)) or len(next_pop) >= args.population:
                break
            if any(same_genome_structure(g, x) for x in next_pop):
                continue
            next_pop.append(clone_genome_shallow(g))
            qd_injected += 1

        # Preserve only a small number of archive champions.  HoF remains large
        # for memory/diversity, but large injection would crowd out new search.
        hof_injected = 0
        for entry in sorted(current_hof, key=lambda e: e.current_fitness, reverse=True):
            if hof_injected >= max(0, int(args.hof_inject)) or len(next_pop) >= args.population:
                break
            if any(same_genome_structure(entry.genome, g) for g in next_pop):
                continue
            next_pop.append(clone_genome_deep(entry.genome))
            hof_injected += 1

        mutation_regime_counts = {"local": 0, "balanced": 0, "explore": 0}
        operator_counts = {a: 0 for a in operator_arms}
        crossover_kind_counts = {"two_point": 0, "diff_merge": 0}
        diff_merge_rows = 0
        mutation_rows_changed = 0
        mixed_rows = 0
        hof_parent_uses = 0
        qd_parent_uses = 0
        breed_tick = max(1, args.population // 10)

        specialist = CaseSelector(population)
        exploration_pressure = min(1.0, stagnation / 36.0)
        immigrant_rate = min(
            1.0,
            max(0.0, args.immigrant_rate) + args.plateau_immigrants * exploration_pressure,
        )

        def pick_parent():
            nonlocal hof_parent_uses, qd_parent_uses
            if current_hof and random.random() < max(0.0, min(1.0, float(args.hof_parent_rate))):
                hof_parent_uses += 1
                return tournament_hof(current_hof, args.tournament).genome, True
            if qd_elites and random.random() < max(0.0, min(1.0, float(args.qd_parent_rate))):
                qd_parent_uses += 1
                return random.choice(qd_elites), False
            if random.random() < args.specialist_rate:
                return specialist.pick(), False
            return tournament(population, args.tournament), False

        def safe_base_clone(parent: Genome, from_hof: bool) -> Genome:
            return clone_genome_deep(parent) if from_hof else clone_genome_shallow(parent)

        while len(next_pop) < args.population:
            if random.random() < immigrant_rate:
                child = random_genome(args.rules, sampler, not args.no_embedding)
                child._origin_operator = ""
                next_pop.append(child)
            else:
                p1, p1_hof = pick_parent()
                credit_parent = p1
                arm = op_bandit.choose()
                if arm.startswith("mix_") and not linkage_model.modules:
                    arm = "legacy"
                operator_counts[arm] += 1
                changed_mix = 0

                if arm.startswith("mix_"):
                    p2, _ = pick_parent()
                    # Under a serious plateau occasionally compose two learned
                    # modules before the single normal offspring evaluation.
                    mix_n = 1 + int(exploration_pressure > 0.55 and random.random() < 0.35)
                    child, changed_mix = linkage_mix(p1, p2, linkage_model, mix_n)
                    if p1_hof:
                        # Detach untouched archive Rule objects. Same-embedding
                        # donor rows remain immutable/shareable inside normal pop,
                        # but a persistent HoF lineage must stay isolated.
                        child = clone_genome_deep(child)
                    regime = "local" if arm == "mix_local" else "balanced"
                    mutation_regime_counts[regime] += 1
                    mutation_rows_changed += mutate_genome(
                        child,
                        sampler,
                        0.0 if args.no_embedding else args.embedding_mutation_rate,
                        regime=regime,
                    )
                    mixed_rows += changed_mix

                elif arm in ("local", "balanced", "explore"):
                    child = safe_base_clone(p1, p1_hof)
                    mutation_regime_counts[arm] += 1
                    mutation_rows_changed += mutate_genome(
                        child,
                        sampler,
                        0.0 if args.no_embedding else args.embedding_mutation_rate,
                        regime=arm,
                    )

                else:
                    # Mature v57 lane retained as a safety net. The bandit learns
                    # whether old two-point crossover still deserves budget.
                    if random.random() < args.crossover_rate:
                        if random.random() < min(1.0, max(0.0, args.diff_merge_share)):
                            # Three-parent categorical differential merge:
                            # apply (base -> donor) structural patches to target p1.
                            base, _base_hof = pick_parent()
                            donor, _donor_hof = pick_parent()
                            merge_target = clone_genome_deep(p1) if p1_hof else p1
                            child, merged_rows = diff_merge_crossover(
                                merge_target, base, donor, sampler,
                                0.0 if args.no_embedding else args.embedding_crossover_rate,
                            )
                            crossover_kind_counts["diff_merge"] += 1
                            diff_merge_rows += merged_rows
                        else:
                            p2, p2_hof = pick_parent()
                            if p1_hof and not p2_hof:
                                child = crossover(
                                    p2, p1,
                                    0.0 if args.no_embedding else args.embedding_crossover_rate,
                                )
                                credit_parent = p2
                            elif p1_hof and p2_hof:
                                child = crossover(
                                    clone_genome_deep(p1), p2,
                                    0.0 if args.no_embedding else args.embedding_crossover_rate,
                                )
                            else:
                                child = crossover(
                                    p1, p2,
                                    0.0 if args.no_embedding else args.embedding_crossover_rate,
                                )
                            crossover_kind_counts["two_point"] += 1
                    else:
                        child = safe_base_clone(p1, p1_hof)
                    if random.random() < args.mutation_rate:
                        regime = choose_mutation_regime(stagnation)
                        mutation_regime_counts[regime] += 1
                        mutation_rows_changed += mutate_genome(
                            child,
                            sampler,
                            0.0 if args.no_embedding else args.embedding_mutation_rate,
                            regime=regime,
                        )

                mark_offspring_origin(child, credit_parent, arm, changed_mix)
                next_pop.append(child)

            if args.progress and (len(next_pop) % breed_tick == 0 or len(next_pop) == args.population):
                print(f"  breed {len(next_pop)}/{args.population}", end="\r", flush=True)

        breed_seconds = time.perf_counter() - tb
        if args.progress:
            print(" " * 72, end="\r")
        op_summary = "/".join(f"{a}:{operator_counts[a]}" for a in operator_arms)
        print(
            f"gen={generation}: breed={breed_seconds:.2f}s elite_keep={len(elite_sources)} "
            f"mut(local/bal/explore)={mutation_regime_counts['local']}/"
            f"{mutation_regime_counts['balanced']}/{mutation_regime_counts['explore']} "
            f"rows_changed={mutation_rows_changed} mix_rows={mixed_rows} "
            f"link={len(linkage_model.modules)} qd={len(qd_grid)} qd_inject={qd_injected} "
            f"hof_inject={hof_injected} hof_parent_uses={hof_parent_uses} "
            f"qd_parent_uses={qd_parent_uses} ops[{op_summary}] "
            f"cross(2pt/merge)={crossover_kind_counts['two_point']}/{crossover_kind_counts['diff_merge']} "
            f"merge_rows={diff_merge_rows} "
            f"hof_admit={hof_admitted} hof_replace={hof_replaced}",
            flush=True,
        )

        if args.checkpoint and args.checkpoint_every > 0 and (
            (generation + 1) % args.checkpoint_every == 0
        ):
            tc = time.perf_counter()
            if args.current_save:
                save_genome(args.current_save, pareto_champion, generation)
            save_checkpoint(
                args.checkpoint, generation + 1, next_pop, best_ever,
                history, sampler, args, hof_archive,
            )
            print(f"checkpoint: saved {args.checkpoint} ({time.perf_counter()-tc:.2f}s)", flush=True)
        population = next_pop

    if args.current_save and args.generations > start_generation:
        save_genome(args.current_save, pareto_champion, args.generations - 1)
    if args.checkpoint:
        save_checkpoint(args.checkpoint, args.generations, population, best_ever, history, sampler, args, hof_archive)
    if history and not args.no_plot:
        write_plots(history, args.plot_prefix, args.plot_window)
    assert best_ever is not None
    return best_ever


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------


def self_test() -> None:
    random.seed(1)
    np.random.seed(1)
    sampler = CorpusSampler([b"abc abc xyz\n", b"hello world\n"])
    # a * b -> capture; repeated sweep should fire once then fixed-point.
    g = Genome([Rule([ord("a"), -1, ord("b")], [-1])])
    inputs = [list(b"a12b"), list(b"aXYb")]
    x, baseline, stats, out = trajectory_features_cpu(inputs, g, 128)
    assert x.shape == (2, 1)
    assert np.all(x[:, 0] == 1)
    assert baseline.tolist() == [0.0, 0.0]
    assert out == [list(b"12"), list(b"XY")]

    # Spearman / dual ridge smoke.
    xx = np.asarray([[0, 0], [1, 0], [2, 1], [3, 1]], dtype=np.float32)
    yy = np.asarray([1.0, 0.7, 0.4, 0.1])
    w = fit_dual_ridge(xx, yy, 4.0)
    assert w.shape == (2,)
    assert -1.000001 <= spearman([1, 2, 3], [3, 2, 1]) <= -0.999999
    e0 = default_embedding()
    e1 = list(e0)
    e1[65] = 300
    assert valid_embedding(e0) and valid_embedding(e1)
    tr = embedding_translation(e0, e1)
    assert sorted(tr) == list(range(VOCAB))
    assert embed_inputs([[65, 66]], e1)[0] == [300, 66]
    ma = _moving_average([1.0, 2.0, 3.0], 75)
    assert np.allclose(ma, [1.0, 1.5, 2.0])
    ma2 = _moving_average([1.0, 2.0, 3.0, 4.0], 2)
    assert np.allclose(ma2, [1.0, 1.5, 2.5, 3.5])

    # v60.3 categorical differential/git-merge smoke.  A->X changes only the
    # middle token; Y independently changed the tail, so both edits survive.
    merged, edits = merge_sequence_delta([1, 2, 3], [1, 9, 3], [1, 2, 8])
    assert merged == [1, 9, 8] and edits == 1
    # Same-locus Y edit conflicts with A->X, therefore target/Y wins.
    merged, edits = merge_sequence_delta([1, 2, 3], [1, 9, 3], [1, 7, 3])
    assert merged == [1, 7, 3] and edits == 0

    # v22 Hall-of-Fame independence/diversity smoke.
    h0 = Genome([Rule([1], [1], 0.25), Rule([2], [2], -0.5)], fitness=0.8, case_scores=[0.7, 0.9])
    e0 = make_hof_entry(h0, (10, 11), 0, 32, 0.70)
    assert genome_structural_distance(h0, e0.genome) == 0.0
    h0.rules[0].weight = 99.0
    assert e0.genome.rules[0].weight != 99.0  # deep archive isolation
    h1 = clone_genome_deep(e0.genome)
    h1.rules[0] = Rule([3], [3], 0.0)
    assert genome_structural_distance(h1, e0.genome) > 0.0
    refresh_hof_entry(e0, (10, 11), 32, 0.70)
    assert math.isfinite(e0.archive_score)
    print("self-test: PASS")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("cpu", "mps"), default="mps" if torch is not None and torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--local-corpus", default="github-code.txt")
    ap.add_argument("--corpus-chunks", type=int, default=65536*4)
    ap.add_argument("--min-chunk", type=int, default=96)
    ap.add_argument("--max-chunk", type=int, default=2000)
    ap.add_argument(
        "--trajectory-len", type=int, default=2000,
        help="maximum tokens per corpus case; shorter chunks stay short, longer chunks use a random contiguous window; 0 keeps full chunks",
    )
    ap.add_argument(
        "--case-rotate-every", type=int, default=2,
        help="replace exactly one evaluation case slot every N generations (v22 default: 2)",
    )
    ap.add_argument(
        "--stagnation-epsilon", type=float, default=1.0e-6,
        help="minimum mean Spearman gain on overlapping case slots that resets stagnation",
    )
    ap.add_argument("--cases", type=int, default=2)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--max-noise", type=float, default=0.35)
    # v60.1 string-side inference objective. Default work is 3 cases x 12 candidates
    # x 6 rounds, then automatically reduced to the --inference-budget-ratio cap.
    ap.add_argument("--inference-cases", type=int, default=3, help="independent noisy snippets optimized by the inner string GA")
    ap.add_argument("--inference-population", type=int, default=16, help="candidate strings per inference case")
    ap.add_argument("--inference-generations", type=int, default=256, help="inner GA generations per outer model generation; v60.1 default 6 remains well inside the inference work cap")
    ap.add_argument("--inference-elites", type=int, default=3, help="top candidate strings copied unchanged each inner GA generation; clamped to leave at least one offspring slot")
    ap.add_argument("--inference-span", type=int, default=2000, help="maximum bytes in each inference snippet")
    ap.add_argument("--inference-noise", type=float, default=0.012, help="fraction of snippet bytes deliberately corrupted and mutable")
    ap.add_argument("--inference-ensemble", type=int, default=64, help="top Spearman genomes voting to evolve the shared candidate pool")
    ap.add_argument("--inference-rotate-every", type=int, default=2, help="replace one inference case every N outer generations")
    ap.add_argument("--inference-budget-ratio", type=float, default=0.75, help="extra unique trajectory work relative to one outer evaluation; hard-capped at 0.85")
    ap.add_argument("--no-inference", action="store_true", help="disable v60 string inference and reduce Pareto selection to Spearman")
    ap.add_argument("--population", type=int, default=450)
    ap.add_argument("--rules", type=int, default=1500)
    ap.add_argument("--elites", type=int, default=40, help="outer Pareto elites copied structurally unchanged into the next generation")
    ap.add_argument("--hof-size", type=int, default=16384, help="rolling elite-of-elites archive capacity")
    ap.add_argument("--hof-eval", type=int, default=128, help="random archive evaluation budget per generation; 0 disables archive evaluation")
    ap.add_argument("--hof-candidates", type=int, default=1, help="top current genomes considered for HoF admission each generation")
    ap.add_argument("--hof-inject", type=int, default=4, help="freshly evaluated archived champions injected into each next population")
    ap.add_argument("--hof-parent-rate", type=float, default=0.12, help="probability each selected parent comes from the HoF")
    ap.add_argument("--hof-history-cases", type=int, default=128, help="maximum rolling case scores retained per HoF entry")
    ap.add_argument("--hof-current-weight", type=float, default=0.70, help="weight of current rolling-set fitness in HoF ranking")
    ap.add_argument("--hof-min-distance", type=float, default=0.015, help="minimum structural distance for a distinct HoF lineage")
    ap.add_argument("--plateau-window", type=int, default=100, help="compare two adjacent windows of champion fitness")
    ap.add_argument("--plateau-min-gain", type=float, default=0.0002, help="minimum long-window mean Spearman gain")
    ap.add_argument("--plateau-immigrants", type=float, default=0.04, help="maximum additional immigrant fraction under stagnation")
    ap.add_argument("--specialist-rate", type=float, default=0.25, help="parent selection probability using case-wise epsilon lexicase; 0 disables")
    ap.add_argument("--qd-bins", type=int, default=6, help="bins per MAP-Elites-style descriptor axis")
    ap.add_argument("--qd-inject", type=int, default=24, help="diverse evaluated QD cells preserved into the next population")
    ap.add_argument("--qd-parent-rate", type=float, default=0.08, help="probability of selecting a parent directly from a populated QD cell")
    ap.add_argument("--linkage-refresh", type=int, default=8, help="relearn sparse rule linkage every N generations")
    ap.add_argument("--linkage-elites", type=int, default=96, help="top evaluated genomes used to learn linkage")
    ap.add_argument("--linkage-loci", type=int, default=128, help="maximum important rule positions considered by linkage learning")
    ap.add_argument("--linkage-modules", type=int, default=96, help="maximum learned linkage modules kept")
    ap.add_argument("--linkage-max-module", type=int, default=16, help="maximum rules transplanted by one learned module")
    ap.add_argument("--operator-ucb", type=float, default=3.0e-4, help="UCB exploration scale for adaptive reproduction operators")
    ap.add_argument("--operator-epsilon", type=float, default=0.04, help="uniform exploration probability for adaptive reproduction operators")
    ap.add_argument("--tournament", type=int, default=8)
    ap.add_argument("--crossover-rate", type=float, default=0.45)
    ap.add_argument("--diff-merge-share", type=float, default=0.50, help="within crossover events, probability of A->X delta/git-style three-parent merge instead of legacy two-point crossover")
    ap.add_argument("--mutation-rate", type=float, default=0.90)
    ap.add_argument("--immigrant-rate", type=float, default=0.01)
    ap.add_argument("--embedding-mutation-rate", type=float, default=EMBEDDING_MUTATION_RATE)
    ap.add_argument("--embedding-crossover-rate", type=float, default=EMBEDDING_CROSSOVER_RATE)
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--ridge-lambda", type=float, default=RIDGE_LAMBDA)
    ap.add_argument(
        "--readout-mode", choices=("pairwise-all", "legacy-holdout"), default="pairwise-all",
        help="pairwise-all uses every trajectory for sparse rank-ridge and Spearman; legacy-holdout restores the old 1/3-vs-2/3 split",
    )
    ap.add_argument(
        "--rank-max-features", type=int, default=PAIRWISE_RANK_MAX_FEATURES,
        help="maximum rule columns kept by pairwise rank screening before the exact dual solve",
    )
    ap.add_argument("--max-output", type=int, default=MAX_OUTPUT)
    ap.add_argument(
        "--mps-genome-batch", type=int, default=512,
        help="genomes sharing scratch buffers in one Metal dispatch; 4 is conservative, 8 may improve occupancy",
    )
    ap.add_argument(
        "--mps-result-chunk", type=int, default=512,
        help="genomes queued before one MPS->CPU result synchronization; 32 balances queue depth and responsiveness",
    )
    ap.add_argument(
        "--mps-threadgroup", type=int, default=32, choices=(32, 64),
        help="cooperative Metal threads per trajectory; 32 usually maps to one Apple SIMD-group and is the v10 default",
    )
    ap.add_argument(
        "--mps-index-capacity", type=int, default=0,
        help="deprecated compatibility option; ignored in v10",
    )
    ap.add_argument(
        "--mps-rule-pool-max", type=int, default=2_000_000,
        help="hard-cap trigger for compacting the persistent global rule pool",
    )
    ap.add_argument(
        "--mps-rule-pool-gc-ratio", type=float, default=1.35,
        help="compact when pool rules exceed this multiple of current population rule references (default: 1.35)",
    )
    ap.add_argument(
        "--mps-rule-pool-gc-min-dead", type=int, default=150_000,
        help="minimum guaranteed-dead rules before ratio-triggered pool compaction (default: 150000)",
    )
    ap.add_argument("--generations", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--current-save", default="latest_minimal_gp.json", help="latest balanced Pareto-knee champion; --save remains best observed Spearman")
    ap.add_argument("--save", default="best_minimal_gp.json", help="best-ever genome JSON")
    ap.add_argument("--checkpoint", default="minimal_gp_checkpoint.npz", help="full resumable population checkpoint")
    ap.add_argument("--checkpoint-every", type=int, default=10, help="save full checkpoint every N generations; 0 disables periodic saves")
    ap.add_argument("--load", default="", help="resume from a full .npz checkpoint")
    ap.add_argument("--history-csv", default="fitness_history.csv")
    ap.add_argument("--plot-prefix", default="training")
    ap.add_argument("--plot-every", type=int, default=1)
    ap.add_argument("--plot-window", type=int, default=200)
    ap.add_argument("--no-plot", action="store_true")
    ap.set_defaults(progress=True)
    ap.add_argument("--progress", dest="progress", action="store_true", help="show initialization/evaluation progress (default)")
    ap.add_argument("--no-progress", dest="progress", action="store_false", help="suppress progress lines")
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    evolve(args)


if __name__ == "__main__":
    main()
