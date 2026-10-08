#!/usr/bin/env python3
"""Lens / Replacer: 256-byte, teacher-forced, stateful autoregressive GA.

For a text x[0..L-1]:
    state = [x[0], 0]                # final zero is an ordinary-byte prediction slot
    for t in 1..L-1:
        state = ordered_one_sweep(state, rules)
        score += (state[-1] == x[t])
        state[-1] = x[t]             # teacher forcing; preserve ALL OTHER state
        state.append(0)             # prediction slot for next step

No probability distribution, learned readout, Spearman objective, secondary
string-side GA, fixed hidden memory, context window or sequence truncation.
All visible tokens are 0..255. Embedding is a 256-element permutation LUT used
*only* within sort/+1/-1/*2//2, immediately undone by its inverse LUT.

The one-pass network permits output longer than the matched input. Rule lengths
and recurrent state lengths have no imposed model-level ceiling; they are only
limited by physical memory and representable allocation sizes. The legacy fixed-
capacity Metal kernel is used only for compatible nonexpanding 64-token rules;
otherwise the native CPU evaluator is selected automatically.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

TOKEN_COUNT = 256
RULE_COUNT = 450
MPS_MAX_RULE_TOKENS = 64  # Metal fast path only; CPU/Python have no rule-length cap
MAX_WILDCARDS = 16
PREDICTION_SLOT = 0  # byte NUL is used as the slot placeholder, not a 257th token
CHECKPOINT_VERSION = 4
_MPS_CROSSCHECK_DONE = False
_NATIVE_CROSSCHECK_DONE = False
_NATIVE_UNAVAILABLE_WARNED = False
_EXPANSION_FALLBACK_WARNED = False
# One-generation reuse of exact evaluation scores on identical training cases.
# Checkpoint formats are unaffected (this cache is never persisted).
_SCORE_CACHE_KEY = None
_SCORE_CACHE_ROWS = {}

try:
    import gpu_replace_persistent as gpu_backend
except ImportError:
    gpu_backend = None
try:
    import native_cpu
except ImportError:
    native_cpu = None

@dataclass(slots=True)
class Rule:
    pattern: list[int]
    replacement: list[int]

@dataclass(slots=True)
class Genome:
    rules: list[Rule]
    embedding: list[int] = field(default_factory=lambda: list(range(TOKEN_COUNT)))
    fitness: float = -1.0
    case_scores: list[float] = field(default_factory=list)


def valid_lut(lut: Sequence[int]) -> bool:
    return len(lut) == TOKEN_COUNT and set(lut) == set(range(TOKEN_COUNT))


def inverse_lut(lut: Sequence[int]) -> list[int]:
    if not valid_lut(lut):
        raise ValueError('Embedding LUT must be a permutation of all 256 bytes')
    inverse = [0] * TOKEN_COUNT
    for i, value in enumerate(lut):
        inverse[value] = i
    return inverse


def is_valid_rule(rule: Rule) -> bool:
    """Structural legality; do NOT impose rule length or output growth caps."""
    p, q = rule.pattern, rule.replacement
    if not p or all(v == -1 for v in p):
        return False
    if any(v != -1 and not 0 <= v < TOKEN_COUNT for v in p):
        return False
    wc = p.count(-1)
    for op in q:
        if op >= 0:
            if op >= TOKEN_COUNT:
                return False
        elif op < -111 or not wc:
            return False
        else:
            ci = ((-op - 16) % 16 + 1) if op <= -16 else -op
            if ci > wc:
                return False
    return True


def is_nonexpanding(rule: Rule) -> bool:
    """Legacy Metal fast-path test, not a constraint on legal genomes."""
    if not is_valid_rule(rule):
        return False
    p, q = rule.pattern, rule.replacement
    if len(p) > MPS_MAX_RULE_TOKENS or len(q) > MPS_MAX_RULE_TOKENS:
        return False
    if p.count(-1) > MAX_WILDCARDS:
        return False
    remaining = sum(v >= 0 for v in p)
    if sum(v >= 0 for v in q) > remaining:
        return False
    seen = set()
    for op in q:
        if op < 0:
            ci = ((-op - 16) % 16 + 1) if op <= -16 else -op
            if ci in seen:
                return False
            seen.add(ci)
    return True


def repair_rule(rule: Rule, rnd: random.Random | None = None) -> Rule:
    """Clean invalid tokens without clipping length, captures or output growth."""
    rng = rnd or random
    pattern = [(int(v) if 0 <= int(v) < TOKEN_COUNT else -1)
               for v in rule.pattern]
    if not pattern:
        pattern = [rng.randrange(TOKEN_COUNT)]
    if all(v == -1 for v in pattern):
        pattern[0] = rng.randrange(TOKEN_COUNT)
    wc = pattern.count(-1)
    replacement = []
    for raw in rule.replacement:
        op = int(raw)
        if op >= 0:
            replacement.append(op % TOKEN_COUNT)
        elif -111 <= op < 0 and wc:
            ci = ((-op - 16) % 16 + 1) if op <= -16 else -op
            if ci <= wc:
                replacement.append(op)
    if not replacement:
        replacement = [-1] if wc else [pattern[0]]
    out = Rule(pattern, replacement)
    assert is_valid_rule(out)
    return out

def repair_evolved_rule(rule: Rule, rnd: random.Random | None = None) -> Rule:
    """Keep new mutations usable instead of producing exponential blow-ups.

    NOT a model length/memory cap: manually authored rules, saved models, and
    outputs may expand arbitrarily. Mutation and crossover preferentially
    introduce one extra literal at a time, *only* for a trailing prediction-slot
    context of length >=3. Repeated captures, which can duplicate the entire
    recurrent state on every step, are not created automatically.
    """
    r = repair_rule(rule, rnd)
    wc = r.pattern.count(-1)
    used = set()
    filtered = []
    for op in r.replacement:
        if op < 0:
            ci = ((-op - 16) % 16 + 1) if op <= -16 else -op
            if ci in used:
                continue
            used.add(ci)
        filtered.append(op)
    lit_in = sum(v >= 0 for v in r.pattern)
    lit_out = sum(v >= 0 for v in filtered)
    # Expansion is allowed through a predictive guard; its state can keep
    # growing over arbitrarily many steps, with no fixed context window.
    extra = 1 if (len(r.pattern) >= 3 and r.pattern[-1] == PREDICTION_SLOT) else 0
    if lit_out > lit_in + extra:
        needed = lit_out - lit_in - extra
        for i in range(len(filtered)-1,-1,-1):
            if filtered[i] >= 0:
                del filtered[i]
                needed -= 1
                if not needed:
                    break
    return repair_rule(Rule(r.pattern, filtered), rnd)


def _match_at(state: Sequence[int], pattern: Sequence[int], start: int):
    """Greedy shortest subsequent literal; trailing wildcard consumes suffix."""
    n = len(state)
    i = 0
    pos = start
    caps = []
    while i < len(pattern) and pattern[i] >= 0:
        if pos >= n or state[pos] != pattern[i]:
            return None
        pos += 1
        i += 1
    while i < len(pattern):
        if pattern[i] != -1:
            return None
        i += 1
        cap_start = pos
        j = i
        while i < len(pattern) and pattern[i] >= 0:
            i += 1
        lit = pattern[j:i]
        if not lit:
            if i == len(pattern):
                caps.append(state[pos:])
                pos = n
            else:
                caps.append([])
            continue
        first = lit[0]
        found = -1
        for k in range(pos, n-len(lit)+1):
            if state[k] == first and state[k:k+len(lit)] == lit:
                found = k
                break
        if found < 0:
            return None
        caps.append(state[cap_start:found])
        pos = found + len(lit)
    return (pos, caps) if pos > start else None


def _emit_replacement(rep: Sequence[int], caps: Sequence[Sequence[int]],
                      lut: Sequence[int], inverse: Sequence[int]) -> list[int]:
    out = []
    for op in rep:
        if op >= 0:
            out.append(op)
            continue
        if not caps:
            continue
        if op >= -15:
            kind, ci = 0, -op-1
        elif op >= -31:
            kind, ci = 2, -op-16  # sort
        elif op >= -47:
            kind, ci = 1, -op-32  # reverse, NO LUT
        else:
            x = -op-48
            kind, ci = 3 + x//16, x%16
        if ci >= len(caps):
            ci = 0
        vals = list(caps[ci])
        if kind == 1:
            vals.reverse()
        elif kind == 2:
            vals = [inverse[k] for k in sorted(lut[x] for x in vals)]
        elif kind >= 3:
            coded = [lut[x] for x in vals]
            if kind == 3:
                coded = [(x+1)&255 for x in coded]
            elif kind == 4:
                coded = [(x-1)&255 for x in coded]
            elif kind == 5:
                coded = [(x*2)&255 for x in coded]
            elif kind == 6:
                coded = [x//2 for x in coded]
            vals = [inverse[x] for x in coded]
        out.extend(vals)
    return out


def replace_once(state: Sequence[int], rule: Rule, lut: Sequence[int], inverse: Sequence[int]) -> list[int]:
    """Apply one rule left-to-right non-overlapping, once; no memory cap."""
    pat = rule.pattern
    if not pat or not state:
        return list(state)
    n = len(state)
    out = []
    scan = 0
    prev = 0
    matched = False
    while scan < n:
        found = None
        if pat[0] >= 0:
            head = pat[0]
            for s in range(scan, n):
                if state[s] != head:
                    continue
                hit = _match_at(state, pat, s)
                if hit is not None:
                    found = (s, *hit)
                    break
        else:
            hit = _match_at(state, pat, scan)
            if hit is not None:
                found = (scan, *hit)
        if found is None:
            break
        s, finish, caps = found
        matched = True
        out.extend(state[prev:s])
        out.extend(_emit_replacement(rule.replacement, caps, lut, inverse))
        prev = finish
        scan = finish
    if not matched:
        return list(state)
    out.extend(state[prev:])
    return out


def _anchors(rules: Sequence[Rule]):
    """Sound token/pair prerequisites, independent of the current state."""
    anchors = []
    for r in rules:
        pair = next(((a,b) for a,b in zip(r.pattern,r.pattern[1:]) if a>=0 and b>=0),None)
        if pair is not None:
            anchors.append((2,pair))
        else:
            token = next((v for v in r.pattern if v>=0),None)
            anchors.append((1,token) if token is not None else (0,None))
    return anchors


def sweep(state: Sequence[int], rules: Sequence[Rule], lut: Sequence[int],
          inverse: Sequence[int] | None = None, anchors=None) -> list[int]:
    if inverse is None:
        inverse = inverse_lut(lut)
    if anchors is None:
        anchors = _anchors(rules)
    state = list(state)
    present = set(state)
    pairs = set(zip(state,state[1:]))
    for rule,(kind,anchor) in zip(rules,anchors):
        if kind==1 and anchor not in present or kind==2 and anchor not in pairs:
            continue
        nxt = replace_once(state,rule,lut,inverse)
        if nxt != state:
            state = nxt
            present = set(state)
            pairs = set(zip(state,state[1:]))
    return state


def autoregressive_rollout(genome: Genome, text: Sequence[int], *,
                           predictions: bool = False, return_states: bool = False):
    """Teacher forcing with persistent modified state; no prefix recomputation."""
    if len(text)<2:
        return (0,0, [], []) if return_states else (0,0, [])
    lut=genome.embedding
    inv=inverse_lut(lut)
    anchors=_anchors(genome.rules)
    state=[int(text[0]),PREDICTION_SLOT]
    correct=0
    preds=[]
    traces=[]
    for target in text[1:]:
        state=sweep(state,genome.rules,lut,inv,anchors)
        if not state:
            state=[PREDICTION_SLOT]
        pred=int(state[-1])
        correct+=int(pred==target)
        if predictions:
            preds.append(pred)
        if return_states:
            traces.append(list(state))
        # Only prediction slot is corrected; transformed past state is preserved.
        state[-1]=int(target)
        state.append(PREDICTION_SLOT)
    if return_states:
        return correct,len(text)-1,preds,traces
    return correct,len(text)-1,preds


def free_generate(genome: Genome, seed: bytes, count: int) -> bytes:
    """Generate bytes with no teacher forcing; state has no fixed memory limit."""
    if not seed:
        raise ValueError('seed needs at least one byte')
    if count<0:
        raise ValueError('count must be >=0')
    lut=genome.embedding
    inv=inverse_lut(lut)
    anchors=_anchors(genome.rules)
    # To condition on a multi-byte seed, first teacher-force its supplied prefix.
    state=[seed[0],PREDICTION_SLOT]
    for token in seed[1:]:
        state=sweep(state,genome.rules,lut,inv,anchors)
        if not state:
            state=[PREDICTION_SLOT]
        state[-1]=token
        state.append(PREDICTION_SLOT)
    output=bytearray()
    for _ in range(count):
        state=sweep(state,genome.rules,lut,inv,anchors)
        if not state:
            state=[PREDICTION_SLOT]
        nxt=int(state[-1])
        output.append(nxt)
        state.append(PREDICTION_SLOT)
    return bytes(output)


CORPUS_SPLIT_MARKER = b"===SPLIT==="


def load_corpus(path: str, min_length: int = 2, max_examples: int = 10000, max_length: int = 1500) -> list[bytes]:
    """Read complete byte sequences separated *only* by ===SPLIT===.

    The delimiter may appear between lines or within one line. Newlines inside
    a chunk are kept as input data; only CR/LF adjacent to its outer boundaries
    are stripped, as in the original Lens corpus parser. A corpus without the
    marker is one single document, NOT one document per line. Each accepted
    chunk remains whole: chunks with max_length or more bytes are SKIPPED, never cropped.

    Scan the file incrementally instead of materializing the entire corpus,
    allowing --corpus-chunks to stop early for large source files. The buffer
    is carried across reads, so markers spanning I/O blocks are recognized.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Corpus not found: {p}. Use --local-corpus FILE')
    if max_examples < 1:
        raise ValueError('--corpus-chunks must be at least 1')
    if max_length <= min_length:
        raise ValueError('--max-chunk must exceed --min-chunk (exclusive maximum)')
    marker = CORPUS_SPLIT_MARKER
    docs: list[bytes] = []
    pending = bytearray()

    def accept_chunk() -> bool:
        part = bytes(pending).strip(b'\r\n')
        if min_length <= len(part) < max_length:
            docs.append(part)
        pending.clear()
        return len(docs) >= max_examples

    # Split each large I/O block once in optimized C, instead of repeatedly
    # copying the entire remaining 1 MiB tail for every delimiter (quadratic
    # behavior on corpora with thousands of small ===SPLIT=== documents).
    with p.open('rb') as f:
        while block := f.read(1 << 20):
            pieces=(bytes(pending)+block).split(marker)
            for complete in pieces[:-1]:
                part=complete.strip(b'\r\n')
                if min_length <= len(part) < max_length:
                    docs.append(part)
                    if len(docs)>=max_examples:
                        return docs
            pending=bytearray(pieces[-1])
    accept_chunk()
    if not docs:
        raise ValueError(f'Corpus contains no chunks with {min_length} <= byte length < {max_length}; check --max-chunk or your ===SPLIT=== delimiters')
    return docs


def make_training_examples(corpus: Sequence[bytes], cases: int, seed: int) -> list[bytes]:
    """Legacy independent sampler; evolve() now uses RollingTrainingCases."""
    rng=random.Random(seed)
    if len(corpus)>=cases:
        return rng.sample(list(corpus),cases)
    return [rng.choice(corpus) for _ in range(cases)]


class RollingTrainingCases:
    """A fixed-size evaluation set with exactly one replacement per rotation.

    Uses a dedicated RNG, independent of GA operations, so resuming and toggling
    tqdm do not perturb evolutionary randomness. Surviving slots retain order.
    """

    def __init__(self, corpus: Sequence[bytes], cases: int, seed: int,
                 period: int, saved: dict | None = None):
        if not corpus:
            raise ValueError('Cannot rotate cases from an empty corpus')
        if cases < 1 or period < 1:
            raise ValueError('--cases and --case-rotate-every must both be positive')
        self.corpus = corpus
        self.period = int(period)
        self.rng = random.Random(seed)
        self.corpus_digest = self._digest(corpus)
        if saved is None:
            n = len(corpus)
            self.indices = (self.rng.sample(range(n), cases) if n >= cases
                            else [self.rng.randrange(n) for _ in range(cases)])
            self.rotations_done = 0
        else:
            if (saved.get('corpus_digest') != self.corpus_digest
                    or saved.get('period') != period
                    or saved.get('cases') != cases):
                raise ValueError('Checkpoint case rotation state differs from corpus, '
                                 '--cases, or --case-rotate-every; restore matching settings')
            self.indices = list(saved['indices'])
            if len(self.indices) != cases or not all(0 <= i < len(corpus) for i in self.indices):
                raise ValueError('Checkpoint has invalid rolling case indices')
            self.rotations_done = int(saved['rotations_done'])
            self.rng.setstate(saved['rng'])

    @staticmethod
    def _digest(corpus: Sequence[bytes]) -> str:
        h = hashlib.blake2b(digest_size=16)
        for sample in corpus:
            h.update(len(sample).to_bytes(8, 'little'))
            h.update(sample)
        return h.hexdigest()

    def _replace_one(self) -> None:
        slot = self.rotations_done % len(self.indices)
        old = self.indices[slot]
        n = len(self.corpus)
        in_use = set(self.indices)
        if n > len(in_use):
            # Choose uniformly from unoccupied corpus records, without scanning
            # or copying the entire corpus on every rotation.
            index = self.rng.randrange(n - len(in_use))
            for occupied in sorted(in_use):
                if occupied <= index:
                    index += 1
                else:
                    break
        elif n > 1:
            # No unused record exists (e.g. corpus size == cases). Change one
            # slot anyway, allowing duplicated records when unavoidable.
            index = self.rng.randrange(n - 1)
            if index >= old:
                index += 1
        else:
            index = old
        self.indices[slot] = index
        self.rotations_done += 1

    def for_generation(self, generation: int) -> list[bytes]:
        target = generation // self.period
        if target < self.rotations_done:
            raise ValueError('Rolling cases cannot rewind without a checkpoint')
        while self.rotations_done < target:
            self._replace_one()
        return [self.corpus[i] for i in self.indices]

    def snapshot(self) -> dict:
        return {'indices': list(self.indices), 'rotations_done': self.rotations_done,
                'rng': self.rng.getstate(), 'corpus_digest': self.corpus_digest,
                'period': self.period, 'cases': len(self.indices)}


def _bigram_seeds(corpus: Sequence[bytes], limit: int=500000) -> list[tuple[int,int]]:
    out=[]
    for line in corpus:
        out.extend(zip(line,line[1:]))
        if len(out)>=limit:
            break
    return out


def make_rule(rng: random.Random, corpus: Sequence[bytes], bigrams: Sequence[tuple[int,int]]) -> Rule:
    choice=rng.random()
    a,b=rng.choice(bigrams)
    if choice<0.36:
        return Rule([a, PREDICTION_SLOT],[a,b])
    if choice<0.53:
        return Rule([a,-1,PREDICTION_SLOT],[a,-1,b])
    if choice<0.72:
        x=rng.choice(corpus)
        if len(x) >= 2 and rng.random() < 0.80:
            pos = rng.randrange(1, len(x))
            width = rng.randint(1, min(6, pos))
            context = list(x[pos-width:pos])
            return Rule(context + [PREDICTION_SLOT], context + [x[pos]])
        size=(log_uniform_count(rng,min(len(x),48)) if rng.random()<0.28
              else rng.randint(1,min(7,len(x))))
        start=rng.randrange(len(x)-size+1)
        p=list(x[start:start+size])
        rep=p.copy()
        rep[-1]=rng.randrange(256)
        return Rule(p,rep)
    if choice<0.85:
        kind=rng.choice([1,2,3,4,5])
        opcode=-(16+kind*16)  # reverse, +1, -1, *2, //2
        return Rule([a,-1,PREDICTION_SLOT],[a,opcode,PREDICTION_SLOT])
    if choice<0.92:
        return Rule([a,-1,PREDICTION_SLOT],[a,-16,b])  # sort($1)
    p=[rng.randrange(256) for _ in range(log_uniform_count(rng,40))]
    if len(p)>=3 and rng.random()<0.4:
        p[rng.randrange(len(p))]=-1
    rep=[rng.randrange(256) for _ in range(len(p))]
    return Rule(p,rep)


def new_genome(rules: int, rng: random.Random, corpus, bigrams, enable_embedding=True) -> Genome:
    lut=list(range(256))
    if enable_embedding:
        # Initial LUT near identity; mutations subsequently explore any permutation.
        for _ in range(rng.randint(0,6)):
            x,y=rng.sample(range(256),2)
            lut[x],lut[y]=lut[y],lut[x]
    return Genome([repair_evolved_rule(make_rule(rng,corpus,bigrams),rng) for _ in range(rules)], lut)


def log_uniform_count(rng: random.Random, maximum: int) -> int:
    return max(1,min(maximum,int(round(math.exp(rng.uniform(0,math.log(max(1,maximum))))))))


def _mutation_radius(rng: random.Random, n: int, *, local: int = 6,
                     balanced: int = 32) -> int:
    """Heavy-tailed search, but favor small steps that preserve useful programs.

    The older implementation applied a log-uniform draw over all 450 rows for
    *every* child: its typical disruption was unnecessarily large. This uses
    three regimes with rare full-length explorations instead.
    """
    if n <= 1:
        return max(0, n)
    u = rng.random()
    maximum = min(n, local if u < 0.72 else balanced if u < 0.95 else n)
    return log_uniform_count(rng, maximum)


def mutate_lut(lut: list[int], rng: random.Random, enabled: bool) -> list[int]:
    """Permutation-preserving, mostly-local mutation (swap/cycle/inversion)."""
    if not enabled or rng.random() >= 0.25:
        return list(lut)
    out = list(lut)
    k = _mutation_radius(rng, 256, local=4, balanced=24)
    op = rng.random()
    if op < 0.70:
        for _ in range(k):
            x, y = rng.sample(range(256), 2)
            out[x], out[y] = out[y], out[x]
    elif op < 0.90:
        indices = rng.sample(range(256), min(256, max(2, k)))
        values = [out[i] for i in indices]
        out[indices[0]] = values[-1]
        for i, value in zip(indices[1:], values[:-1]):
            out[i] = value
    else:
        left = rng.randrange(255)
        right = min(256, left + max(2, k))
        out[left:right] = reversed(out[left:right])
    return out


def crossover_lut(a: Sequence[int], b: Sequence[int], rng: random.Random) -> list[int]:
    """Permutation-preserving partial donor transplant via value swaps."""
    out = list(a)
    inverse = {v: i for i, v in enumerate(out)}
    for i in rng.sample(range(256), _mutation_radius(rng, 256, local=8, balanced=48)):
        want = b[i]
        if out[i] == want:
            continue
        j = inverse[want]
        inverse[out[i]] = j
        inverse[want] = i
        out[i], out[j] = out[j], out[i]
    return out


def mutate_rule(rule: Rule, rng: random.Random, corpus, bigrams) -> Rule:
    """Mostly 1-4 token edits; occasionally rebuild/reshape whole small rules."""
    if rng.random() < 0.07:
        return repair_evolved_rule(make_rule(rng, corpus, bigrams), rng)
    p = rule.pattern.copy()
    q = rule.replacement.copy()
    edits = _mutation_radius(rng, max(1, len(p) + len(q)), local=4, balanced=12)
    for _ in range(edits):
        action = rng.randrange(16)
        s = p if rng.random() < 0.52 else q
        if action <= 5 and s:
            pos = rng.randrange(len(s))
            if s is p:
                # Preserve the common last-byte prediction-slot guard in many
                # local variants, without forcing this convention on all rules.
                if pos == len(p) - 1 and p[pos] == PREDICTION_SLOT and rng.random() < 0.65:
                    continue
                s[pos] = (-1 if rng.random() < 0.12 else
                          rng.choice(bigrams)[rng.randrange(2)])
            else:
                wc = p.count(-1)
                if wc and rng.random() < 0.46:
                    family = rng.randrange(7)
                    cap = rng.randint(1, min(15 if family == 0 else 16, wc))
                    s[pos] = -(cap if family == 0 else 16 + (family - 1)*16 + cap - 1)
                else:
                    s[pos] = rng.choice(bigrams)[rng.randrange(2)] if rng.random() < 0.65 else rng.randrange(256)
        elif action == 6 and len(s) > 1:
            i, j = rng.sample(range(len(s)), 2)
            s[i], s[j] = s[j], s[i]
        elif action in (7, 8):
            pos = rng.randrange(len(s) + 1)
            if s is p and rng.random() < 0.15:
                s.insert(pos, -1)
            else:
                s.insert(pos, rng.choice(bigrams)[rng.randrange(2)])
        elif action == 9 and len(s) > 1:
            s.pop(rng.randrange(len(s)))
        elif action == 10 and len(s) > 1:
            i, j = sorted(rng.sample(range(len(s)), 2))
            s[i:j+1] = reversed(s[i:j+1])
        elif action == 11 and len(s) > 2:
            i, j = sorted(rng.sample(range(len(s)), 2))
            block = s[i:j+1]
            del s[i:j+1]
            to = rng.randrange(len(s) + 1)
            s[to:to] = block
        elif action == 12 and len(p) > 1 and rng.random() < 0.35:
            p[:] = make_rule(rng, corpus, bigrams).pattern
        elif action == 13 and rng.random() < 0.35:
            q[:] = make_rule(rng, corpus, bigrams).replacement
        elif action == 14 and s is p:
            # Extend a locally matched literal context with a corpus bigram.
            a, b = rng.choice(bigrams)
            i = rng.randrange(len(p)+1)
            p[i:i] = [a, b]
        elif action == 15 and s is q and q:
            # Promote/demote a capture operation without changing the pattern.
            wc = p.count(-1)
            if wc:
                i = rng.randrange(len(q))
                family = rng.randrange(7)
                ci = rng.randint(1, min(15 if family == 0 else 16, wc))
                q[i] = -(ci if family == 0 else 16+(family-1)*16+ci-1)
    # Rare geometric expansion / duplication of a meaningful rule fragment.
    # Step size is finite; there is no upper bound across generations.
    if rng.random() < 0.035:
        s = p if rng.random() < 0.58 else q
        if s:
            start = rng.randrange(len(s))
            width = min(len(s)-start, log_uniform_count(rng, max(1, len(s))))
            block = s[start:start+width]
            at = rng.randrange(len(s)+1)
            s[at:at] = block
    out = repair_evolved_rule(Rule(p, q), rng)
    return out


def _rule_signature(r: Rule) -> tuple:
    return (tuple(r.pattern), tuple(r.replacement))


def _diff3_sequence(base: Sequence, donor: Sequence, target: Sequence,
                    rng: random.Random, *, signatures=None,
                    prefer_target: float = 0.94) -> list:
    """3-way sequence merge, not position-wise 'mutation by another parent'.

    Extract insertion/deletion/replacement hunks A->X and A->Y with Myers-like
    alignment (SequenceMatcher), apply non-overlapping changes from both sides,
    and explicitly resolve overlapping hunks. No edits are discarded merely
    because lengths change; this is important inside pattern/replacement lists.
    """
    from difflib import SequenceMatcher
    base = list(base)
    donor = list(donor)
    target = list(target)
    if signatures is None:
        sb, sd, st = base, donor, target
    else:
        sb = [signatures(v) for v in base]
        sd = [signatures(v) for v in donor]
        st = [signatures(v) for v in target]
    if sd == sb:
        return target
    if st == sb:
        return donor
    if sd == st:
        return target

    def edits(src_keys, dst_keys, dst):
        return [(i1,i2,list(dst[j1:j2]))
                for tag,i1,i2,j1,j2 in SequenceMatcher(
                    None,src_keys,dst_keys,autojunk=False).get_opcodes()
                if tag != 'equal']
    dx = edits(sb, sd, donor)
    dy = edits(sb, st, target)
    ix = iy = pos = 0
    merged = []

    def render(lo, hi, hunks):
        out = []
        cursor = lo
        for start, end, rep in hunks:
            out.extend(base[cursor:start])
            out.extend(rep)
            cursor = end
        out.extend(base[cursor:hi])
        return out

    while ix < len(dx) or iy < len(dy):
        start = min(dx[ix][0] if ix < len(dx) else len(base)+1,
                    dy[iy][0] if iy < len(dy) else len(base)+1)
        merged.extend(base[pos:start])
        end = start
        hx, hy = [], []
        # Closed boundaries deliberately coalesce insertions at the same point.
        # The loop always consumes at least one hunk, including zero-length ones.
        while True:
            advanced = False
            while ix < len(dx) and dx[ix][0] <= end:
                h = dx[ix]; hx.append(h); ix += 1
                end = max(end, h[1]); advanced = True
            while iy < len(dy) and dy[iy][0] <= end:
                h = dy[iy]; hy.append(h); iy += 1
                end = max(end, h[1]); advanced = True
            if not advanced:
                break
        old = base[start:end]
        candidate_x = render(start, end, hx)
        candidate_y = render(start, end, hy)
        if not hx or candidate_x == old:
            merged.extend(candidate_y)
        elif not hy or candidate_y == old or candidate_x == candidate_y:
            merged.extend(candidate_x)
        else:
            # On a genuine conflict prefer the target being improved. Very rare
            # donor/base choices let evolution escape over-conservative merges.
            roll = rng.random()
            if roll < prefer_target:
                merged.extend(candidate_y)
            elif roll < prefer_target + (1.0-prefer_target)*0.8:
                merged.extend(candidate_x)
            else:
                merged.extend(old)
        pos = end
    merged.extend(base[pos:])
    return merged


def _diff3_lut(base: Sequence[int], donor: Sequence[int], target: Sequence[int],
               rng: random.Random) -> list[int]:
    """Transfer a bounded number of permutation edits using reciprocal swaps."""
    out = list(target)
    inverse = {v:i for i,v in enumerate(out)}
    loci = [i for i,(x,y) in enumerate(zip(base,donor)) if x != y]
    if not loci:
        return out
    for i in rng.sample(loci, min(len(loci), _mutation_radius(rng, len(loci), local=4, balanced=16))):
        if target[i] != base[i] and rng.random() < 0.95:
            continue
        wanted = donor[i]
        j = inverse[wanted]
        if j == i:
            continue
        old = out[i]
        out[i], out[j] = out[j], out[i]
        inverse[wanted] = i
        inverse[old] = j
    return out


def _move_rule_block(rows: list[Rule], rng: random.Random) -> None:
    """Reorder a short coadapted block without changing its internal order."""
    if len(rows) < 3:
        return
    span = min(len(rows) - 1, _mutation_radius(rng, len(rows), local=4, balanced=16))
    start = rng.randrange(len(rows) - span + 1)
    block = rows[start:start+span]
    del rows[start:start+span]
    destination = rng.randrange(len(rows) + 1)
    rows[destination:destination] = block


def mutate_genome(parent: Genome, rng: random.Random, corpus, bigrams,
                  enabled=True) -> Genome:
    rows = parent.rules.copy()
    n = len(rows)
    if n:
        k = _mutation_radius(rng, n, local=6, balanced=36)
        if k > 1 and rng.random() < 0.23:
            # Cluster neighboring changes (coadapted rules) in a small segment.
            start = rng.randrange(n-k+1)
            indices = range(start,start+k)
        else:
            indices = rng.sample(range(n),k)
        for ri in indices:
            rows[ri] = mutate_rule(rows[ri],rng,corpus,bigrams)
        if n > 1 and rng.random() < 0.16:
            if rng.random() < 0.58:
                i,j = rng.sample(range(n),2)
                rows[i],rows[j] = rows[j],rows[i]
            else:
                _move_rule_block(rows,rng)
    return Genome(rows,mutate_lut(parent.embedding,rng,enabled))


def _relative_rule_distance(a: Genome, b: Genome) -> int:
    """Exact structural Hamming distance; shared Rule objects are a fast path."""
    return sum((x is not y and (x.pattern != y.pattern or x.replacement != y.replacement))
               for x, y in zip(a.rules, b.rules)) + abs(len(a.rules)-len(b.rules))


def _choose_diff3_base(donor: Genome, target: Genome, candidates: Sequence[Genome]) -> Genome:
    """Prefer a related pseudo-ancestor so A->X means a *small* useful patch.

    The GA doesn't retain a genealogy; taking three unrelated genomes at random
    often makes every row one giant 'replace' hunk. Select the nearest available
    candidate to both donor and target instead, without introducing a fitness
    evaluation or changing any model semantics.
    """
    return min(candidates,key=lambda c: (_relative_rule_distance(c, donor) +
                                          _relative_rule_distance(c, target),
                                          _relative_rule_distance(c, donor)))


def breed_three(base: Genome, donor: Genome, target: Genome, rng: random.Random,
                corpus, bigrams, embedding_enabled: bool) -> Genome:
    """Differential diff3 crossover: transplant base->donor edits onto target.

    Rule order matters, so combine a short aligned sequence merge with rule-local
    pattern/replacement diff3. Incompatible edits preferentially keep target.
    Unlike DE arithmetic, this transfers *categorical edits* with no numeric
    interpolation. Subsequent mutation allows exploration around the merged child.
    """
    n = len(target.rules)
    if len(base.rules) != n or len(donor.rules) != n:
        raise ValueError('diff3 parents must have equal rule counts')
    rows = target.rules.copy()
    changed = [i for i in range(n) if _rule_signature(base.rules[i]) != _rule_signature(donor.rules[i])]
    if changed:
        # Merge a short RULE SEQUENCE as an edit script, allowing shifts to be
        # aligned rather than treating each position as an unrelated gene.
        span = min(n, _mutation_radius(rng,n,local=8,balanced=24))
        center = rng.choice(changed)
        start = min(max(0,center-rng.randrange(span)), n-span)
        stop = start+span
        merged = _diff3_sequence(base.rules[start:stop],donor.rules[start:stop],
                                 rows[start:stop],rng,signatures=_rule_signature)
        if len(merged) == span:
            rows[start:stop] = merged
        # Differentiate within individual rules as well; here edit insertion and
        # deletion in a pattern *are allowed*, then repair enforces exact legality.
        picks = rng.sample(changed, min(len(changed), _mutation_radius(
            rng,len(changed),local=5,balanced=16)))
        for i in picks:
            b,d,t = base.rules[i], donor.rules[i], rows[i]
            p = _diff3_sequence(b.pattern,d.pattern,t.pattern,rng)
            q = _diff3_sequence(b.replacement,d.replacement,t.replacement,rng)
            merged_rule = repair_evolved_rule(Rule(p,q),rng)
            if _rule_signature(merged_rule) != _rule_signature(t):
                rows[i] = merged_rule
        if all(x is y for x,y in zip(rows,target.rules)):
            # When unrelated parents offer no compatible patch, make a bounded
            # donor-row transplant; normal mutation will still run afterward.
            i = rng.choice(changed)
            rows[i] = donor.rules[i]
    lut = ( _diff3_lut(base.embedding,donor.embedding,target.embedding,rng)
           if embedding_enabled and rng.random() < 0.30 else list(target.embedding) )
    child = Genome(rows,lut)
    return mutate_genome(child,rng,corpus,bigrams,embedding_enabled)


def breed(a: Genome,b: Genome,rng: random.Random,corpus,bigrams,
          embedding_enabled: bool) -> Genome:
    """Two-parent crossover with block or sparse, then smaller mutation."""
    n = len(a.rules)
    if n <= 1:
        rows = a.rules.copy()
    elif rng.random() < 0.60:
        # Preserve the order of interacting groups instead of mixing all loci.
        span = _mutation_radius(rng,n,local=8,balanced=64)
        start = rng.randrange(n-span+1)
        rows = a.rules.copy()
        rows[start:start+span] = b.rules[start:start+span]
    else:
        rows = a.rules.copy()
        for i in rng.sample(range(n),_mutation_radius(rng,n,local=6,balanced=40)):
            rows[i] = b.rules[i]
    lut = crossover_lut(a.embedding,b.embedding,rng) if embedding_enabled and rng.random()<0.25 else a.embedding.copy()
    return mutate_genome(Genome(rows,lut),rng,corpus,bigrams,embedding_enabled)


def guided_prediction_children(parent: Genome, examples: Sequence[bytes],
                               rng: random.Random, offspring: int = 4,
                               probe_length: int = 48) -> list[Genome]:
    """Suggest local suffix corrections from *actual* teacher-forced errors.

    This is a low-frequency proposal operator, not gradient learning. New rules
    are evaluated on the usual full texts with the exact same accuracy metric.
    A small prefix keeps Python tracing from dominating native CPU evaluation.
    """
    if not examples or not parent.rules or offspring <= 0:
        return []
    text = rng.choice(examples)[:max(2, probe_length)]
    if len(text) < 2:
        return []
    _, _, preds, states = autoregressive_rollout(parent, text,
                                                 predictions=True, return_states=True)
    errors = [(state, expected) for state, pred, expected
              in zip(states, preds, text[1:]) if pred != expected and len(state) >= 2]
    if not errors:
        return []
    proposals: list[Genome] = []
    for _ in range(offspring):
        state, expected = rng.choice(errors)
        width = rng.randint(2, min(8, len(state)))
        pattern = list(state[-width:])
        replacement = pattern[:-1] + [int(expected)]
        rule = Rule(pattern, replacement)
        if not is_valid_rule(rule):
            continue
        # Place the corrective rule last, after all context/state transforms.
        # The effect can be generalized or rejected by normal GA selection.
        child_rows = parent.rules.copy()
        child_rows[-1] = rule
        proposals.append(Genome(child_rows, parent.embedding.copy()))
    return proposals


def fingerprint(g: Genome) -> int:
    return hash((tuple(g.embedding),tuple((tuple(r.pattern),tuple(r.replacement)) for r in g.rules)))


def _same_structure(a: Genome, b: Genome) -> bool:
    """Defend the evaluation cache against rare fingerprint hash collisions."""
    return (a is b or (a.embedding == b.embedding and len(a.rules)==len(b.rules)
            and all(x is y or (x.pattern==y.pattern and x.replacement==y.replacement)
                    for x,y in zip(a.rules,b.rules))))


def score_population(population: Sequence[Genome], examples: Sequence[bytes], backend: str,
                     mps_batch: int = 16, cpu_workers: int = 0,
                     use_cache: bool = True, progress=None) -> None:
    denominators=np.asarray([max(0,len(x)-1) for x in examples],dtype=np.int64)
    total=int(denominators.sum())
    if total<=0:
        raise ValueError('Every case must have >=2 tokens')
    global _MPS_CROSSCHECK_DONE, _NATIVE_CROSSCHECK_DONE, _NATIVE_UNAVAILABLE_WARNED
    global _SCORE_CACHE_KEY, _SCORE_CACHE_ROWS
    original_population = population
    # Reuse only when the same exact ordered set of byte examples is evaluated.
    # Checkpoints never carry this cache, and rotated examples invalidate it.
    if backend == 'mps' and any(not is_nonexpanding(r)
                                for g in population for r in g.rules):
        # The Metal kernel is fixed-stride and nonexpanding by construction.
        # Never pass an expanding or long rule into it: use identical native CPU
        # semantics rather than dropping, clipping or corrupting output.
        backend = 'cpu'
        global _EXPANSION_FALLBACK_WARNED
        if not _EXPANSION_FALLBACK_WARNED:
            print('Lens: MPS kernel cannot represent expanding/long rules; '
                  'using native CPU evaluator for these generations.', flush=True)
            _EXPANSION_FALLBACK_WARNED = True
    cache_enabled = use_cache and backend in ('mps','cpu')
    cache_key = (backend, tuple(bytes(x) for x in examples))
    prior = _SCORE_CACHE_ROWS if cache_enabled and _SCORE_CACHE_KEY==cache_key else {}
    groups = {}
    missing = []
    missing_map = []
    missing_representatives=[]
    cached = [None] * len(population)
    identifiers = []
    if cache_enabled:
        for i,g in enumerate(population):
            key=fingerprint(g)
            identifiers.append(key)
            hit = None
            for prev, values in prior.get(key, ()):
                if _same_structure(g,prev):
                    hit=values
                    break
            if hit is None:
                for j in groups.get(key, ()):
                    if _same_structure(g,population[j]):
                        missing_map.append((i,j))
                        hit = 'duplicate'
                        break
            if hit is None:
                missing.append(g)
                missing_representatives.append(i)
                groups.setdefault(key,[]).append(i)
            elif isinstance(hit,np.ndarray):
                cached[i]=hit
        evaluated_population=missing
    else:
        evaluated_population=population
    # tqdm accounts for all requested genomes, including exact cached hits and
    # within-generation duplicates. Only truly evaluated genomes reach the backend.
    if progress is not None:
        cache_hits = len(original_population) - len(evaluated_population)
        if cache_hits:
            progress.update(cache_hits)
            progress.set_postfix_str(f'cache={cache_hits}', refresh=False)
    population=evaluated_population
    if len(population)==0:
        counts=np.empty((0,len(examples)),dtype=np.int32)
    elif backend=='mps':
        if gpu_backend is None:
            raise RuntimeError('gpu_replace_persistent.py missing')
        counts=gpu_backend.evaluate_mps(population,examples,len(population[0].rules),
                                        mps_batch,
                                        on_batch_done=progress.update if progress is not None else None)
        if not _MPS_CROSSCHECK_DONE:
            if progress is not None:
                progress.set_postfix_str('初回MPS/CPU照合中')
            # Check full GPU examples against the native CPU evaluator when available.
            # The old 2x2 Python full-length self-check can take minutes for 1499-byte
            # texts and was unrelated to the Metal kernel's actual performance.
            probes=population[:min(2,len(population))]
            probe_texts=examples[:min(2,len(examples))]
            checked=False
            if native_cpu is not None:
                try:
                    expected=native_cpu.evaluate_cpu(probes,probe_texts,
                                                    len(probes[0].rules),cpu_workers)
                    np.testing.assert_array_equal(counts[:len(probes),:len(probe_texts)],expected,
                        err_msg='MPS/native-CPU accuracy mismatch: use --backend cpu')
                    checked=True
                except (RuntimeError,OSError):
                    pass
            if not checked:
                # Compiler unavailable: compare both evaluators on an identical
                # short prefix, rather than doing a quadratic Python long rollout.
                short=[bytes(row[:min(len(row),64)]) for row in probe_texts]
                gpu_short=gpu_backend.evaluate_mps(probes,short,len(probes[0].rules),mps_batch)
                for gi,g in enumerate(probes):
                    for i,txt in enumerate(short):
                        expected=autoregressive_rollout(g,txt)[0]
                        if int(gpu_short[gi,i])!=expected:
                            raise RuntimeError(f'MPS/Python short-probe mismatch: '
                                               f'MPS={gpu_short[gi,i]} Python={expected}')
            _MPS_CROSSCHECK_DONE=True
    else:
        counts=None
        if backend == 'cpu' and native_cpu is not None:
            try:
                counts=native_cpu.evaluate_cpu(population,examples,len(population[0].rules),
                                               cpu_workers,
                                               on_genome_done=progress.update if progress is not None else None)
                if not _NATIVE_CROSSCHECK_DONE:
                    if progress is not None:
                        progress.set_postfix_str('初回CPU/Python照合中')
                    # Keep a one-time independent Python check, but avoid full
                    # length examples here; 1500-byte Python rollouts are slow.
                    probes=population[:min(2,len(population))]
                    short=[bytes(row[:min(len(row),64)]) for row in examples[:2]]
                    cpu_short=native_cpu.evaluate_cpu(probes,short,
                                                      len(probes[0].rules),cpu_workers)
                    for gi,g in enumerate(probes):
                        for i,txt in enumerate(short):
                            expected=autoregressive_rollout(g,txt)[0]
                            if int(cpu_short[gi,i])!=expected:
                                raise AssertionError(f'Native CPU mismatch: '
                                    f'genome={gi} sample={i}, '
                                    f'C++={cpu_short[gi,i]}, Python={expected}')
                    _NATIVE_CROSSCHECK_DONE=True
            except (RuntimeError, OSError) as exc:
                if not _NATIVE_UNAVAILABLE_WARNED:
                    print(f'Native CPU unavailable ({exc}); using Python reference',flush=True)
                    _NATIVE_UNAVAILABLE_WARNED=True
        if counts is None:
            counts=np.zeros((len(population),len(examples)),dtype=np.int32)
            for gi,g in enumerate(population):
                for i,txt in enumerate(examples):
                    counts[gi,i]=autoregressive_rollout(g,txt)[0]
                if progress is not None:
                    progress.update(1)
    if cache_enabled:
        full_counts=np.empty((len(original_population),len(examples)),dtype=np.int32)
        for i,row in enumerate(cached):
            if row is not None:
                full_counts[i]=row
        for j,rep_i in enumerate(missing_representatives):
            full_counts[rep_i]=counts[j]
        for i,rep_i in missing_map:
            full_counts[i]=full_counts[rep_i]
        # Keep only the current live population, releasing dead generations.
        new_cache={}
        for i,g in enumerate(original_population):
            new_cache.setdefault(identifiers[i],[]).append((g,full_counts[i].copy()))
        _SCORE_CACHE_KEY=cache_key
        _SCORE_CACHE_ROWS=new_cache
        counts=full_counts
        population=original_population
    for gi,g in enumerate(population):
        g.fitness=float(int(counts[gi].sum())/total)
        g.case_scores=[float(counts[gi,i]/denominators[i]) for i in range(len(examples))]



# Keep per-case counts across rotations, not just across identical four-case batches.
# The backend evaluator above is left unchanged for compatibility.  Small sets of
# new (genome, case) pairs are dispatched through the same CPU/MPS implementation.
_score_population_backend = score_population
_CASE_SCORE_CACHE: OrderedDict[int, list] = OrderedDict()
_CASE_SCORE_MAX_GENOMES = 1100


def clear_case_score_cache() -> None:
    """Discard in-memory evaluation reuse without changing model/checkpoint state."""
    _CASE_SCORE_CACHE.clear()


def score_population(population: Sequence[Genome], examples: Sequence[bytes], backend: str,
                     mps_batch: int = 16, cpu_workers: int = 0,
                     use_cache: bool = True, progress=None) -> None:
    """Evaluate exact byte accuracy; reuse only verified genome/case pairs.

    Cached values are exact integer correct-byte counts.  Case rotations only
    invalidate the replaced case, not all other evaluation data.  Duplicated
    genomes in the current call are evaluated once.  Progress counts genomes,
    including cached ones, so callers' progress bars stay accurate.
    """
    if not population:
        return
    global _NATIVE_UNAVAILABLE_WARNED
    if backend == 'cpu' and native_cpu is None and not _NATIVE_UNAVAILABLE_WARNED:
        print('Lens warning: native_cpu module not found. CPU mode is using the '
              'much slower Python reference evaluator; install/build the '
              'native_cpu backend for performance.', flush=True)
        _NATIVE_UNAVAILABLE_WARNED = True
    if not use_cache:
        return _score_population_backend(population, examples, backend, mps_batch,
                                         cpu_workers, use_cache=False, progress=progress)
    denominators = [len(t) - 1 for t in examples]
    total = sum(denominators)
    if total <= 0 or any(n <= 0 for n in denominators):
        raise ValueError('All training cases must have at least two bytes')
    examples = [bytes(t) for t in examples]
    fingerprints = [fingerprint(g) for g in population]
    count_matrix = np.full((len(population), len(examples)), -1, dtype=np.int32)
    representative_indices: dict[int, list[int]] = {}
    duplicates: list[tuple[int, int]] = []
    unique_indices: list[int] = []
    prior_entries: dict[int, tuple[Genome, dict[bytes, int]]] = {}
    for i, g in enumerate(population):
        fp = fingerprints[i]
        duplicate = next((j for j in representative_indices.get(fp, ())
                          if _same_structure(g, population[j])), None)
        if duplicate is not None:
            duplicates.append((i, duplicate))
            continue
        representative_indices.setdefault(fp, []).append(i)
        unique_indices.append(i)
        for prev_g, case_map in _CASE_SCORE_CACHE.get(fp, ()):
            if _same_structure(g, prev_g):
                prior_entries[i] = (prev_g, case_map)
                for j, text in enumerate(examples):
                    if text in case_map:
                        count_matrix[i, j] = case_map[text]
                break
    missing_groups: dict[tuple[int, ...], list[int]] = {}
    for i in unique_indices:
        missing = tuple(j for j in range(len(examples)) if count_matrix[i, j] < 0)
        if missing:
            missing_groups.setdefault(missing, []).append(i)
    completed = len(unique_indices) - sum(map(len, missing_groups.values())) + len(duplicates)
    if progress is not None and completed:
        progress.update(completed)
    for missing, indices in missing_groups.items():
        subsamples = [examples[j] for j in missing]
        genomes = [population[i] for i in indices]
        # Core code uses precisely the same backend and the same CPU crosschecks.
        _score_population_backend(genomes, subsamples, backend, mps_batch,
                                  cpu_workers, use_cache=False, progress=None)
        for i, genome in zip(indices, genomes):
            for k, j in enumerate(missing):
                count_matrix[i, j] = round(genome.case_scores[k] * denominators[j])
        if progress is not None:
            progress.update(len(indices))
    for i, representative in duplicates:
        count_matrix[i, :] = count_matrix[representative, :]
    if np.any(count_matrix < 0):
        raise AssertionError('Unresolved genome/case evaluation')
    for i, g in enumerate(population):
        g.fitness = float(int(count_matrix[i].sum()) / total)
        g.case_scores = [float(count_matrix[i, j] / denominators[j])
                         for j in range(len(examples))]
    # Limit references to whole genomes. Keep recent populations so evaluations
    # survive a changing rolling training case without memory leaks.
    for i in unique_indices:
        g = population[i]
        fp = fingerprints[i]
        new_map = dict(prior_entries[i][1]) if i in prior_entries else {}
        for j, text in enumerate(examples):
            new_map[text] = int(count_matrix[i, j])
        # Bound stored cases per genome (most recently used are at the end).
        if len(new_map) > 48:
            new_map = dict(list(new_map.items())[-48:])
        entries = _CASE_SCORE_CACHE.pop(fp, [])
        entries = [(old, counts) for old, counts in entries
                   if not _same_structure(old, g)]
        entries.append((g, new_map))
        _CASE_SCORE_CACHE[fp] = entries
    while sum(map(len, _CASE_SCORE_CACHE.values())) > _CASE_SCORE_MAX_GENOMES:
        _CASE_SCORE_CACHE.popitem(last=False)

def write_history(path: str, history: Sequence[dict]) -> None:
    if not path or not history:
        return
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(key for record in history for key in record))
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def append_history_row(path: str, row: dict) -> None:
    """O(1) append each generation, instead of rewriting an ever-growing CSV."""
    if not path:
        return
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    create_header=not p.exists() or p.stat().st_size == 0
    with p.open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if create_header:
            writer.writeheader()
        writer.writerow(row)


def plot_history(history: Sequence[dict], prefix: str, window: int) -> None:
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x=[row['generation'] for row in history]
    y=np.asarray([row['best_accuracy'] for row in history])
    cum=np.concatenate(([0.],np.cumsum(y,dtype=np.float64)))
    avg=np.asarray([(cum[i+1]-cum[max(0,i-window+1)]) /
                    (i+1-max(0,i-window+1)) for i in range(len(y))])
    fig,ax=plt.subplots(figsize=(11,5.5))
    # Thin only the displayed lines; retain every generation in CSV/history.
    ix=np.unique(np.r_[np.arange(0,len(x),max(1,len(x)//20000)),len(x)-1])
    ax.plot(np.asarray(x)[ix],y[ix],alpha=.35,label='Rolling-case best train accuracy')
    ax.plot(np.asarray(x)[ix],avg[ix],label='Rolling-case moving average')
    champion = np.asarray([float(r.get('best_ever_accuracy', r['best_accuracy']))
                           for r in history])
    ax.plot(np.asarray(x)[ix], champion[ix],
            label='Champion accuracy (current rolling cases)')
    ax.set(xlabel='Generation', ylabel='Teacher-forced next-byte accuracy',
           title='Lens: rolling training accuracy (no extra evaluation)')
    ax.legend(); fig.tight_layout()
    out=Path(prefix+'_accuracy.png')
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=150); plt.close(fig)


def genome_object(g: Genome) -> dict:
    return {'version':CHECKPOINT_VERSION,'mode':'autoregressive-byte',
            'token_count':256,'prediction_placeholder':PREDICTION_SLOT,
            'rules':[{'a':r.pattern,'b':r.replacement} for r in g.rules],
            'embedding_lut':g.embedding,'fitness':g.fitness,'case_scores':g.case_scores}


def genome_from_object(v: dict) -> Genome:
    if v.get('version')!=CHECKPOINT_VERSION or v.get('mode')!='autoregressive-byte':
        raise ValueError('Only v4 autoregressive genomes are loadable; old 512-token checkpoints use incompatible semantics')
    rules=[Rule(list(r['a']),list(r['b'])) for r in v['rules']]
    g=Genome(rules,list(v['embedding_lut']),float(v.get('fitness',-1)),list(v.get('case_scores',[])))
    if not valid_lut(g.embedding) or not all(map(is_valid_rule,g.rules)):
        raise ValueError('invalid genome: LUT/rule invariant')
    return g


def save_model(path: str,g: Genome) -> None:
    if not path:
        return
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(genome_object(g),ensure_ascii=False,separators=(',',':')))
    os.replace(temp,p)


def save_checkpoint(path:str, generation:int,population: Sequence[Genome],best:Genome,
                    history:list[dict],rng:random.Random, archive: Sequence[Genome] = (),
                    rolling_cases: RollingTrainingCases | None = None) -> None:
    if not path:
        return
    payload={'version':CHECKPOINT_VERSION,'generation':generation,
             'population':[genome_object(g) for g in population],
             'best':genome_object(best),'history':history,'rng':rng.getstate(),
             'archive':[genome_object(g) for g in archive],
             'rolling_cases': rolling_cases.snapshot() if rolling_cases is not None else None}
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.tmp')
    with tmp.open('wb') as f:
        pickle.dump(payload,f,protocol=5)
    os.replace(tmp,dest)


def load_checkpoint(path:str, *, include_rotation_state: bool = False):
    with open(path,'rb') as f:
        v=pickle.load(f)
    if v.get('version')!=CHECKPOINT_VERSION:
        raise ValueError('Old checkpoints have incompatible token counts and training goals; start a new v4 run')
    # IMPORTANT: Only open checkpoint files from trusted sources (pickle).
    rng=random.Random()
    rng.setstate(v['rng'])
    result = (v['generation'], [genome_from_object(g) for g in v['population']],
              genome_from_object(v['best']), v['history'], rng,
              [genome_from_object(g) for g in v.get('archive', [])])
    return (*result, v.get('rolling_cases')) if include_rotation_state else result


def evolve(args) -> Genome:
    show_progress = not getattr(args, 'no_tqdm', False)
    if show_progress and tqdm is None:
        raise RuntimeError('tqdm が必要です: python -m pip install tqdm （または --no-tqdm を指定）')
    if args.plateau_generations < 1 or args.guided_every < 0:
        raise ValueError('plateau-generations must be >=1 and guided-every must be >=0')
    if args.guided_offspring < 0 or args.guided_probe_length < 2:
        raise ValueError('guided-offspring must be >=0, guided-probe-length >=2')
    if not 0 <= args.diff3_rate <= 1:
        raise ValueError('--diff3-rate must be in [0,1]')
    rng=random.Random(args.seed)
    corpus=load_corpus(args.local_corpus,args.min_chunk,args.corpus_chunks,args.max_chunk)
    bigrams=_bigram_seeds(corpus)
    if not bigrams:
        raise ValueError('Training corpus needs adjacent token pairs')
    if args.load:
        first,population,best_ever,history,rng,archive,rotation_state=load_checkpoint(
            args.load, include_rotation_state=True)
        args.population=len(population)
        args.rules=len(population[0].rules)
    else:
        rotation_state=None
        first=0
        history=[]
        best_ever=None
        archive=[]
        population=[]
        init_bar=tqdm(total=args.population,desc='初期集団を生成',unit='個体',
                      dynamic_ncols=True,mininterval=0.5,leave=False) if show_progress else None
        try:
            for _ in range(args.population):
                population.append(new_genome(args.rules,rng,corpus,bigrams,not args.no_embedding))
                if init_bar is not None:
                    init_bar.update(1)
        finally:
            if init_bar is not None:
                init_bar.close()
    # Migrate checkpoints made with the former held-out split.  Map the saved
    # rolling-case indices to the full corpus instead of reinitializing the GA.
    # No additional examples are evaluated: every score uses only train cases.
    legacy_best = None
    if rotation_state is not None:
        full_digest = RollingTrainingCases._digest(corpus)
        if rotation_state['corpus_digest'] != full_digest:
            old_holdout = rotation_state.get('validation_indices')
            if old_holdout is None:
                raise ValueError('Checkpoint uses an unknown corpus split; cannot remap rolling case indices')
            excluded = set(old_holdout)
            kept = [i for i in range(len(corpus)) if i not in excluded]
            old_corpus = [corpus[i] for i in kept]
            if RollingTrainingCases._digest(old_corpus) != rotation_state['corpus_digest']:
                raise ValueError('Checkpoint corpus mismatch: cannot safely migrate case indices')
            rotation_state = dict(rotation_state)
            rotation_state['indices'] = [kept[i] for i in rotation_state['indices']]
            rotation_state['corpus_digest'] = full_digest
    rolling = RollingTrainingCases(corpus, args.cases, args.seed * 1000003,
                                   args.case_rotate_every, rotation_state)
    if rotation_state is None and first:
        rolling.for_generation(first - 1)
    # Saved fitness from the older validation-based version is incomparable to
    # training ACC; keep its genome as a candidate and rescore it on train only.
    if best_ever is not None and any('validation_accuracy' in r for r in history):
        legacy_best = best_ever
        best_ever = None
        archive.append(copy.deepcopy(legacy_best))

    history_keys = ('generation', 'best_accuracy', 'mean_accuracy', 'median_accuracy',
                    'best_ever_accuracy', 'correct_bytes', 'evaluated_bytes',
                    'eval_seconds', 'rules', 'tokens', 'backend', 'case_rotation',
                    'rotated_case', 'train_improved', 'last_train_improve',
                    'plateau_active', 'unique_fitness', 'breed_seconds',
                    'compute_seconds')
    last_train_improve = first
    if history and best_ever is not None:
        last_train_improve = int(history[-1].get('last_train_improve', first))
    # Drop all obsolete validation columns, including for resumed CSV histories;
    # rows have the exact same schema as newly appended rows.
    history = [{key: (old_row.get('best_accuracy', '')
                      if key == 'best_ever_accuracy' and 'validation_accuracy' in old_row
                      else old_row.get(key, ''))
                for key in history_keys} for old_row in history]
    if args.history_csv and history:
        write_history(args.history_csv, history)
    generation_bar=(tqdm(total=max(0,args.generations-first),
                         desc='Lens 学習',unit='世代',dynamic_ncols=True,
                         mininterval=0.5,position=0,leave=True)
                    if show_progress else None)
    for generation in range(first,args.generations):
        t0=time.perf_counter()
        # Replace exactly one case on each rotation boundary; keep all others.
        train=rolling.for_generation(generation)
        if generation_bar is not None:
            generation_bar.set_description_str(f'Lens 学習 gen={generation}')
        # Historical fitness was measured on different rolling cases; do NOT
        # allow those stale scores to guide parent selection. Score the sampled
        # HOF individuals on the *same* current examples as the population.
        hof_pool = (rng.sample(archive, min(len(archive), max(8, args.hof_inject)))
                    if args.hof_inject and archive else [])
        if best_ever is not None and not any(_same_structure(best_ever, g) for g in hof_pool):
            hof_pool.append(best_ever)
        if legacy_best is not None and generation == first and not any(
                _same_structure(legacy_best, g) for g in hof_pool):
            hof_pool.append(legacy_best)
        evaluated = population + hof_pool
        eval_bar=(tqdm(total=len(evaluated),desc=f'gen={generation} 評価/{args.backend}',
                       unit='個体',position=1,leave=False,dynamic_ncols=True,
                       mininterval=0.5) if show_progress else None)
        try:
            score_population(evaluated,train,args.backend,args.mps_batch,args.cpu_workers,
                             use_cache=not args.no_eval_cache,progress=eval_bar)
        finally:
            if eval_bar is not None:
                eval_bar.close()
        current_hof = sorted(hof_pool, key=lambda g:g.fitness, reverse=True)
        population.sort(key=lambda g:g.fitness,reverse=True)
        current=population[0]
        # The incumbent is already included in this generation's HOF evaluation.
        # Compare candidates only on the SAME rotating training cases; no
        # validation passes or additional full-length rollouts are performed.
        candidates = [current] + [g for g in current_hof if g is not best_ever]
        challenger = max(candidates, key=lambda g: g.fitness)
        train_improved = (best_ever is None or
                          challenger.fitness > best_ever.fitness + 1e-12)
        if train_improved:
            best_ever = copy.deepcopy(challenger)
            last_train_improve = generation
            save_model(args.save, best_ever)
        if args.hof_size > 0:
            # The HOF is a *diversity reservoir*, not a ranking of incomparable
            # fitness values from different generations. Keep newest unique states.
            archive.append(copy.deepcopy(current))
            seen = set()
            unique_reverse = []
            for g in reversed(archive):
                sig = fingerprint(g)
                if sig not in seen:
                    unique_reverse.append(g)
                    seen.add(sig)
                if len(unique_reverse) >= args.hof_size:
                    break
            archive = list(reversed(unique_reverse))
        plateau_active = generation - last_train_improve >= args.plateau_generations
        mean_fit=statistics.fmean(g.fitness for g in population)
        median_fit=statistics.median(g.fitness for g in population)
        elapsed=time.perf_counter()-t0
        row={'generation':generation,'best_accuracy':current.fitness,
             'mean_accuracy':mean_fit,'median_accuracy':median_fit,
             'best_ever_accuracy':best_ever.fitness,
             'correct_bytes':round(current.fitness*sum(len(s)-1 for s in train)),
             'evaluated_bytes':sum(len(s)-1 for s in train),'eval_seconds':elapsed,
             'rules':args.rules,'tokens':256,'backend':args.backend,
             'case_rotation':rolling.rotations_done,'rotated_case':(
                 (rolling.rotations_done-1)%args.cases if generation and
                 generation%args.case_rotate_every == 0 else -1),
             'train_improved':int(train_improved),
             'last_train_improve':last_train_improve,
             'plateau_active':int(plateau_active),
             'unique_fitness':len({g.fitness for g in population})}
        history.append(row)
        longest_pattern=max(map(lambda rule: len(rule.pattern), current.rules), default=0)
        longest_replacement=max(map(lambda rule: len(rule.replacement), current.rules), default=0)
        # Evaluation-only time is recorded above. Breeding/merging and chart
        # serialization were previously hidden from the printed duration.
        breed_start = time.perf_counter()
        next_pop=[copy.deepcopy(g) for g in population[:min(args.elites,args.population)]]
        if current_hof and args.hof_inject:
            present = {fingerprint(x) for x in next_pop}
            for g in current_hof[:args.hof_inject]:
                sig = fingerprint(g)
                if len(next_pop)<args.population and sig not in present:
                    next_pop.append(copy.deepcopy(g))
                    present.add(sig)
        if best_ever is not None and len(next_pop) < args.population and not any(
                _same_structure(best_ever, g) for g in next_pop):
            next_pop.append(copy.deepcopy(best_ever))
        if args.guided_every > 0 and generation % args.guided_every == 0:
            for child in guided_prediction_children(current, train, rng,
                                                    args.guided_offspring,
                                                    args.guided_probe_length):
                if len(next_pop) < args.population:
                    next_pop.append(child)
        breed_bar=(tqdm(total=args.population,initial=len(next_pop),
                        desc=f'gen={generation} 次世代作成',unit='個体',position=1,
                        dynamic_ncols=True,mininterval=0.5,leave=False)
                   if show_progress else None)
        def pick():
            # Tournament selection on teacher-forced byte accuracy.
            if current_hof and rng.random() < args.hof_parent_rate:
                return rng.choice(current_hof[:min(16, len(current_hof))])
            tournament_size = min((max(2, args.tournament // 2) if plateau_active
                                   else args.tournament), len(population))
            return max(rng.choices(population,k=tournament_size),key=lambda g:g.fitness)
        try:
            while len(next_pop)<args.population:
                if rng.random() < (min(0.12, 2 * args.immigrant_rate)
                                   if plateau_active else args.immigrant_rate):
                    # Restart near a promising local optimum rather than spending
                    # almost all evaluations on unfit 1500-row random immigrants.
                    if plateau_active and rng.random() < 0.85:
                        child = mutate_genome(pick(),rng,corpus,bigrams,not args.no_embedding)
                        for _ in range(2):
                            child = mutate_genome(child,rng,corpus,bigrams,not args.no_embedding)
                    else:
                        child=new_genome(args.rules,rng,corpus,bigrams,not args.no_embedding)
                elif rng.random()<args.crossover_rate:
                    if rng.random()<args.diff3_rate:
                        # Pick a structurally related pseudo-ancestor A;
                        # transplant its A->X edit script into target Y.
                        donor, target = pick(), pick()
                        base = _choose_diff3_base(donor,target,[pick() for _ in range(3)])
                        child=breed_three(base,donor,target,rng,corpus,bigrams,
                                          not args.no_embedding)
                    else:
                        child=breed(pick(),pick(),rng,corpus,bigrams,not args.no_embedding)
                else:
                    child=mutate_genome(pick(),rng,corpus,bigrams,not args.no_embedding)
                next_pop.append(child)
                if breed_bar is not None:
                    breed_bar.update(1)
        finally:
            if breed_bar is not None:
                breed_bar.close()
        row['breed_seconds'] = time.perf_counter() - breed_start
        row['compute_seconds'] = time.perf_counter() - t0
        summary=(f"gen={generation:6d} acc={current.fitness:.5f} mean={mean_fit:.5f} "
                 f"champion={best_ever.fitness:.5f} tested={row['evaluated_bytes']} "
                 f"eval={elapsed:.2f}s breed={row['breed_seconds']:.2f}s "
                 f"compute={row['compute_seconds']:.2f}s backend={args.backend} "
                 f"rotate={rolling.rotations_done} slot={row['rotated_case']} "
                 f"ruleLen={longest_pattern}/{longest_replacement} "
                 f"improved={int(train_improved)} plateau={int(plateau_active)}")
        if generation_bar is not None:
            generation_bar.set_postfix_str(f'acc={current.fitness:.5f} compute={row["compute_seconds"]:.1f}s',refresh=False)
            tqdm.write(summary)
        else:
            print(summary,flush=True)
        if args.history_csv:
            append_history_row(args.history_csv,row)
        if not args.no_plot and (generation+1)%max(1,args.plot_every)==0:
            plot_history(history,args.plot_prefix,args.plot_window)
        if args.checkpoint and args.checkpoint_every>0 and (generation+1)%args.checkpoint_every==0:
            save_checkpoint(args.checkpoint,generation+1,next_pop,best_ever,history,rng,archive,rolling)
        population=next_pop
        if generation_bar is not None:
            generation_bar.update(1)
    if generation_bar is not None:
        generation_bar.close()
    if best_ever is None:
        raise ValueError('no generations evaluated')
    if args.checkpoint:
        save_checkpoint(args.checkpoint,args.generations,population,best_ever,history,rng,archive,rolling)
    if not args.no_plot:
        plot_history(history,args.plot_prefix,args.plot_window)
    return best_ever


def self_test() -> None:
    lut=list(range(256))
    inv=inverse_lut(lut)
    assert is_valid_rule(Rule([65,0],[65,66]))
    assert is_valid_rule(Rule([65,0],[65,66,67]))
    assert replace_once([65,0],Rule([65,0],[65,66]),lut,inv)==[65,66]
    assert replace_once([65,0],Rule([65,0],[65,66]),lut,inv)==[65,66]
    assert replace_once([1,10,3],Rule([1,-1,3],[-16]),lut,inv)==[10]
    assert replace_once([1,8,7,3],Rule([1,-1,3],[-32]),lut,inv)==[7,8]
    # LUT affects sort's order but not ordinary literal or captured bytes.
    rotated=list(range(256));rotated[1],rotated[2]=2,1
    ri=inverse_lut(rotated)
    assert replace_once([8,1,2,9],Rule([8,-1,9],[-16]),rotated,ri)==[2,1]
    assert replace_once([8,1,9],Rule([8,-1,9],[-48]),rotated,ri)==[3]  # forward LUT(1)=2; +1=3; inverse LUT(3)=3
    g=Genome([Rule([65,0],[65,66])],lut)
    c,tot,preds,traces=autoregressive_rollout(g,b'ABAC',predictions=True,return_states=True)
    assert tot==3 and preds[0]==66 and traces[0]==[65,66]
    assert traces[1][0]==65 # persistent state wasn't rebuilt from original text
    # Corrected teacher token is retained and new prediction slot appended.
    # First two predictions are B after A and slot literal 0, the third may differ.
    g2=Genome([Rule([65,0],[65,66]),Rule([66,0],[66,67])],lut)
    assert autoregressive_rollout(g2,b'ABC')[0]>=1
    r=random.Random(1)
    for _ in range(100):
        p=repair_rule(Rule([r.randrange(256),-1,r.randrange(256)],
                           [-1,r.randrange(256),-48]),r)
        assert is_valid_rule(p)
    assert valid_lut(mutate_lut(lut,r,True))
    print('PASS: autoregressive teacher-forcing, expanding state, unbounded rule length, 256 LUT')


def parse_args():
    ap=argparse.ArgumentParser(description='Lens: stateful byte-level autoregressive Replacer GA')
    # CPU was substantially faster for the real Lens workload, and supports
    # every legal expanding / variable-length program.
    auto_backend = 'cpu'
    ap.add_argument('--backend',choices=['cpu','mps','python'],default="cpu",
                    help='cpu uses a locally compiled C++ accelerator if available; python is the reference')
    ap.add_argument('--local-corpus',default='github-code.txt')
    ap.add_argument('--corpus-chunks',type=int,default=20000,help='maximum ===SPLIT=== separated chunks to load (not a state memory cap)')
    ap.add_argument('--min-chunk',type=int,default=2)
    ap.add_argument('--max-chunk',type=int,default=3500,
                    help='exclusive maximum chunk length in bytes: skip entire chunks of 1500 bytes or more (never crop)')
    ap.add_argument('--cases',type=int,default=4,help='whole-text examples evaluated per generation (4 by default)')
    ap.add_argument('--case-rotate-every',type=int,default=4,
                    help='rotate exactly one case per N generations (round-robin; default 4)')
    ap.add_argument('--population',type=int,default=450)
    ap.add_argument('--rules',type=int,default=1500)
    ap.add_argument('--generations',type=int,default=1000000)
    ap.add_argument('--plateau-generations',type=int,default=80,
                    help='boost diversity after N generations without an improvement over the current training champion')
    ap.add_argument('--guided-every',type=int,default=25,help='create error-informed local rule proposals every N generations; 0 disables')
    ap.add_argument('--guided-offspring',type=int,default=4,help='number of guided local proposals')
    ap.add_argument('--guided-probe-length',type=int,default=48,help='prefix byte count for lightweight Python error tracing')
    ap.add_argument('--elites',type=int,default=12)
    ap.add_argument('--tournament',type=int,default=5)
    ap.add_argument('--hof-size',type=int,default=1024)
    ap.add_argument('--hof-inject',type=int,default=64)
    ap.add_argument('--hof-parent-rate',type=float,default=0.05)
    ap.add_argument('--crossover-rate',type=float,default=0.50)
    ap.add_argument('--diff3-rate',type=float,default=0.35,
                    help='share of crossover offspring using 3-parent categorical diff3 (0..1)')
    ap.add_argument('--immigrant-rate',type=float,default=0.02)
    ap.add_argument('--no-embedding',action='store_true',help='disable evolution of the 256-element operator LUT')
    ap.add_argument('--mps-batch',type=int,default=64,
                    help='independent genome evaluations per Metal batch (default 64, was 16)')
    ap.add_argument('--no-eval-cache',action='store_true',
                    help='disable exact duplicate and cross-generation score reuse')
    ap.add_argument('--no-tqdm',action='store_true',
                    help='disable progress bars (without changing training results)')
    ap.add_argument('--cpu-workers',type=int,default=0,
                    help='native CPU parallel genome evaluators (0=automatic)')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--checkpoint',default='lens_ar_checkpoint.pkl')
    ap.add_argument('--checkpoint-every',type=int,default=10)
    ap.add_argument('--load',default='')
    ap.add_argument('--save',default='best_lens_ar.json')
    ap.add_argument('--history-csv',default='lens_ar_history.csv')
    ap.add_argument('--plot-prefix',default='lens_ar')
    ap.add_argument('--plot-window',type=int,default=300)
    ap.add_argument('--plot-every',type=int,default=1)
    ap.add_argument('--no-plot',action='store_true')
    ap.add_argument('--generate',default='',help='saved autoregressive model; generate without teacher forcing')
    ap.add_argument('--prompt',default='Hello')
    ap.add_argument('--output-bytes',type=int,default=64)
    ap.add_argument('--self-test',action='store_true')
    return ap.parse_args()


def main():
    args=parse_args()
    if args.self_test:
        self_test()
        return
    if args.generate:
        obj=json.loads(Path(args.generate).read_text())
        generated=free_generate(genome_from_object(obj),args.prompt.encode(),args.output_bytes)
        print(generated.decode('utf-8',errors='replace'))
    else:
        evolve(args)

if __name__=='__main__':
    main()
