#!/usr/bin/env python3
"""Lens / Replacer: 256-byte, teacher-forced, greedy n-gram training.

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
*only* within sort/+1/-1/*2//2 and numeric binary operators, then inverted.

The one-pass network permits output longer than the matched input. Rule lengths
and recurrent state lengths have no imposed model-level ceiling; they are only
limited by physical memory and representable allocation sizes. The legacy fixed-
capacity Metal kernel is used only for compatible nonexpanding 64-token rules;
otherwise the native CPU evaluator is selected automatically.
"""
from __future__ import annotations

import argparse
import copy
import csv
from collections import Counter
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
# $1..$4 x $1..$4 => 16 encodings per binary operator.
# All binary opcodes are int16-safe and disjoint from existing -1..-111.
BINARY_OP_BASE = 112
BINARY_OP_NAMES = (
    'INTERSECT', 'DIFF', 'CONV', 'ZIP_ADD', 'ZIP_XOR',
    'EQUAL', 'OVERLAP', 'XCORR', 'FILTER_IN',
)
BINARY_OP_MIN = -(BINARY_OP_BASE + 16 * len(BINARY_OP_NAMES) - 1)  # -255


def binary_opcode(kind: int, left: int, right: int) -> int:
    """Encode one of nine binary operators; capture indices are 1-based (1..4)."""
    if not 0 <= kind < len(BINARY_OP_NAMES) or not 1 <= left <= 4 or not 1 <= right <= 4:
        raise ValueError('binary operator kind or capture index out of range')
    return -(BINARY_OP_BASE + kind * 16 + (left - 1) * 4 + right - 1)


def decode_binary_opcode(op: int):
    if not BINARY_OP_MIN <= op <= -BINARY_OP_BASE:
        return None
    x = -op - BINARY_OP_BASE
    return (x // 16, (x % 16) // 4, x % 4)
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
        elif (binary := decode_binary_opcode(op)) is not None:
            _, left, right = binary
            if left >= wc or right >= wc:
                return False
        elif -111 <= op < 0 and wc:
            ci = ((-op - 16) % 16 + 1) if op <= -16 else -op
            if ci > wc:
                return False
        else:
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
        if decode_binary_opcode(op) is not None:
            return False  # Metal has no binary opcode or variable-length emission.
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
        elif (binary := decode_binary_opcode(op)) is not None:
            _, left, right = binary
            if left < wc and right < wc:
                replacement.append(op)
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
    binary_count = 0
    for op in r.replacement:
        if decode_binary_opcode(op) is not None:
            # Newly evolved rules get at most one binary emission to reduce
            # recursive duplication; manually-authored rules stay unrestricted.
            if binary_count:
                continue
            binary_count += 1
            filtered.append(op)
            continue
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


def _emit_binary(kind: int, a: Sequence[int], b: Sequence[int],
                 lut: Sequence[int], inverse: Sequence[int]) -> list[int]:
    """Deterministic byte-sequence operations shared with the native CPU path.

    INTERSECT and DIFF are stable *set* operations (unique outputs in a-order);
    FILTER_IN preserves the full first input's order and multiplicities.
    Numeric arithmetic is on LUT coordinates modulo 256; boolean equality
    emits the literal 0/1 byte, irrespective of the embedding.
    """
    if kind in (0, 1, 8):
        members = set(b)
        if kind == 8:  # FILTER_IN / ordered multiset filtering
            return [v for v in a if v in members]
        seen = set()
        out = []
        for v in a:
            if (v in members) == (kind == 0) and v not in seen:
                out.append(v)
                seen.add(v)
        return out
    if kind == 5:  # EQUAL: a single unencoded boolean byte
        return [int(a == b)]
    if kind == 6:  # OVERLAP: a + b without a repeated suffix/prefix
        if not b:
            return list(a)
        pi = [0] * len(b)
        j = 0
        for i in range(1, len(b)):
            while j and b[i] != b[j]:
                j = pi[j-1]
            if b[i] == b[j]:
                j += 1
            pi[i] = j
        k = 0
        for v in a:
            while k and (k == len(b) or b[k] != v):
                k = pi[k-1]
            if k < len(b) and b[k] == v:
                k += 1
        return list(a) + list(b[k:])
    if kind in (3, 4):  # zip arithmetic stops at the shorter capture
        if kind == 3:
            return [inverse[(lut[x] + lut[y]) & 255] for x, y in zip(a, b)]
        return [inverse[lut[x] ^ lut[y]] for x, y in zip(a, b)]
    if not a or not b:
        return []
    result = [0] * (len(a) + len(b) - 1)
    if kind == 2:  # classical linear discrete convolution
        for i, x in enumerate(a):
            lx = lut[x]
            for j, y in enumerate(b):
                k = i + j
                result[k] = (result[k] + lx * lut[y]) & 255
    elif kind == 7:  # cross-correlation, lag range -(len(b)-1)..len(a)-1
        base = len(b) - 1
        for i, x in enumerate(a):
            lx = lut[x]
            for j, y in enumerate(b):
                k = i - j + base
                result[k] = (result[k] + lx * lut[y]) & 255
    else:
        raise ValueError(f'unsupported binary operation: {kind}')
    return [inverse[v] for v in result]


def _emit_replacement(rep: Sequence[int], caps: Sequence[Sequence[int]],
                      lut: Sequence[int], inverse: Sequence[int]) -> list[int]:
    out = []
    for op in rep:
        if op >= 0:
            out.append(op)
            continue
        binary = decode_binary_opcode(op)
        if binary is not None:
            kind, left, right = binary
            if left < len(caps) and right < len(caps):
                out.extend(_emit_binary(kind, caps[left], caps[right], lut, inverse))
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
    if choice < 0.025:
        # Seed a small number of genuine two-capture operators; evolution can
        # refine or promote them, instead of waiting for two wildcard insertions.
        kind = rng.choices(range(len(BINARY_OP_NAMES)),
                           weights=[4, 4, 1, 4, 4, 3, 2, 1, 4])[0]
        return Rule([a, -1, b, -1, PREDICTION_SLOT],
                    [a, binary_opcode(kind, 1, 2), PREDICTION_SLOT])
    if choice<0.36:
        return Rule([a, PREDICTION_SLOT],[a,b])
    if choice<0.53:
        return Rule([a,-1,PREDICTION_SLOT],[a,-1,b])
    if choice<0.72:
        x=rng.choice(corpus)
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
                if wc >= 2 and rng.random() < 0.12:
                    kind = rng.choices(range(len(BINARY_OP_NAMES)),
                                       weights=[4,4,1,4,4,3,2,1,4])[0]
                    left, right = rng.choices(range(1, min(4, wc)+1), k=2)
                    s[pos] = binary_opcode(kind, left, right)
                elif wc and rng.random() < 0.46:
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
                if wc >= 2 and rng.random() < 0.35:
                    kind = rng.choices(range(len(BINARY_OP_NAMES)),
                                       weights=[4,4,1,4,4,3,2,1,4])[0]
                    left, right = rng.choices(range(1, min(4, wc)+1), k=2)
                    q[i] = binary_opcode(kind, left, right)
                else:
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
            print('Lens: MPS kernel cannot represent binary/expanding/long rules; '
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
    avg=np.asarray([np.mean(y[max(0,i-window+1):i+1]) for i in range(len(y))])
    fig,ax=plt.subplots(figsize=(11,5.5))
    ax.plot(x,y,alpha=.5,label='Active-window next-byte accuracy')
    ax.plot(x,avg,label='Moving average')
    ax.plot(x,np.maximum.accumulate(y),label='Best observed (different windows)')
    ax.set(xlabel='Greedy trial',ylabel='Teacher-forced next-byte accuracy',title='Lens n-gram greedy training')
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


# ---------------------------------------------------------------------------
# Data-driven one-genome / one-rule greedy optimization.
# There is no breeding, tournament, archive, or evaluated population here.
# ---------------------------------------------------------------------------

def _trace_rule_python(genome: Genome, examples: Sequence[bytes], rule_index: int,
                       samples: int, max_bytes: int):
    """Correct but slow trace fallback if a local C++ compiler is unavailable."""
    lut=genome.embedding
    inverse=inverse_lut(lut)
    anchors=_anchors(genome.rules)
    before=[]; after=[]; scores=[]
    for row in examples:
        state=[row[0],PREDICTION_SLOT]
        hits=0
        steps=min(samples,len(row)-1)
        capture_steps={1 if steps==1 else 1 + j*(len(row)-2)//(steps-1)
                       for j in range(steps)}
        for t,target in enumerate(row[1:],1):
            sampled=t in capture_steps
            present=set(state)
            pairs=set(zip(state,state[1:]))
            for ri,(rule,(kind,anchor)) in enumerate(zip(genome.rules,anchors)):
                if sampled and ri==rule_index:
                    before.append(bytes(state[-max_bytes:]))
                if kind==1 and anchor not in present or kind==2 and anchor not in pairs:
                    continue
                new_state=replace_once(state,rule,lut,inverse)
                if new_state!=state:
                    state=new_state
                    present=set(state)
                    pairs=set(zip(state,state[1:]))
            if sampled and state:
                after.append(bytes(state[-max_bytes:]))
            if not state:
                state=[PREDICTION_SLOT]
            hits+=state[-1]==target
            state[-1]=target
            state.append(PREDICTION_SLOT)
        scores.append(hits)
    return before,after,np.asarray(scores,dtype=np.int32)


def trace_rule(genome: Genome, examples: Sequence[bytes], rule_index: int,
               samples: int = 6, max_bytes: int = 384, backend: str = 'cpu'):
    """Observed states immediately before a rule and after the full sweep.

    Trace returns the baseline's accuracy counts from the *same* execution.
    Recording is only observational; no states are truncated while evaluating.
    """
    if backend != 'python' and native_cpu is not None:
        try:
            return native_cpu.trace_cpu(genome,examples,rule_index,samples,max_bytes)
        except (RuntimeError,OSError) as exc:
            global _NATIVE_UNAVAILABLE_WARNED
            if not _NATIVE_UNAVAILABLE_WARNED:
                print(f'Native trace unavailable ({exc}); using slower Python',flush=True)
                _NATIVE_UNAVAILABLE_WARNED=True
    return _trace_rule_python(genome,examples,rule_index,samples,max_bytes)


def _grammar_lengths(rng: random.Random, *, max_blocks: int = 6,
                     max_literal: int = 12) -> tuple[int,...]:
    """Choose both canonical and open-ended wildcard grammars.

    A grammar is multiple nonempty literal blocks joined by capture wildcards.
    The random branch allows 1..6 blocks of nonfixed length, not just the
    seven original hard-coded templates.
    """
    classic=((2,),(3,),(4,),(5,),(2,2),(3,3),(2,2,2))
    if rng.random()<0.53:
        return rng.choice(classic)
    block_count=rng.randint(1,max_blocks)
    if block_count==1:
        return (log_uniform_count(rng, max_literal*2),)
    return tuple(log_uniform_count(rng,max_literal) for _ in range(block_count))


def _sample_wildcard_gaps(rng: random.Random, block_count: int) -> tuple[int,...]:
    # Gaps are actual sampled byte distances. The encoded -1 wildcards retain
    # their original variable-length greedy matching semantics.
    return tuple(rng.choices((0,1,2,3,4,6,8,12,16,24,32),
                             weights=(2,5,5,5,5,3,2,2,1,1,1))[0]
                 for _ in range(block_count-1))


def frequent_templates(states: Sequence[bytes], lengths: Sequence[int],
                       gaps: Sequence[int], rng: random.Random, top_k: int = 5,
                       budget: int = 100000) -> list[list[int]]:
    """Frequency-ranked motifs from *observed* states, using one grammar.

    Literal segments may be arbitrarily sized up to configured sampling limits;
    internal gaps are encoded as wildcards. A cap controls sampling work rather
    than the length of a rule or of the recurrent state. The template is fixed
    per proposal so frequencies are comparable within that top-5 selection.
    """
    if len(gaps)!=len(lengths)-1 or not states:
        return []
    span=sum(lengths)+sum(gaps)
    if span<=0:
        return []
    rows=[row for row in states if len(row)>=span]
    if not rows:
        return []
    opportunities=[len(row)-span+1 for row in rows]
    count=sum(opportunities)
    freq=Counter()
    def extract(row,offset):
        pattern=[]
        pos=offset
        for i,width in enumerate(lengths):
            if i: pattern.append(-1)
            pattern.extend(row[pos:pos+width])
            pos+=width
            if i<len(gaps): pos+=gaps[i]
        return tuple(pattern)
    if count<=budget:
        for row,n in zip(rows,opportunities):
            for at in range(n):
                freq[extract(row,at)]+=1
    else:
        # Weighted uniform sampling over every possible position in all states.
        # A fixed budget avoids O(total recurrent state length) every trial.
        import bisect
        cumulative=[];total=0
        for n in opportunities:
            total+=n;cumulative.append(total)
        for _ in range(budget):
            k=rng.randrange(total)
            ri=bisect.bisect_right(cumulative,k)
            at=k-(cumulative[ri-1] if ri else 0)
            freq[extract(rows[ri],at)]+=1
    if not freq:
        return []
    # Break ties randomly, avoiding byte-lexicographic bias for rare shapes.
    ordered=list(freq.items())
    rng.shuffle(ordered)
    ordered.sort(key=lambda item:item[1],reverse=True)
    return [list(pattern) for pattern,_ in ordered[:top_k]]


def top5_pattern(states: Sequence[bytes], rng: random.Random, *,
                 top_k: int = 5, max_blocks: int = 6,
                 max_literal: int = 12, budget: int = 100000) -> list[int]:
    for _ in range(12):
        lengths=_grammar_lengths(rng,max_blocks=max_blocks,max_literal=max_literal)
        gaps=_sample_wildcard_gaps(rng,len(lengths))
        options=frequent_templates(states,lengths,gaps,rng,top_k,budget)
        if options:
            return rng.choice(options)
    # When all states are short, fall back to a frequent observed byte.
    singles=frequent_templates(states,(1,),(),rng,top_k,budget)
    return rng.choice(singles) if singles else [PREDICTION_SLOT]


def top5_output(states: Sequence[bytes], rng: random.Random, max_literals: int,
                *,top_k: int=5,budget: int=12000) -> list[int]:
    # Replacement literals are actual final-state n-grams; -1 is NOT a wildcard
    # in a replacement, but a capture reference. Longer outputs are supported
    # when the input pattern has enough literal context to guard state growth.
    upper=max(1,min(32,max_literals))
    if upper==1:
        width=1
    else:
        width=(rng.randint(2,min(upper,5)) if rng.random()<0.65
               else rng.randint(1,upper))
    options=frequent_templates(states,(width,),(),rng,top_k,budget)
    if not options:
        options=frequent_templates(states,(1,),(),rng,top_k,budget)
    return rng.choice(options) if options else [PREDICTION_SLOT]


def propose_ngram_rule(rule: Rule, before: Sequence[bytes], after: Sequence[bytes],
                       rng: random.Random, *, top_k: int=5,
                       max_blocks: int=6, max_literal: int=12,
                       budget: int=12000) -> tuple[Rule,str]:
    """Mutate exactly one rule; use observed pre-rule and final-sweep states."""
    p=rule.pattern.copy(); q=rule.replacement.copy()
    mode=rng.choices(('input','output','both','extend'),weights=(30,30,25,15))[0]
    if mode=='extend' and p.count(-1)>=15:
        mode='input'
    if mode in ('input','both'):
        p=top5_pattern(before,rng,top_k=top_k,max_blocks=max_blocks,
                       max_literal=max_literal,budget=budget)
    elif mode=='extend':
        # Make a new context-aware rule out of the old one without replacing it.
        # Both prepend and append are allowed, including multiple iterations.
        fragment=top5_output(before,rng,2,top_k=top_k,budget=budget)
        p=([-1]+fragment+p if rng.random()<0.5 else p+[-1]+fragment)
    if mode in ('output','both'):
        lit_in=sum(v>=0 for v in p)
        guard=1 if len(p)>=3 and p[-1]==PREDICTION_SLOT else 0
        q=top5_output(after,rng,lit_in+guard,top_k=top_k,budget=budget)
        wc=p.count(-1)
        # Sometimes preserve the computational meaning of a captured segment;
        # the selected literal n-gram still comes from the observed final state.
        if wc and rng.random()<0.20:
            capture=-rng.randint(1,min(15,wc))
            if rng.random()<0.5: q=[capture]+q
            else: q=q+[capture]
    result=repair_evolved_rule(Rule(p,q),rng)
    return result,mode


def _accuracy_from_counts(g: Genome, counts: Sequence[int], texts: Sequence[bytes]):
    total=sum(max(0,len(t)-1) for t in texts)
    if total<=0: raise ValueError('training chunks must be at least 2 bytes')
    g.fitness=sum(int(v) for v in counts)/total
    g.case_scores=[int(v)/max(1,len(t)-1) for t,v in zip(texts,counts)]


def save_greedy_checkpoint(path: str, trial: int, current: Genome,
                           history: Sequence[dict], rng: random.Random,
                           rolling: RollingTrainingCases, best_observed: float):
    if not path: return
    payload={'version':CHECKPOINT_VERSION,'kind':'ngram-greedy-v1',
             'trial':trial,'current':genome_object(current),'history':list(history),
             'rng':rng.getstate(),'rolling_cases':rolling.snapshot(),
             'best_observed':best_observed}
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.tmp')
    with tmp.open('wb') as f: pickle.dump(payload,f,protocol=5)
    os.replace(tmp,dest)


def load_greedy_checkpoint(path: str):
    # Existing v4 saved-model JSON is a valid starting point. Legacy GA pickle
    # checkpoints are importable, but their population/history aren't retained.
    if path.lower().endswith('.json'):
        return 0,genome_from_object(json.loads(Path(path).read_text())),[],None,None,-1.0
    with open(path,'rb') as f:
        data=pickle.load(f)  # trusted local checkpoints only
    if data.get('version')!=CHECKPOINT_VERSION:
        raise ValueError('Incompatible checkpoint version')
    if data.get('kind')=='ngram-greedy-v1':
        return (data['trial'],genome_from_object(data['current']),
                data['history'],data['rng'],data['rolling_cases'],
                data.get('best_observed',-1.0))
    if 'best' in data:  # migration from previous GA; discard other genomes
        return 0,genome_from_object(data['best']),[],None,None,-1.0
    raise ValueError('Not a Lens greedy or legacy GA checkpoint')


def evolve(args) -> Genome:
    """Sequential stochastic hill climbing; exactly one active genome."""
    show_progress=not args.no_tqdm
    if show_progress and tqdm is None:
        raise RuntimeError('tqdm is required; pip install tqdm or specify --no-tqdm')
    rng=random.Random(args.seed)
    corpus=load_corpus(args.local_corpus,args.min_chunk,args.corpus_chunks,args.max_chunk)
    if args.load:
        first,current,history,rng_state,rotation_state,best_observed=load_greedy_checkpoint(args.load)
        if rng_state is not None: rng.setstate(rng_state)
        args.rules=len(current.rules)
    else:
        first=0;history=[];rotation_state=None;best_observed=-1.0
        bigrams=_bigram_seeds(corpus)
        if not bigrams: raise ValueError('training corpus needs at least one bigram')
        current=new_genome(args.rules,rng,corpus,bigrams,not args.no_embedding)
    if not current.rules: raise ValueError('genome needs at least one rule')
    rolling=RollingTrainingCases(corpus,args.cases,args.seed*1000003,
                                 args.case_rotate_every,rotation_state)
    if rotation_state is None and first:
        rolling.for_generation(first-1)
    if args.history_csv: write_history(args.history_csv,history)
    progress=(tqdm(total=max(0,args.generations-first),desc='Lens greedy n-gram',
                   unit='trial',dynamic_ncols=True,mininterval=0.5)
              if show_progress else None)
    previous_rotation=rolling.rotations_done
    accepted=0
    for trial in range(first,args.generations):
        start=time.perf_counter()
        train=rolling.for_generation(trial)
        changed_cases=(trial>0 and trial%args.case_rotate_every==0)
        previous_rotation=rolling.rotations_done
        # Select exactly one rule, trace its genuine pre-rule and final-sweep
        # states while computing the baseline on this SAME active set of 8 texts.
        rule_index=rng.randrange(len(current.rules))
        before,after,base_counts=trace_rule(current,train,rule_index,
            samples=args.ngram_trace_samples,max_bytes=args.ngram_trace_bytes,
            backend=args.backend)
        _accuracy_from_counts(current,base_counts,train)
        base_acc=current.fitness
        proposal,mode=propose_ngram_rule(current.rules[rule_index],before,after,rng,
            top_k=args.ngram_top_k,max_blocks=args.ngram_max_blocks,
            max_literal=args.ngram_max_literal,budget=args.ngram_budget)
        changed=(proposal!=current.rules[rule_index])
        approved=False
        candidate_acc=base_acc
        if changed:
            candidate=Genome(current.rules.copy(),current.embedding.copy())
            candidate.rules[rule_index]=proposal
            score_population([candidate],train,args.backend,args.mps_batch,
                             args.cpu_workers,use_cache=not args.no_eval_cache)
            candidate_acc=candidate.fitness
            if candidate_acc > base_acc:  # strictly better; ties do not drift
                current=candidate
                approved=True
                accepted+=1
                save_model(args.save,current)
        best_observed=max(best_observed,base_acc,candidate_acc if approved else base_acc)
        elapsed=time.perf_counter()-start
        row={'generation':trial,'best_accuracy':current.fitness,
             'mean_accuracy':current.fitness,'median_accuracy':current.fitness,
             'best_ever_accuracy':best_observed,'baseline_accuracy':base_acc,
             'candidate_accuracy':candidate_acc,'accepted':int(approved),
             'changed':int(changed),'mutated_rule':rule_index,'mutation_mode':mode,
             'correct_bytes':round(current.fitness*sum(len(t)-1 for t in train)),
             'evaluated_bytes':sum(len(t)-1 for t in train),
             'eval_seconds':elapsed,'rules':len(current.rules),'tokens':256,
             'backend':args.backend,'case_rotation':rolling.rotations_done,
             'rotated_case':((rolling.rotations_done-1)%args.cases
                  if changed_cases and trial>0 else -1)}
        history.append(row)
        if args.history_csv: append_history_row(args.history_csv,row)
        if not args.no_plot and (trial+1)%args.plot_every==0:
            plot_history(history,args.plot_prefix,args.plot_window)
        msg=(f'trial={trial:7d} acc={current.fitness:.6f} base={base_acc:.6f} '
             f'candidate={candidate_acc:.6f} accepted={int(approved)} '
             f'rule={rule_index} mode={mode} rotation={rolling.rotations_done} '
             f'sec={elapsed:.2f}')
        if progress:
            progress.update(1)
            progress.set_postfix_str(f'acc={current.fitness:.5f} accept={accepted}',refresh=False)
            if trial==first or approved or (trial+1)%16==0 or changed_cases:
                tqdm.write(msg)
        else: print(msg,flush=True)
        if args.checkpoint and args.checkpoint_every>0 and (trial+1)%args.checkpoint_every==0:
            save_greedy_checkpoint(args.checkpoint,trial+1,current,history,rng,rolling,best_observed)
    if progress: progress.close()
    if args.checkpoint:
        save_greedy_checkpoint(args.checkpoint,args.generations,current,history,rng,rolling,best_observed)
    save_model(args.save,current)
    if not args.no_plot: plot_history(history,args.plot_prefix,args.plot_window)
    return current

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
    assert binary_opcode(0, 1, 2) == -113
    assert is_valid_rule(Rule([65,-1,66,-1,0], [binary_opcode(0,1,2)]))
    assert _emit_binary(2,[1,2],[3,4],lut,inv) == [3,10,8]
    assert _emit_binary(7,[1,2],[3,4],lut,inv) == [4,11,6]
    assert valid_lut(mutate_lut(lut,r,True))
    print('PASS: autoregressive teacher-forcing, expanding state, unbounded rule length, 256 LUT, binary opcodes')


def parse_args():
    ap=argparse.ArgumentParser(description='Lens: single-genome n-gram-guided greedy optimizer')
    ap.add_argument('--backend',choices=['cpu','mps','python'],default='cpu',
                    help='CPU native evaluator by default; MPS falls back on expanding/binary rules')
    ap.add_argument('--local-corpus',default='github-code.txt')
    ap.add_argument('--corpus-chunks',type=int,default=20000)
    ap.add_argument('--min-chunk',type=int,default=2)
    ap.add_argument('--max-chunk',type=int,default=3500,
                    help='exclusive maximum chunk length; skip oversize chunks, never crop')
    ap.add_argument('--cases',type=int,default=8)
    ap.add_argument('--case-rotate-every',type=int,default=16,
                    help='replace exactly one of the eight chunks every 16 trials')
    ap.add_argument('--rules',type=int,default=1500)
    ap.add_argument('--generations',type=int,default=1000000,
                    help='number of single-rule mutation trials')
    ap.add_argument('--ngram-top-k',type=int,default=5)
    ap.add_argument('--ngram-max-blocks',type=int,default=6,
                    help='maximum literal blocks in a pattern, joined by wildcards')
    ap.add_argument('--ngram-max-literal',type=int,default=12,
                    help='scale of block sizes; contiguous candidates can be 2x this')
    ap.add_argument('--ngram-budget',type=int,default=100000,
                    help='maximum counted windows per proposal (default exact for 8x6x384 snapshots)')
    ap.add_argument('--ngram-trace-samples',type=int,default=6,
                    help='time snapshots per text for the selected rule')
    ap.add_argument('--ngram-trace-bytes',type=int,default=384,
                    help='max contiguous tail bytes per observational snapshot')
    ap.add_argument('--no-embedding',action='store_true',
                    help='start with identity LUT (the LUT is held constant while hill climbing)')
    ap.add_argument('--mps-batch',type=int,default=64)
    ap.add_argument('--no-eval-cache',action='store_true')
    ap.add_argument('--no-tqdm',action='store_true')
    ap.add_argument('--cpu-workers',type=int,default=0)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--checkpoint',default='lens_ngram_checkpoint.pkl')
    ap.add_argument('--checkpoint-every',type=int,default=10)
    ap.add_argument('--load',default='',help='resume a greedy checkpoint or import v4 saved model/GA checkpoint')
    ap.add_argument('--save',default='best_lens_ar.json')
    ap.add_argument('--history-csv',default='lens_ar_history.csv')
    ap.add_argument('--plot-prefix',default='lens_ar')
    ap.add_argument('--plot-window',type=int,default=3000)
    ap.add_argument('--plot-every',type=int,default=1)
    ap.add_argument('--no-plot',action='store_true')
    ap.add_argument('--generate',default='')
    ap.add_argument('--prompt',default='Hello')
    ap.add_argument('--output-bytes',type=int,default=64)
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    for name in ('cases','case_rotate_every','rules','generations','ngram_top_k',
                 'ngram_max_blocks','ngram_max_literal','ngram_budget',
                 'ngram_trace_samples','ngram_trace_bytes','plot_every'):
        if getattr(args,name)<1: ap.error(f'--{name.replace("_","-")} must be positive')
    return args


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
