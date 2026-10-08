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

A one-pass nonexpanding network ensures memory grows at most with the context,
without an arbitrary max-memory setting. GPU buffers are dynamically sized to
the actual training examples and thus limited only by physical resources.
"""
from __future__ import annotations

import argparse
import copy
import csv
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
MAX_RULE_TOKENS = 64
MAX_WILDCARDS = 16
PREDICTION_SLOT = 0  # byte NUL is used as the slot placeholder, not a 257th token
CHECKPOINT_VERSION = 4
_MPS_CROSSCHECK_DONE = False
_NATIVE_CROSSCHECK_DONE = False
_NATIVE_UNAVAILABLE_WARNED = False
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


def is_nonexpanding(rule: Rule) -> bool:
    """Sufficient and necessary condition for ANY capture lengths."""
    p = rule.pattern
    q = rule.replacement
    if not p or len(p) > MAX_RULE_TOKENS or len(q) > MAX_RULE_TOKENS:
        return False
    if any(v != -1 and not 0 <= v < 256 for v in p):
        return False
    wc = p.count(-1)
    if wc > MAX_WILDCARDS or all(v == -1 for v in p):
        return False
    lit_budget = sum(v >= 0 for v in p)
    if sum(v >= 0 for v in q) > lit_budget:
        return False
    used = set()
    for op in q:
        if op >= 0:
            if op > 255:
                return False
            continue
        if op < -111:
            return False
        if wc < 1:
            return False
        ci = ((-op - 16) % 16 + 1) if op <= -16 else -op
        if not 1 <= ci <= wc or ci in used:
            return False
        used.add(ci)
    return True


def repair_rule(rule: Rule, rnd: random.Random | None = None) -> Rule:
    """Normalize old tokens and guarantee hard nonexpanding invariant."""
    rng = rnd or random
    pattern = [(v if 0 <= v < TOKEN_COUNT else -1) for v in rule.pattern[:MAX_RULE_TOKENS]]
    if not pattern:
        pattern = [rng.randrange(256)]
    cnt = 0
    for i, v in enumerate(pattern):
        if v == -1:
            cnt += 1
            if cnt > MAX_WILDCARDS:
                pattern[i] = rng.randrange(256)
    if all(t == -1 for t in pattern):
        pattern[0] = rng.randrange(256)
    wc = pattern.count(-1)
    remaining = sum(t >= 0 for t in pattern)
    captured = set()
    replacement = []
    for op in rule.replacement[:MAX_RULE_TOKENS]:
        if op >= 0:
            if remaining > 0:
                replacement.append(op % 256)
                remaining -= 1
        elif -111 <= op < 0 and wc:
            ci = ((-op - 16) % 16 + 1) if op <= -16 else -op
            if ci <= wc and ci not in captured:
                replacement.append(op)
                captured.add(ci)
    if not replacement:
        replacement = [-1] if wc else [pattern[0]]
    out = Rule(pattern, replacement)
    assert is_nonexpanding(out)
    return out


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
                      lut: Sequence[int], inverse: Sequence[int], allowance: int) -> list[int]:
    out = []
    for op in rep:
        if op >= 0:
            if len(out) < allowance:
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
        # Never partially emit a capture: matches original semantics.
        if len(out) + len(vals) > allowance:
            continue
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
        out.extend(_emit_replacement(rule.replacement, caps, lut, inverse, finish-s))
        prev = finish
        scan = finish
    if not matched:
        return list(state)
    out.extend(state[prev:])
    if len(out) > n:
        raise AssertionError('Nonexpanding invariant broken')
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
    """Sample full corpus records, not truncated prefixes; rotate each generation."""
    rng=random.Random(seed)
    if len(corpus)>=cases:
        return rng.sample(list(corpus),cases)
    return [rng.choice(corpus) for _ in range(cases)]


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
        size=rng.randint(1,min(5,len(x)))
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
    p=[rng.randrange(256) for _ in range(rng.randint(1,6))]
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
    return Genome([repair_rule(make_rule(rng,corpus,bigrams),rng) for _ in range(rules)], lut)


def log_uniform_count(rng: random.Random, maximum: int) -> int:
    return max(1,min(maximum,int(round(math.exp(rng.uniform(0,math.log(max(1,maximum))))))))


def mutate_lut(lut: list[int], rng: random.Random, enabled: bool) -> list[int]:
    if not enabled or rng.random()>=0.25:
        return list(lut)
    out=list(lut)
    k=log_uniform_count(rng,256)
    for _ in range(k):
        x,y=rng.sample(range(256),2)
        out[x],out[y]=out[y],out[x]
    return out


def crossover_lut(a: Sequence[int], b: Sequence[int], rng: random.Random) -> list[int]:
    """Cycle-safe partial permutation crossover."""
    out=list(a)
    inverse={v:i for i,v in enumerate(out)}
    for i in rng.sample(range(256),log_uniform_count(rng,256)):
        want=b[i]
        if out[i]==want:
            continue
        j=inverse[want]
        inverse[out[i]]=j
        inverse[want]=i
        out[i],out[j]=out[j],out[i]
    return out


def mutate_rule(rule: Rule, rng: random.Random, corpus, bigrams) -> Rule:
    if rng.random()<0.12:
        return repair_rule(make_rule(rng,corpus,bigrams),rng)
    p=rule.pattern.copy()
    q=rule.replacement.copy()
    for _ in range(log_uniform_count(rng,max(1,len(p)+len(q)))):
        action=rng.randrange(11)
        s = p if rng.random()<0.5 else q
        if action<=4 and s:
            pos=rng.randrange(len(s))
            if s is p:
                s[pos]=(-1 if rng.random()<0.10 else rng.choice(bigrams)[rng.randrange(2)])
            else:
                wc=p.count(-1)
                if wc and rng.random()<0.35:
                    family=rng.randrange(7)
                    cap=rng.randint(1,min(15 if family==0 else 16,wc))
                    s[pos]=-(cap if family==0 else 16+(family-1)*16+cap-1)
                else:
                    s[pos]=rng.randrange(256)
        elif action==5 and len(s)>1:
            x,y=rng.sample(range(len(s)),2)
            s[x],s[y]=s[y],s[x]
        elif action==6 and len(s)<MAX_RULE_TOKENS:
            s.insert(rng.randrange(len(s)+1),rng.randrange(256))
        elif action==7 and len(s)>1:
            s.pop(rng.randrange(len(s)))
        elif action==8 and s:
            x=rng.randrange(len(s))
            y=rng.randrange(x,len(s))
            s[x:y+1]=reversed(s[x:y+1])
        elif action==9 and rng.random()<0.3:
            p[:]=make_rule(rng,corpus,bigrams).pattern
        elif action==10 and rng.random()<0.3:
            q[:]=make_rule(rng,corpus,bigrams).replacement
    return repair_rule(Rule(p,q),rng)


def mutate_genome(parent: Genome,rng: random.Random,corpus,bigrams,enabled=True) -> Genome:
    rows=parent.rules.copy()
    n=len(rows)
    if n:
        k=min(n,log_uniform_count(rng,n))
        for ri in rng.sample(range(n),k):
            rows[ri]=mutate_rule(rows[ri],rng,corpus,bigrams)
        if n>1 and rng.random()<0.12:
            i,j=rng.sample(range(n),2)
            rows[i],rows[j]=rows[j],rows[i]
    return Genome(rows,mutate_lut(parent.embedding,rng,enabled))


def breed(a: Genome,b: Genome,rng: random.Random,corpus,bigrams,embedding_enabled: bool) -> Genome:
    n=len(a.rules)
    if n<=1:
        rows=a.rules.copy()
    elif rng.random()<0.5:
        i,j=sorted(rng.sample(range(n+1),2))
        rows=a.rules[:i]+b.rules[i:j]+a.rules[j:]
    else:
        # Position-wise sparse donor transplant, then regular mutation.
        rows=a.rules.copy()
        for i in rng.sample(range(n),log_uniform_count(rng,n)):
            rows[i]=b.rules[i]
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
        writer=csv.DictWriter(f,fieldnames=list(history[0]))
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
    ax.plot(x,y,alpha=.5,label='Generation best next-byte accuracy')
    ax.plot(x,avg,label='Moving average')
    ax.plot(x,np.maximum.accumulate(y),label='Best observed')
    ax.set(xlabel='Generation',ylabel='Teacher-forced next-byte accuracy',title='Lens autoregressive training')
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
    if not valid_lut(g.embedding) or not all(map(is_nonexpanding,g.rules)):
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
                    history:list[dict],rng:random.Random, archive: Sequence[Genome] = ()) -> None:
    if not path:
        return
    payload={'version':CHECKPOINT_VERSION,'generation':generation,
             'population':[genome_object(g) for g in population],
             'best':genome_object(best),'history':history,'rng':rng.getstate(),
             'archive':[genome_object(g) for g in archive]}
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.tmp')
    with tmp.open('wb') as f:
        pickle.dump(payload,f,protocol=5)
    os.replace(tmp,dest)


def load_checkpoint(path:str):
    with open(path,'rb') as f:
        v=pickle.load(f)
    if v.get('version')!=CHECKPOINT_VERSION:
        raise ValueError('Old checkpoints have incompatible token counts and training goals; start a new v4 run')
    # IMPORTANT: Only open checkpoint files from trusted sources (pickle).
    rng=random.Random()
    rng.setstate(v['rng'])
    return v['generation'], [genome_from_object(g) for g in v['population']], genome_from_object(v['best']),v['history'],rng,[genome_from_object(g) for g in v.get('archive',[])]


def evolve(args) -> Genome:
    show_progress = not getattr(args, 'no_tqdm', False)
    if show_progress and tqdm is None:
        raise RuntimeError('tqdm が必要です: python -m pip install tqdm （または --no-tqdm を指定）')
    rng=random.Random(args.seed)
    corpus=load_corpus(args.local_corpus,args.min_chunk,args.corpus_chunks,args.max_chunk)
    bigrams=_bigram_seeds(corpus)
    if not bigrams:
        raise ValueError('Training corpus needs adjacent token pairs')
    if args.load:
        first,population,best_ever,history,rng,archive=load_checkpoint(args.load)
        args.population=len(population)
        args.rules=len(population[0].rules)
    else:
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
    if args.history_csv:
        # On a resumed run, restore the completed rows once. Thereafter append
        # only the new row, including when started from generation zero.
        write_history(args.history_csv,history)
    generation_bar=(tqdm(total=max(0,args.generations-first),
                         desc='Lens 学習',unit='世代',dynamic_ncols=True,
                         mininterval=0.5,position=0,leave=True)
                    if show_progress else None)
    for generation in range(first,args.generations):
        t0=time.perf_counter()
        # Evaluation rotation is explicit; identical input sets in same generation.
        train=make_training_examples(corpus,args.cases,args.seed*1000003+generation//max(1,args.case_rotate_every))
        if generation_bar is not None:
            generation_bar.set_description_str(f'Lens 学習 gen={generation}')
        eval_bar=(tqdm(total=len(population),desc=f'gen={generation} 評価/{args.backend}',
                       unit='個体',position=1,leave=False,dynamic_ncols=True,
                       mininterval=0.5) if show_progress else None)
        try:
            score_population(population,train,args.backend,args.mps_batch,args.cpu_workers,
                             use_cache=not args.no_eval_cache,progress=eval_bar)
        finally:
            if eval_bar is not None:
                eval_bar.close()
        population.sort(key=lambda g:g.fitness,reverse=True)
        current=population[0]
        if best_ever is None or current.fitness>best_ever.fitness:
            best_ever=copy.deepcopy(current)
            save_model(args.save,best_ever)
        if args.hof_size>0:
            archive.append(copy.deepcopy(current))
            archive.sort(key=lambda g:g.fitness,reverse=True)
            unique=[]; seen=set()
            for g in archive:
                sig=fingerprint(g)
                if sig not in seen:
                    unique.append(g);seen.add(sig)
                if len(unique)>=args.hof_size:
                    break
            archive=unique
        mean_fit=statistics.fmean(g.fitness for g in population)
        median_fit=statistics.median(g.fitness for g in population)
        elapsed=time.perf_counter()-t0
        row={'generation':generation,'best_accuracy':current.fitness,
             'mean_accuracy':mean_fit,'median_accuracy':median_fit,
             'best_ever_accuracy':best_ever.fitness,
             'correct_bytes':round(current.fitness*sum(len(s)-1 for s in train)),
             'evaluated_bytes':sum(len(s)-1 for s in train),'eval_seconds':elapsed,
             'rules':args.rules,'tokens':256,'backend':args.backend}
        history.append(row)
        summary=(f"gen={generation:6d} acc={current.fitness:.5f} mean={mean_fit:.5f} "
                 f"ever={best_ever.fitness:.5f} tested={row['evaluated_bytes']} "
                 f"seconds={elapsed:.2f} backend={args.backend}")
        if generation_bar is not None:
            generation_bar.set_postfix_str(f'acc={current.fitness:.5f} eval={elapsed:.1f}s',refresh=False)
            tqdm.write(summary)
        else:
            print(summary,flush=True)
        if args.history_csv:
            append_history_row(args.history_csv,row)
        if not args.no_plot and (generation+1)%max(1,args.plot_every)==0:
            plot_history(history,args.plot_prefix,args.plot_window)
        next_pop=[copy.deepcopy(g) for g in population[:min(args.elites,args.population)]]
        if archive and args.hof_inject:
            for g in archive[:args.hof_inject]:
                if len(next_pop)<args.population and not any(fingerprint(x)==fingerprint(g) for x in next_pop):
                    next_pop.append(copy.deepcopy(g))
        breed_bar=(tqdm(total=args.population,initial=len(next_pop),
                        desc=f'gen={generation} 次世代作成',unit='個体',position=1,
                        dynamic_ncols=True,mininterval=0.5,leave=False)
                   if show_progress else None)
        def pick():
            # Tournament selection on teacher-forced byte accuracy.
            if archive and rng.random()<args.hof_parent_rate:
                return rng.choice(archive[:max(1,min(16,len(archive)))])
            return max(rng.choices(population,k=min(args.tournament,len(population))),key=lambda g:g.fitness)
        try:
            while len(next_pop)<args.population:
                if rng.random()<args.immigrant_rate:
                    child=new_genome(args.rules,rng,corpus,bigrams,not args.no_embedding)
                elif rng.random()<args.crossover_rate:
                    child=breed(pick(),pick(),rng,corpus,bigrams,not args.no_embedding)
                else:
                    child=mutate_genome(pick(),rng,corpus,bigrams,not args.no_embedding)
                next_pop.append(child)
                if breed_bar is not None:
                    breed_bar.update(1)
        finally:
            if breed_bar is not None:
                breed_bar.close()
        if args.checkpoint and args.checkpoint_every>0 and (generation+1)%args.checkpoint_every==0:
            save_checkpoint(args.checkpoint,generation+1,next_pop,best_ever,history,rng,archive)
        population=next_pop
        if generation_bar is not None:
            generation_bar.update(1)
    if generation_bar is not None:
        generation_bar.close()
    if best_ever is None:
        raise ValueError('no generations evaluated')
    if args.checkpoint:
        save_checkpoint(args.checkpoint,args.generations,population,best_ever,history,rng,archive)
    if not args.no_plot:
        plot_history(history,args.plot_prefix,args.plot_window)
    return best_ever


def self_test() -> None:
    lut=list(range(256))
    inv=inverse_lut(lut)
    assert is_nonexpanding(Rule([65,0],[65,66]))
    assert not is_nonexpanding(Rule([65,0],[65,66,67]))
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
        assert is_nonexpanding(p)
    assert valid_lut(mutate_lut(lut,r,True))
    print('PASS: autoregressive teacher-forcing, one sweep, 256 token LUT, sort, mutations, no memory truncation')


def parse_args():
    ap=argparse.ArgumentParser(description='Lens: stateful byte-level autoregressive Replacer GA')
    auto_backend = ('mps' if gpu_backend is not None and getattr(gpu_backend, 'torch', None) is not None
        and gpu_backend.torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--backend',choices=['cpu','mps','python'],default="cpu",
                    help='cpu uses a locally compiled C++ accelerator if available; python is the reference')
    ap.add_argument('--local-corpus',default='github-code.txt')
    ap.add_argument('--corpus-chunks',type=int,default=10000,help='maximum ===SPLIT=== separated chunks to load (not a state memory cap)')
    ap.add_argument('--min-chunk',type=int,default=2)
    ap.add_argument('--max-chunk',type=int,default=1500,
                    help='exclusive maximum chunk length in bytes: skip entire chunks of 1500 bytes or more (never crop)')
    ap.add_argument('--cases',type=int,default=8,help='whole-text examples evaluated per generation')
    ap.add_argument('--case-rotate-every',type=int,default=3)
    ap.add_argument('--population',type=int,default=450)
    ap.add_argument('--rules',type=int,default=1500)
    ap.add_argument('--generations',type=int,default=1000000)
    ap.add_argument('--elites',type=int,default=24)
    ap.add_argument('--tournament',type=int,default=8)
    ap.add_argument('--hof-size',type=int,default=64)
    ap.add_argument('--hof-inject',type=int,default=2)
    ap.add_argument('--hof-parent-rate',type=float,default=0.10)
    ap.add_argument('--crossover-rate',type=float,default=0.50)
    ap.add_argument('--immigrant-rate',type=float,default=0.02)
    ap.add_argument('--no-embedding',action='store_true',help='disable evolution of the 256-element operator LUT')
    ap.add_argument('--mps-batch',type=int,default=512,
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
    ap.add_argument('--plot-window',type=int,default=30)
    ap.add_argument('--plot-every',type=int,default=1)
    ap.add_argument('--no-plot',action='store_true')
    ap.add_argument('--generate',default='',help='path to a saved v4 model; generate without teacher forcing')
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
