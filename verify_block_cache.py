#!/usr/bin/env python3
"""Independent randomized semantics model for safe per-row interval reuse.
This is NOT a Nim build. Nim --self-test compares against the real replace.nim.
"""
from pathlib import Path
import random
from collections import deque

s = Path(__file__).with_name('cms.nim').read_text()
for phrase in (
    'proc diffusionLevels(length, noiseSteps: int, r: var Rng,',
    'result.sort()',
    'proc blockGeneHash(genes: seq[Gene], first, after: int): uint64 =',
    'proc blockRowHash(geneHash: uint64, input: seq[int]): uint64 =',
    'entry.input == input and sameBlockGenes(genomeGenes, first, entry.genes)',
    'if cache.maxBlockBytes > 0 and cache.maxBlockEntries > 0:',
    'for j in first..<after:',
    'var intervalCache = newFeatureCache(0, 3, 0, 8, 300)',
    'doAssert getFeatures(rows, invisibleEdit, intervalCache) ==',
    'clearFeatureCache(trainCache)',
    'cfg.diffusionSampling',
    'benchBlockCache()',
):
    assert phrase in s, phrase

# For every source/chunk, diff sampling must use an independent stable seed.
# Verify the mathematics of a without-replacement random stage sampler.
def levels(n, steps, seed):
    rng = random.Random(seed)
    candidates = list(range(1, n))
    take = min(steps-1, n-1)
    for i in range(take):
        j = rng.randrange(i, len(candidates))
        candidates[i], candidates[j] = candidates[j], candidates[i]
    return [0] + sorted(candidates[:take]) + [n]

rng = random.Random(512)
for _ in range(500):
    n = rng.randrange(2, 1500)
    steps = rng.randrange(2, 80)
    seed = rng.randrange(1 << 32)
    sample = levels(n, steps, seed)
    assert sample == levels(n, steps, seed)
    assert sample[0] == 0 and sample[-1] == n
    assert all(x < y for x, y in zip(sample, sample[1:]))
    assert len(sample) == min(steps-1, n-1) + 2
    assert all(0 <= 1-x/n <= 1 for x in sample)

# Simplified deterministic replacement language: every gene substitutes one
# token and emits a rule-index marker. Catches errors when no-op edits lead
# to different genomes but identical intermediate row states.
def apply(gene, state, locus):
    count = state.count(gene)
    return tuple(256+locus if ch == gene else ch for ch in state), count

def raw(genome, rows):
    cols = [[0]*len(rows) for _ in genome]
    states = list(rows)
    for locus, gene in enumerate(genome):
        for i, state in enumerate(states):
            states[i], count = apply(gene, state, locus)
            cols[locus][i] = count
    return cols

def cached(genome, rows, cache, stride=8):
    cols = [[0]*len(rows) for _ in genome]
    states = list(rows)
    hits = 0
    for first in range(0, len(genome), stride):
        stop = min(len(genome), first+stride)
        signature = tuple(genome[first:stop])
        for i, state in enumerate(states):
            key = (first, stop, signature, state)
            if key in cache:
                states[i], counts = cache[key]
                hits += 1
            else:
                current = state
                counts = []
                for j, gene in enumerate(signature, first):
                    current, count = apply(gene, current, j)
                    counts.append(count)
                states[i] = current
                cache[key] = (current, counts)
            for j, count in enumerate(counts, first):
                cols[j][i] = count
    return cols, hits

rows = [tuple(rng.randrange(6) for _ in range(rng.randint(20, 50))) for _ in range(12)]
base = [rng.randrange(6) if rng.random() < .70 else 200 + rng.randrange(30) for _ in range(130)]
cache = {}
assert cached(base, rows, cache)[0] == raw(base, rows)
hits = 0
for t in range(350):
    g = base.copy()
    for _ in range(rng.choice([1, 1, 2, 5, 30])):
        index = rng.randrange(130)
        g[index] = rng.choice([rng.randrange(6), 200 + rng.randrange(30), 256 + rng.randrange(index) if index else 3])
    actual, reused = cached(g, rows, cache)
    hits += reused
    assert actual == raw(g, rows), f'diverged at iteration {t}'
assert hits > 0
print('PASS: 500 random diffusion-level sets, 350 divergent genomes, interval reuse == serial computation; hits', hits)
