#!/usr/bin/env python3
"""Standalone reference tests for the GA chunk rotation policy.
Note: validates the intended mathematics/schedule; Nim --self-test also checks runtime.
"""
import random
from pathlib import Path

src = (Path(__file__).parent / 'cms.nim').read_text()

def sampled_rotation(n, generations, batch, every=1, stride=1, seed=1729):
    rng = random.Random(seed)
    pool = list(range(n))
    rng.shuffle(pool)
    result = []
    for gen in range(generations):
        step = gen // every % n
        start = step * (stride % n) % n
        result.append([pool[(start+i) % n] for i in range(batch)])
    return result

for n in [3, 5, 17, 31, 37, 101]:
    for train in [1, 2, min(n, 5), n]:
        batches = sampled_rotation(n, n + 2, train)
        assert batches[0] == batches[n]
        assert set.union(*map(set, batches[:n])) == set(range(n))
        assert all(len(b) == len(set(b)) == train for b in batches)
        assert all(0 <= idx < n for b in batches for idx in b)

for every in (2, 3, 4, 7):
    batches = sampled_rotation(23, 3*every, 4, every=every)
    assert all(batches[i] == batches[i // every * every]
               for i in range(len(batches)))
    assert batches[0] != batches[every]

for stride in (1, 2, 4, 11):
    batches = sampled_rotation(31, 31, 3, stride=stride)
    assert set.union(*map(set, batches)) == set(range(31))

# The actual Nim code must hold to the same safety properties.
assert 'source[0..<poolSize]' in src, 'raw held-out tokens must not influence noise alphabet'
assert 'for i in poolSize..<source.len: validationIds.add(i)' in src
assert 'let batchIds = trainingIndices(order, gen, cfg.trainChunks' in src
assert 'for i in 0..<population.len: population[i].score = -Inf' in src
assert 'fit(fixedRows, population[i], cfg.ridgeLambda, fixedCache, false).looPearson' in src
assert 'archive.sort(proc(a, b: ArchiveEntry)' in src
assert 'let score = fit(fixedRows' in src
assert 'rotation.dat' in src and 'rotation_order.dat' in src
assert '(cfg.chunks == 0 or result.len < cfg.chunks)' in src
print('PASS: rotation covers the training pool, wraps, respects cadence,')
print('      excludes validation, and separates rotating vs fixed scores.')
