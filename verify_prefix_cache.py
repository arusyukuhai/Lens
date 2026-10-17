#!/usr/bin/env python3
"""Static invariants + small independent shared-prefix reference model.
This does not compile Nim; Nim --self-test covers runtime equivalent paths.
"""
import random
from pathlib import Path
s = Path(__file__).with_name('cms.nim').read_text()
checks = [
    'proc commonPrefix(a, b: seq[Gene]): int =',
    'geneEqual(a[result], b[result])',
    'if cp.ruleEnd <= prefix and cp.ruleEnd > startRule:',
    'states[i] = trace.checkpoints[checkpointIndex].states[i]',
    'result[j] = trace.features[j]',
    'clearFeatureCache(trainCache)',
    'var fixedCache = newFeatureCache',
    'var validCache = newFeatureCache',
    'if tok >= 256 and not produced[tok - 256]:',
    'let parentModel = fit(rows, parent, cfg.ridgeLambda, cache)',
    'if not fullModel: return',
    'proc benchCache() =',
    'cache_stats.dat',
    'doAssert getFeaturesReference(rows, deadMarker) ==',
]
for check in checks:
    assert check in s, f'missing required feature: {check}'
assert 'clearFeatureCache(fixedCache)' not in s
assert 'clearFeatureCache(validCache)' not in s

# Abstract rewrite simulator: first match marker propagation; random genomes
# with shared and divergent prefixes must yield identical rule counts/states.
def apply(seq, rule, j):
    token = rule
    n = seq.count(token)
    return tuple(256+j if x == token else x for x in seq), n

def raw(g, rows):
    states = list(rows)
    features = []
    for j, rule in enumerate(g):
        counts = []
        for i in range(len(states)):
            states[i], n = apply(states[i], rule, j)
            counts.append(n)
        features.append(tuple(counts))
    return features

def cached(g, rows, saved, stride=7):
    match = 0
    if saved is not None:
        while match < len(g) and g[match] == saved['g'][match]: match += 1
    start = match // stride * stride
    feats = []
    checkpoints = {}
    if start:
        feats.extend(saved['feats'][:start])
        states = list(saved['checkpoints'][start])
        checkpoints.update({k: v for k, v in saved['checkpoints'].items() if k <= start})
    else:
        states = list(rows)
    for j in range(start, len(g)):
        counts = []
        for i in range(len(states)):
            states[i], n = apply(states[i], g[j], j)
            counts.append(n)
        feats.append(tuple(counts))
        if (j+1) % stride == 0:
            checkpoints[j+1] = tuple(states)
    assert feats == raw(g, rows)
    return {'g': list(g), 'feats': feats, 'checkpoints': checkpoints}

rng = random.Random(317)
rows = [tuple(rng.randrange(6) for _ in range(20)) for _ in range(5)]
g = [rng.randrange(6) if rng.random() < .75 else 256+rng.randrange(j) if j else 2 for j in range(90)]
entry = cached(g, rows, None)
for _ in range(100):
    other = list(entry['g'])
    for __ in range(rng.choice([1, 1, 1, 10, 50])):
        i = rng.randrange(len(other))
        other[i] = rng.randrange(6) if rng.random() < .6 else (256+rng.randrange(i) if i else 0)
    entry = cached(other, rows, entry)
print('PASS: cache source invariants and 100 randomized prefix-reuse equivalence checks')
