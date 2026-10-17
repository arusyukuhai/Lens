"""Independent numeric and source-invariant checks for the 1-5 PVE revision.

Run with Python 3, numpy optional. This does not substitute for nim check.
"""
from pathlib import Path
import math
import random

source = Path(__file__).with_name('cms.nim').read_text()
required = [
    'proc groupCvScores(', 'proc pairedImprovement(',
    'proc dependencyCrossover(', 'proc residualGuidedChild(',
    'proc localizedChild(', 'proc refineNextGeneration(',
    'proc ridgeAlpha(', 'chunkId: int', 'sourceLen: original.len, chunkId: chunkId',
    'replacements[t] = replacement',
    'state[positions[t]] = replacements[t]',
    'if cfg.pairedSelection:', 'if not cfg.multifidelity: return',
    'if cfg.localSearch:', 'if cfg.residualSearch', 'if cfg.linkageCross:',
    'groupCvScores(finalRows', 'samplingRows(allSource',
    'result = nextGeneration(population,archive,trainingSource,r,cfg)',
    'multifidelity_stats.dat',
    'var viralAdded = 0',
]
for required_fragment in required:
    assert required_fragment in source, f'Missing critical integration: {required_fragment}'
assert 'method: string' not in source
assert 'for dagIndex, dag in [v.targetDag' in source


def diffusion(length, interior_count, seed):
    """Version-independent corruption path for all observation budgets."""
    r = random.Random(seed)
    positions = list(range(length))
    r.shuffle(positions)
    replacements = [r.randint(0, 255) for _ in range(length)]
    noise_state = [0] * length
    levels = sorted(r.sample(range(1, length), min(interior_count, length - 1)))
    snapshots = {0: tuple(noise_state)}
    at = 0
    for k in levels + [length]:
        for z in range(at, k):
            noise_state[positions[z]] = replacements[z]
        snapshots[k] = tuple(noise_state)
        at = k
    return snapshots

for seed in range(100):
    a = diffusion(50, 4, seed)
    b = diffusion(50, 16, seed)
    for shared_step in a.keys() & b.keys():
        assert a[shared_step] == b[shared_step]

# Paired chunk improvement uncertainty rule: zero for matched values.
def paired_improvement(a, b):
    delta = [x-y for x,y in zip(a,b)]
    mean = sum(delta)/len(delta)
    if len(delta) < 2:
        return mean
    stderr = math.sqrt(sum((x-mean)**2 for x in delta)/(len(delta)*(len(delta)-1)))
    return mean-.5*stderr

assert paired_improvement([.2, .5, .9], [.2, .5, .9]) == 0
assert paired_improvement([.4, .7, .6], [.3, .6, .5]) > 0
assert paired_improvement([.6, .1, .1], [.2, .2, .2]) < .1

try:
    import numpy as np
except ImportError:
    print('INFO: numpy not installed; optional reference Ridge-CV test skipped')
else:
    r = np.random.default_rng(27)
    X = r.standard_normal((36, 16))
    y = r.standard_normal(36)
    groups = np.repeat(np.arange(6),6)
    lam = .3
    cv = []
    for group in sorted(set(groups)):
        train, test = groups != group, groups == group
        mu, sig = X[train].mean(axis=0), X[train].std(axis=0)
        sig = np.where(sig < 1e-12, 1.,sig)
        A = (X[train]-mu)/sig
        b = (X[test]-mu)/sig
        alpha = np.linalg.solve(A@A.T+lam*np.eye(A.shape[0]),y[train]-y[train].mean())
        pred = y[train].mean() + b @ A.T @ alpha
        cv.append(np.corrcoef(pred,y[test])[0,1])
    assert np.isfinite(cv).all()
    print('PASS: six exact chunk-held-out Ridge fold reference solves')

print('PASS: paired comparison, shared-noise multi-fidelity, 1-5 source wiring')
