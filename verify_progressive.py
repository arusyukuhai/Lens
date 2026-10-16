#!/usr/bin/env python3
"""Independent checks of progressive online moments, row coverage and exact reuse.

These test the design numerically, NOT the compiled Nim implementation.
"""
import math
import random
from pathlib import Path


def order(n):
    if n <= 0:
        return []
    q = [(0, n)]
    out = []
    for lo, hi in q:
        if lo >= hi:
            continue
        mid = lo + (hi - lo) // 2
        out.append(mid)
        if lo < mid:
            q.append((lo, mid))
        if mid + 1 < hi:
            q.append((mid + 1, hi))
    return out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    xy = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    xx = sum((x-ma)**2 for x in a)
    yy = sum((y-mb)**2 for y in b)
    return abs(xy) / math.sqrt(xx * yy) if xx > 1e-24 and yy > 1e-24 else 0.0


def online(a, b):
    mx = my = xx = yy = cross = 0.0
    for k, (x, y) in enumerate(zip(a, b), 1):
        dx, dy = x-mx, y-my
        mx += dx/k
        my += dy/k
        xx += dx*(x-mx)
        yy += dy*(y-my)
        cross += dx*(y-my)
    return min(1.0, abs(cross)/math.sqrt(xx*yy)) if xx>1e-24 and yy>1e-24 else 0.0


rng = random.Random(20261016)
for n in [1, 2, 3, 7, 8, 17, 32, 64, 65, 96, 319, 1001, 8192]:
    indices = order(n)
    assert len(indices) == n and sorted(indices) == list(range(n))
    for _ in range(30):
        xs = [rng.random()*5 for _ in range(n)]
        residuals = [rng.gauss(0,1) + x*0.23 for x in xs]
        assert abs(online(xs, residuals)-pearson(xs, residuals)) < 2e-12

# Verify that partially measured exact match features can be inserted into
# their original row positions and scanning the remaining rows yields the
# same full vector, with counts normalized by arbitrary original lengths.
for _ in range(150):
    n = rng.randrange(65, 400)
    pattern = rng.randrange(1, 16)
    texts = [rng.choices(range(16), k=rng.randrange(6, 90)) for _ in range(n)]
    original_lengths = [rng.randrange(8, 150) for _ in range(n)]
    full = [row.count(pattern) / original_lengths[i] for i, row in enumerate(texts)]
    k = rng.randrange(16, n)
    pos = order(n)[:k]
    partial = [full[i] for i in pos]
    rebuilt = [None] * n
    for i, value in zip(pos, partial):
        rebuilt[i] = value
    for i in range(n):
        if rebuilt[i] is None:
            rebuilt[i] = texts[i].count(pattern) / original_lengths[i]
    assert rebuilt == full

code = Path(__file__).with_name('cms.nim').read_text()
checks = {
    'density uses pre-noise length': 'originalLengths[i]' in code,
    'config progressive toggle': 'of "progressive": result.progressive' in code,
    'config start': 'of "progressive-start": result.progressiveStart' in code,
    'config keep': 'of "progressive-keep": result.progressiveKeep' in code,
    'config factor': 'of "progressive-factor": result.progressiveFactor' in code,
    'online stats O(1)': 'proc observe(s: var OnlinePearson' in code,
    'online residual Pearson': 'online[j].observe(x, fit.residual[i])' in code,
    'exact finalist Ridge': 'candidate.gain = candidateGain(counts, fit)' in code,
    'reuse samples for exact': 'countWithProgressiveSamples(texts, candidate.pat' in code,
    'fine still each generation': 'let selectedAdd = rankBranches(fineRows' in code,
    'selftest on-off': 'progressiveCfg.progressive = false' in code,
    'selftest cached': 'cachedInfo.reusedPairs > 0' in code,
}
for k, v in checks.items():
    print(('OK' if v else 'FAIL'), k)
assert all(checks.values())
print('PASS: 390 online correlation vectors, 150 exact-count resumptions, row schedules and source checks')
