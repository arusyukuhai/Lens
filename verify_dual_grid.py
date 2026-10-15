"""Independent Python reference tests for Nim dual-grid noise-count generation.

Checks formulas and source wiring; not a substitute for compiling Nim.
"""
from math import floor, isfinite, log
from pathlib import Path


def levels(length: int, power: float) -> list[int]:
    assert length >= 0 and isfinite(power) and power >= 1.0
    if length == 0:
        return [0]
    if power == 1.0:
        return list(range(length + 1))
    out = [0]
    k = 1
    while True:
        val = length if power * log(k) >= log(length) else min(length, max(1, floor(k ** power)))
        if val > out[-1]:
            out.append(val)
        if val == length:
            break
        k += 1
    stride = length ** (1.0 - 1.0 / power)
    k = 1
    while True:
        val = min(length, max(1, floor(k * stride)))
        out.append(val)
        if val == length:
            break
        k += 1
    return sorted(set(out))

assert levels(25, 1.0) == list(range(26))
assert levels(25, 2.0) == [0, 1, 4, 5, 9, 10, 15, 16, 20, 25]
assert levels(30, 2.0) == [0, 1, 4, 5, 9, 10, 16, 21, 25, 27, 30]
assert levels(25, 3.0) == [0, 1, 8, 17, 25]
assert levels(25, 1000) == [0, 1, 24, 25]
assert len(levels(2000, 2.0)) == 89
assert len(levels(2000, 1.5)) == 312
for n in (0, 1, 2, 3, 10, 25, 100, 1000, 2000, 10000):
    for p in (1.0, 1.0001, 1.2, 1.5, 1.75, 2.0, 2.25, 3.0, 5.0, 1000.0):
        result = levels(n, p)
        assert result[0] == 0 and result[-1] == n, (n, p)
        assert all(x < y for x, y in zip(result, result[1:])), (n, p)
        expected = {0, n}
        k = 1
        while k**p < n:
            expected.add(floor(k**p))
            k += 1
        k = 1
        stride = n ** (1 - 1 / p) if n else 1.0
        while k * stride < n:
            expected.add(floor(k * stride))
            k += 1
        assert result == sorted(expected), (n, p)

src = Path(__file__).with_name('cms.nim').read_text()
assert 'let stride = pow(nf, 1.0 - 1.0/exponent)' in src
assert 'result.sort()' in src
assert 'result.setLen(uniqueLen)' in src
assert 'levels = diffusionLevels(original.len, exponent)' in src
for t in ('cfg.coarsePower', 'cfg.finePower', 'cfg.evalPower'):
    assert t in src
assert 'let doEvaluate = evaluationDue(step, cfg.steps, cfg.evalEvery)' in src
assert 'appendDiffusion(sampledDiffusion, hashed[0], 0, tinyCfg, codec, power)' in src
print('PASS: 100 formula cases, exact schedules, common powering, distinct grid coverage, static wiring')
