"""Standalone SOURCE/logic regression checks; does not compile or run Nim."""
from pathlib import Path
import random
src = Path('main.nim').read_text()
old_replace_api = 'proc replaceSeqCompiled*('
assert old_replace_api in Path('replace.nim').read_text()
assert 'when compiles(replaceSeqCompiledInPlace(state, pat, replacement)):' in src
assert 'let transformed = replaceSeqCompiled(state, pat, replacement)' in src
assert 'if transformed.changed and transformed.data != state:' in src
print('PASS: old replace.nim API is accommodated with true-change fallback')
assert 'if dominates(trial.score, current.score):' in src
assert 'if better(trial.score, current.score):' not in src

def dominates(a, b):
    return a[0]>=b[0] and a[1]>=b[1] and (a[0]>b[0] or a[1]>b[1])

for new, old, expected in [
    ((0.3, 0.4), (0.2,0.3), True),
    ((0.3, 0.4), (0.2,0.4), True),
    ((0.3, 0.4), (0.3,0.3), True),
    ((0.3, 0.4), (0.3,0.4), False),
    ((0.4, 0.3), (0.3,0.4), False),
    ((0.1, 0.9), (0.3,0.4), False),
    ((0.2, 0.1), (0.3,0.4), False),
]:
    assert dominates(new,old) == expected
print('PASS: Pareto gate logical cases, including tradeoff rejection and equality')

assert 'diffusionSteps: 100' in src
assert 'for k in 0..<cfg.diffusionSteps:' in src
assert 'if k > 0 and k mod clean.len == 0:' in src
assert 'let p = order[k mod clean.len]' in src
assert 'result.y.add(1.0 - float64(incorrect)/float64(clean.len))' in src
assert 'diffusionRate' not in src
assert '"diffusionSteps": cfg.diffusionSteps' in src
print('PASS: fixed 100-step diffusion uses modulo sweeps and actual mismatch labels')

# Reference simulation of the implemented index/label bookkeeping. This checks
# the logic for short and long snippets, but is NOT a compiled Nim test.
def trajectory(clean, steps):
    rng = random.Random(10)
    state = clean[:]
    order = list(range(len(clean)))
    rng.shuffle(order)
    incorrect = 0
    out = [(state[:], 1.0)]
    for k in range(steps):
        if k>0 and k % len(clean)==0:
            rng.shuffle(order)
        p=order[k % len(clean)]
        before=state[p]
        choices = [0,1,2,3]
        choices.remove(before)
        state[p] = rng.choice(choices)
        if before == clean[p] and state[p]!=clean[p]:
            incorrect+=1
        elif before!=clean[p] and state[p]==clean[p]:
            incorrect-=1
        out.append((state[:],1.0-incorrect/len(clean)))
    return out

for n in (1,2,7,16,80,100,128,256):
    seq = trajectory([0]*n,100)
    assert len(seq)==101
    for k,(state,y) in enumerate(seq):
        assert abs(y - sum(v==0 for v in state)/n)<1e-12, (n,k)
        if k>0:
            assert sum(a!=b for a,b in zip(state,seq[k-1][0]))==1, (n,k)
print('PASS: reference corruption-label bookkeeping for lengths 1,2,7,16,80,100,128,256')
print('NOTICE: Nim compiler unavailable; compiled binary and gnuplot output not tested')
