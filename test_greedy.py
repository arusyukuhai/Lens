"""Reference tests for greedy semantics, stage coordinates, and output schema."""
import math
from pathlib import Path
src=Path('main.nim').read_text()

def dominates(a,b):
    return a[0]>=b[0] and a[1]>=b[1] and (a[0]>b[0] or a[1]>b[1])

def greedy_reduce(start, trials):
    incumbent=start
    for t in trials:
        if dominates(t,incumbent): incumbent=t
    return incumbent

cases=[
    ((0.7,0.1),(0.8,0.1),True),
    ((0.7,0.1),(0.7,0.2),True),
    ((0.7,0.1),(0.8,0.2),True),
    ((0.7,0.1),(0.8,0.09),False),
    ((0.7,0.1),(0.69,0.4),False),
    ((0.7,0.1),(0.7,0.1),False),
    ((0.7,0.1),(0.6,0.01),False),
]
for baseline,proposal,expected in cases:
    assert dominates(proposal, baseline)==expected
assert greedy_reduce((0.3,0.4),[(0.35,0.38),(0.4,0.4),(0.4,0.4),(0.39,0.7),(0.42,0.4),(0.42,0.5)])==(0.42,0.5)
print('PASS: reference strict Pareto greedy selection rejects all regressions, ties and tradeoffs')

def stage_coordinates(step):
    outer=max(0,int((math.sqrt(8*step+1)-1)*0.5))
    while outer*(outer+1)//2>step: outer-=1
    while (outer+1)*(outer+2)//2<=step: outer+=1
    offset=step-outer*(outer+1)//2
    return (outer,outer-offset)
assert [stage_coordinates(i)[1]+1 for i in range(10)]==[1,2,1,3,2,1,4,3,2,1]
print('PASS: one active rule per triangular stage')
assert 'if greedyAccept(result, trial):' in src
assert 'if dominates(trial.score, current.score):' in src
assert 'root["beams"].add' not in src
assert 'chooseBeam(' not in src
assert 'root["beams"][0]' in src
assert 'greedy_checkpoint.json' in src
print('PASS: source-level single incumbent, one DE population, and legacy resume checks')
