"""Static protocol/invariant audit for expanded operators, not Nim execution."""
from pathlib import Path
import re
from collections import Counter
p = Path(__file__).with_name('cms.nim')
s = p.read_text()
assert 'MutationOps = 32' in s and 'VirusActions = 7 + MutationOps + 5' in s
raw = re.search(r'MutationNames: array\[MutationOps, string\] = \[(.*?)\]', s, re.S)
assert raw
names = re.findall(r'"([a-z-]+)"', raw.group(1))
assert len(names) == 32 and len(set(names)) == len(names), (len(names), Counter(names))
body = s[s.index('proc editGene('):s.index('proc mutateGene(')]
assert sorted(map(int,re.findall(r'\n    of (\d+):',body))) == list(range(32))
for token in ['normalizeWildcards(result)','validGene(result,index)','not geneEqual(result,parent)',
              'RuleMaxLen','shallowSegment','randomLiteral','newGene']:
    assert token in body, token
assert 'doAssert edited == 1+rep' in s
assert 'for op in 0..<MutationOps' in s
assert 'let methodId = r.randInt(8)' in s
assert 'proc diff3Crossover(' in s
assert 'mutationCount(r, cfg.rules)' in s
assert 'mutateExactly(control, source, r, k)' in s
assert 'copyGene(g.genes[r.randInt(at)])' in s
assert 'if action in 7..<(7+MutationOps)' in s
assert 'mostFrequentNgram(ctx,at' in s
assert 'trained.looPearson' in s
print('PASS: 32 unique edits / eight crossing modes / viral dispatch / constraints / exact-K baseline')
for i, name in enumerate(names):
    print(f'  {i:02d} {name}')

# Regression checks for the two user-reported Nim compiler errors.
import re
assert not re.search(r'\b(?:let|var|const)\s+block\b', s), 'block is a Nim keyword'
assert not re.search(r'\bif\b[^\n]*:[ \t]*(?:let|var|const)[ \t]+', s), 'declaration after one-line if'
cross = s[s.index('proc twoParentCrossover('):s.index('proc diff3Crossover(')]
assert 'mother, father: Genome' in cross
assert 'let blockSize' in cross and 'father.genes[locus]' in cross
assert 'let firstPos = r.randInt(n)' in s and 'let secondPos = r.randInt(n)' in s
print('PASS: Nim keyword/declaration/crossover-parent-name regression audit')

# Regression: Nim seq/array two-variable `for` means (index, element),
# not destructuring of a tuple element. The prior code used label as an int.
assert 'for label, dag in [("target",v.targetDag)' not in s
assert 'for dagIndex, dag in [v.targetDag, v.partnerDag, v.gateDag]:' in s
assert 'let label = ["target", "partner", "gate"][dagIndex]' in s
print('PASS: Nim array index/value iteration used correctly for virus description')
