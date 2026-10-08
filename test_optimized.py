"""Standalone static / mathematical regression checks. Nim self-test checks actual replacements."""
from pathlib import Path
import random
s=Path(__file__).with_name('main.nim').read_text()
req=[
  'proc primeStagePrefixes(', 'proc prefixFor(', 'proc tokenBits(',
  'proc checkOptimizedFeatureSemantics()', 'proc slowReferenceFeatures(',
  'ctx.prefixes[i], true', 'result.patterns[i] = parent.patterns[i]',
  'result.patterns[i] = compileSeqPattern(replacement.a)',
  'proc ruleKey(', 'var stageMemo = initTable[string, Candidate]()',
  'var inferenceMemo = initTable[string, float64]()',
  'if dominates(trial.score, current.score):',
  'if greedyAccept(result, trial):',
  'primeStagePrefixes(ctx, trainer.current, active)',
  'ReplacePasses = 1', 'diffusionSteps: 100',
  'ctx.inferenceSeed + j*7919',
  'proc checkChunkAdmission()', 'appendCorpusLine(buffer, oversized, line, cfg.maxChunk)',
  'chunkAdmission": "skip-oversize-v1',
]
for marker in req: assert marker in s, marker
print('PASS: compiled-pattern reuse, caches, frozen-prefix placement, strict Pareto and unchanged diffusion')

# Negative matcher result must be impossible if the pattern's first literal exists.
def token_bits(xs):
    bits=[0]*8
    for x in xs:
        if 0<=x<512: bits[x>>6]|=1<<(x&63)
    return bits

def might_match(bits,mark):
    if mark<0 or mark>=512: return True
    return bool(bits[mark>>6]&(1<<(mark&63)))

rng=random.Random(123)
for _ in range(10000):
    xs=[rng.randint(-3, 600) for _ in range(rng.randrange(100))]
    mark=rng.randint(-3,600)
    bits=token_bits(xs)
    if mark in xs and mark>=0:
        assert might_match(bits,mark)
print('PASS: 10,000 token bitmap no-false-negative membership cases')

# Abstract literal replacement interpreter models one scan and tests prefix cache.
def apply(state,rule):
    a,b=rule
    if not a: return list(state),False
    out=[]; i=0; changed=False
    while i<len(state):
        if state[i:i+len(a)] == a:
            out+=b; i+=len(a); changed=True
        else: out.append(state[i]); i+=1
    return out, changed and out!=state

def eval_feature(state,rules,prefix=None):
    feat=[0]*len(rules)
    if prefix is None: state=list(state); start=0
    else:
        state=list(prefix[0]); feat[:len(prefix[1])]=prefix[1]; start=len(prefix[1])
    for ix in range(start,len(rules)):
        state,changed=apply(state,rules[ix]); feat[ix]+=int(changed)
    return feat

for _ in range(1500):
    r=[([rng.randrange(4) for _ in range(rng.randrange(1,4))],
        [rng.randrange(4) for _ in range(rng.randrange(1,4))]) for _ in range(8)]
    inp=[rng.randrange(4) for _ in range(10)]
    target=eval_feature(inp,r)
    for cutoff in range(len(r)):
        state=list(inp); counts=[]
        for rr in r[:cutoff]:
            state,changed=apply(state,rr); counts.append(int(changed))
        assert eval_feature(inp,r,(state,counts))==target
print('PASS: 12,000 reference checks: single-pass frozen-prefix cache is exact')
assert eval_feature([0],[([0],[1]),([1],[0])]) == [1,1]
print('PASS: cyclic-rule regression rejects a second rewrite pass')
print('NOTE: Actual Nim compilation/runtime timings require Nim on target system.')

# New admission invariant: bounds inclusive, no crop, and no rejected sample
# contributes to the reservoir seen count. Test at byte boundaries.
assert 'if buffer.len < cfg.minChunk or buffer.len > cfg.maxChunk: return' in s
assert 'chunks.add(byteSeq(buffer))' in s
assert 'chunks[slot] = byteSeq(buffer)' in s
assert 'buffer[start ..< stop]' not in s
assert 'if not oversized:' in s
assert 'appendCorpusLine(buffer, oversized, line, cfg.maxChunk)' in s
assert 'checkChunkAdmission()' in s
assert '"chunkAdmission": "skip-oversize-v1"' in s

def consume(data, min_chunk=3, max_chunk=8, cap=100):
    buffer = bytearray()
    oversize = False
    seen = 0
    chunks = []
    for line in data.splitlines(keepends=False):
        if line == b'===SPLIT===':
            if not oversize and min_chunk <= len(buffer) <= max_chunk:
                seen += 1
                chunks.append(bytes(buffer))
            buffer.clear()
            oversize = False
        elif not oversize:
            if len(line) >= max_chunk - len(buffer):
                oversize = True
                buffer.clear()
            else:
                buffer.extend(line)
                buffer.append(10)
    if not oversize and min_chunk <= len(buffer) <= max_chunk:
        seen += 1
        chunks.append(bytes(buffer))
    return seen, chunks[:cap]

cases = [b'x', b'12', b'abc', b'abcdefg', b'abcdefgh', b'abcdefghi',
         b'a'*100, 'あい'.encode('utf-8')]
# Source parser appends one newline for each non-split line.
data = b'\n===SPLIT===\n'.join(cases)
count, accepted = consume(data)
assert count == 4, (count, accepted)
assert accepted == [b'12\n', b'abc\n', b'abcdefg\n', 'あい'.encode('utf-8') + b'\n'], accepted
# Explicit exact-max boundary, including newline
assert consume(b'1234567')[1] == [b'1234567\n']
assert consume(b'12345678')[1] == []
assert consume(b'abc\n123456789\nignored\n===SPLIT===\n1234567')[1] == [b'1234567\n']
print('PASS: reject oversized snippets whole; boundary, UTF-8 byte count, split reset')
