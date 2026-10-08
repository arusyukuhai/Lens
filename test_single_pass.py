"""Portable CPU functional and compatibility checks for the single-pass network."""
import random
import numpy as np
import main as m
import gpu_replace_persistent as gpu

random.seed(934);np.random.seed(934)
sampler=m.CorpusSampler([b'abc abc xyz\n',b'hello world\n'])

# Reverse-ordered rules must not be revisited in a single sweep.
g = m.Genome([m.Rule([ord('b')], [ord('c')]), m.Rule([ord('a')], [ord('b')])])
x,base,stats,out=m.trajectory_features_cpu([[ord('a')]],g,100)
assert out==[[ord('b')]],out
assert np.array_equal(x,[[0,1]])
assert stats[0].tolist()==[1,1,0,0],stats
# In normal order, the next rule may still see prior rule's result.
g.rules.reverse()
x,base,stats,out=m.trajectory_features_cpu([[ord('a')]],g,100)
assert out==[[ord('c')]] and np.array_equal(x,[[1,1]]) and stats[0,0]==1
print('PASS: ordered one-sweep semantics')

# Old v64 patterns/opcodes are eliminated in-place on migration.
for a,b in [([65,-2,66],[-128]),([65,-3,66],[-151]),([65,-20,66],[-127]),([65,-1,66],[-111,-1])]:
    g=m.Genome([m.Rule(a[:],b[:])])
    m.stabilize_genome_nonexpanding(g,sampler)
    r=g.rules[0]
    assert all(t>=0 or t==-1 for t in r.pattern),(a,r.pattern)
    assert all(t>=0 or -111<=t<=-1 and not -31<=t<=-16 for t in r.replacement),(b,r.replacement)
    assert m.is_rule_nonexpanding(r)
for _ in range(500):
    r=m.random_rule(sampler,m.default_embedding())
    assert all(t>=0 or t==-1 for t in r.pattern)
    assert all(t>=0 or -111<=t<=-1 for t in r.replacement)
print('PASS: legacy normalization and mutation vocabulary')

# Independent PyTorch CPU replacement helper should agree on plain Replace.
for i in range(60):
    inp=[random.randrange(20) for _ in range(random.randint(3,26))]
    pat=[random.randrange(20) for _ in range(random.randint(1,5))]
    if random.random()<0.5 and len(pat)>1:
        pat[random.randrange(len(pat))]=-1
    wc=sum(x<0 for x in pat)
    b=[random.randrange(20) for _ in range(random.randint(0,sum(x>=0 for x in pat)))]
    if wc and random.random()<0.5: b.append(-1)
    rule=m.Rule(pat,b)
    m.sanitize_rule(rule,sampler)
    out,matched,overflow=m.replace_once_cpu(inp,rule,100)
    gr=gpu.replace_gpu(inp,rule.pattern,rule.replacement,device='cpu')
    if pat.count(-1)==len(pat):
        continue  # helper treats all-wildcard patterns specially
    assert matched == gr.matched,(inp,pat,b,matched,gr)
    assert out == (gr.data.tolist() if gr.applied else inp),(inp,pat,b,out,gr)
    assert not overflow
print('PASS: reference and GPU helper CPU outputs agree on 60 random rules')

# Public helper API rejects deleted wildcard and hidden opcodes.
for bad in (-2, -3, -127):
    try:
        gpu.replace_gpu([65, 66], [65, bad, 66], [65], device="cpu")
        raise AssertionError(f"deleted pattern token was accepted: {bad}")
    except ValueError as exc:
        assert "invalid pattern token" in str(exc)
for bad in (-128, -151):
    try:
        gpu.replace_gpu([65, 66], [65], [bad], device="cpu")
        raise AssertionError(f"deleted replacement token was accepted: {bad}")
    except ValueError as exc:
        assert "invalid replacement opcode" in str(exc)
print("PASS: deleted token families are rejected by the GPU helper")
