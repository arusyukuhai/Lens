import random
import main as m
B=m._GP_INDEX_TOKENS

def perm():
    x=list(range(m.VOCAB)); random.shuffle(x); return x

# translated cached key remains the corresponding contiguous necessary ngram
for _ in range(100000):
    tr=perm()
    if random.random()<0.6:
        n=random.randint(3,12)
        pat=[random.randrange(m.VOCAB) for _ in range(n)]
        q=random.randrange(n-2)
        a,b,c=pat[q:q+3]
        key=(a*B+b)*B+c
        r=m.Rule(pat,[pat[0]], pack_filter_key=key, pack_anchor_offset=q)
        rr=m.recode_rule(r,tr)
        a2,b2,c2=rr.pattern[q:q+3]
        exp=(a2*B+b2)*B+c2
        assert rr.pack_filter_key==exp
        assert rr.pack_anchor_offset==q
    else:
        # wildcard pattern => pair guard
        n=random.randint(2,12)
        pat=[random.randrange(m.VOCAB) for _ in range(n)]
        pat[random.randrange(n)] = -1
        pairs=[q for q in range(n-1) if pat[q]>=0 and pat[q+1]>=0]
        if not pairs: continue
        q=random.choice(pairs); a,b=pat[q],pat[q+1]
        key=a*B+b
        r=m.Rule(pat,[-1], pack_filter_key=key, pack_anchor_offset=q)
        rr=m.recode_rule(r,tr)
        a2,b2=rr.pattern[q],rr.pattern[q+1]
        assert rr.pack_filter_key==a2*B+b2
        assert rr.pack_anchor_offset==q

# sentinels preserved
tr=perm()
for k in (-2,-1):
    r=m.Rule([1,2,3],[1],pack_filter_key=k,pack_anchor_offset=1)
    rr=m.recode_rule(r,tr)
    assert rr.pack_filter_key==k and rr.pack_anchor_offset==1
print('PASS: 100k translated memo contracts + sentinels')
