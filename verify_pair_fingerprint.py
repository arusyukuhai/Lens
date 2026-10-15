"""Independent Python check for Nim's lossless token+adjacent-pair prefilter."""
import random
MASK=(1<<64)-1
SHIFT=51
W=128

def token_bit(t):return ((t*0x9E3779B97F4A7C15)&MASK)>>SHIFT

def pair_bit(a,b):
    h=((a*0x9E3779B97F4A7C15)&MASK)^((b*0xBF58476D1CE4E5B9)&MASK)
    h=((h^(h>>30))*0xBF58476D1CE4E5B9)&MASK
    h=((h^(h>>27))*0x94D049BB133111EB)&MASK
    return (h^(h>>31))>>SHIFT

def fingerprint(text):
    singles=set(token_bit(t) for t in text if t>=0)
    pairs=set(pair_bit(a,b) for a,b in zip(text,text[1:]) if a>=0 and b>=0)
    return singles,pairs

def may_contain(fp, pat, check_pairs=True):
    return all(token_bit(t) in fp[0] and
               (not check_pairs or i==0 or pat[i-1]<0 or pair_bit(pat[i-1],t) in fp[1])
               for i,t in enumerate(pat) if t>=0)

r=random.Random(1729)
num=0
for _ in range(20000):
    text=[r.randrange(1, 1000000) for _ in range(r.randrange(2,100))]
    start=r.randrange(len(text))
    end=r.randrange(start+1,len(text)+1)
    # Construct a guaranteed real match by removing positive-length spans
    # and replacing each with one wildcard. Adjacent literals are kept as-is.
    pat=[]
    i=start
    while i<end:
        if r.random()<.23 and i<end-1:
            step=r.randrange(1,min(8,end-i)+1)
            pat.append(-1);i+=step
        else:
            pat.append(text[i]);i+=1
    fp=fingerprint(text)
    assert may_contain(fp,pat), (text,pat)
    num+=1
# Prove extra rejection over single-token presence when order/adjacency fails.
a,b,c=100001,100003,100002
fp=fingerprint([a,b,c])
assert may_contain(fp,[a,c],False)
assert not may_contain(fp,[a,c],True)
assert may_contain(fp,[a,-1,c],True)
print(f'PASS: {num:,} constructed true matches all retained; exact pair order rejection works.')
