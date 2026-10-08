"""Optional Apple Metal backend for byte-level recurrent Replacer.

Each Metal thread owns one (genome, example) rollout. State memory is allocated
from the actual longest example in the current batch; no fixed context window,
clipping or hidden-memory size limit is introduced. The CPU reference in main.py
is the semantic authority; this kernel mirrors its ordered one-sweep rules.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np

try:
    import torch
except ImportError:
    torch = None

_METAL_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

inline int mod_byte(int v) { return v & 255; }

// Match a rule against one location; literal runs are greedy/shortest.
inline bool match_at(const device short *s, int n, int start,
                     const device short *pat, int pn,
                     thread int *cs, thread int *cl, thread int &finish) {
    int p=0, pos=start, c=0;
    while (p<pn && pat[p]>=0) {
        if (pos>=n || s[pos]!=pat[p]) return false;
        ++p; ++pos;
    }
    while (p<pn) {
        if (pat[p]!=-1 || c>=16) return false;
        ++p;
        int b=pos, q=p;
        while (p<pn && pat[p]>=0) ++p;
        int ll=p-q;
        cs[c]=b;
        if (!ll) {
            if (p==pn) { cl[c]=n-pos; pos=n; }
            else cl[c]=0;
        } else {
            int hit=-1;
            for (int k=pos;k+ll<=n;++k) {
                bool ok=true;
                for(int a=0;a<ll;++a) if(s[k+a]!=pat[q+a]) { ok=false; break; }
                if(ok) { hit=k; break; }
            }
            if(hit<0) return false;
            cl[c]=hit-pos;
            pos=hit+ll;
        }
        ++c;
    }
    finish=pos;
    return pos>start;
}

inline int transformed(int v, int kind, const device int* lut, const device int* inverse) {
    if(kind==0 || kind==1) return v;
    int z=int(lut[v]);
    if(kind==3) z=mod_byte(z+1);
    else if(kind==4) z=mod_byte(z-1);
    else if(kind==5) z=mod_byte(z*2);
    else if(kind==6) z=z/2;
    return int(inverse[z]);
}

// One ordered sweep. Output never grows because host validates every rule.
// Each rule may match several nonoverlapping segments, in original order.
inline void sweep(const device short* pats, const device short* reps,
                  const device int* po, const device int* ro,
                  const device int* lut, const device int* inv,
                  const device int* anchors,
                  device short* a, device short* b, thread int &n,
                  int rules, int capacity, thread bool &on_a) {
    // Bloom filters are strictly prefilters: false positives are allowed,
    // but a true match can never be skipped. One bitset refresh per state change.
    // Indexed rule anchors are chosen on the host using the training corpus.
    thread uint byte_mask[8];
    thread uint pair_bloom[64];
    bool dirty=true;
    for (int r=0;r<rules;++r) {
        int anchor=anchors[r];
        if (anchor==-1) continue;  // provably identity rule
        int p0=po[r],pn=po[r+1]-p0;
        int q0=ro[r],qn=ro[r+1]-q0;
        if(pn==0 || n==0) continue;
        if (dirty) {
            for (int i=0;i<8;++i) byte_mask[i]=0u;
            for (int i=0;i<64;++i) pair_bloom[i]=0u;
            const device short* current=on_a?a:b;
            for (int i=0;i<n;++i) {
                uint v=uint(current[i]);
                byte_mask[v>>5] |= (1u << (v&31u));
                if(i+1<n) {
                    uint key=(v<<8) | uint(current[i+1]);
                    uint h1=(key*2654435761u)&2047u;
                    uint h2=((key*2246822519u)^(key>>7))&2047u;
                    pair_bloom[h1>>5] |= (1u<<(h1&31u));
                    pair_bloom[h2>>5] |= (1u<<(h2&31u));
                }
            }
            dirty=false;
        }
        if(anchor<256) {
            uint v=uint(anchor);
            if((byte_mask[v>>5] & (1u<<(v&31u)))==0u) continue;
        } else {
            uint key=uint(anchor-256);
            uint h1=(key*2654435761u)&2047u;
            uint h2=((key*2246822519u)^(key>>7))&2047u;
            if((pair_bloom[h1>>5] & (1u<<(h1&31u)))==0u ||
               (pair_bloom[h2>>5] & (1u<<(h2&31u)))==0u) continue;
        }
        const device short* pat=pats+p0;
        const device short* rep=reps+q0;
        device short* src=on_a?a:b;
        device short* dst=on_a?b:a;
        int scan=0, prev=0, out=0, wc=0;
        for(int v=0;v<pn;++v) if(pat[v]<0) ++wc;
        bool matched=false, overflow=false;
        while(scan<n) {
            int ss=-1, finish=-1, cs[16], cl[16];
            if(pat[0]>=0) {
                for(int k=scan;k<n;++k) {
                    if(src[k]!=pat[0]) continue;
                    int f=0;
                    if(match_at(src,n,k,pat,pn,cs,cl,f)) { ss=k; finish=f; break; }
                }
            } else {
                int f=0;
                if(match_at(src,n,scan,pat,pn,cs,cl,f)) { ss=scan; finish=f; }
            }
            if(ss<0) break;
            matched=true;
            for(int j=prev;j<ss;++j) dst[out++]=src[j];
            int emitted=0, allowance=finish-ss;
            for(int j=0;j<qn;++j) {
                int t=rep[j];
                if(t>=0) {
                    if(emitted<allowance) { dst[out++]=short(t); ++emitted; }
                    continue;
                }
                if(wc<=0) continue;
                int kind=0, ci=0;
                if(t>=-15) { kind=0; ci=-t-1; }
                else if(t>=-31) { kind=2; ci=-t-16; }
                else if(t>=-47) { kind=1; ci=-t-32; }
                else if(t>=-111) { int off=-t-48; kind=3+off/16; ci=off%16; }
                else continue;
                if(ci>=wc) ci=0;
                int begin=cs[ci], count=cl[ci];
                if(emitted+count>allowance) continue; // same atomic capture policy as CPU
                if(kind==2) {
                    // Counting sort on the 256-valued permutation keys.
                    // This is O(count+256), unlike the old quadratic rank sort.
                    if(count==1) { dst[out]=src[begin]; }
                    else if(count>1) {
                        thread uint hist[256];
                        for(int z=0;z<256;++z) hist[z]=0;
                        for(int j=0;j<count;++j) {
                            uint k=uint(lut[int(src[begin+j])]);
                            ++hist[k];
                        }
                        int cursor=out;
                        for(int z=0;z<256;++z) {
                            int howmany=int(hist[z]);
                            short value=short(inv[z]);
                            for(int j=0;j<howmany;++j) dst[cursor++]=value;
                        }
                    }
                    out+=count; emitted+=count;
                } else {
                    for(int j=0;j<count;++j) {
                        int pos=(kind==1)?begin+count-j-1:begin+j;
                        dst[out++]=short(transformed(int(src[pos]),kind,lut,inv));
                    }
                    emitted+=count;
                }
            }
            prev=finish;
            scan=finish;
        }
        if(matched) {
            for(int j=prev;j<n;++j) dst[out++]=src[j];
            if(out>capacity || overflow) return; // unreachable for valid nonexpanding rules
            n=out;
            on_a=!on_a;
            dirty=true;  // prior anchors no longer describe the modified state
        }
    }
}

kernel void autoregressive_accuracy(
    const device short* texts [[buffer(0)]],
    const device int* lengths [[buffer(1)]],
    const device short* patterns [[buffer(2)]],
    const device short* replacements [[buffer(3)]],
    const device int* pat_offsets [[buffer(4)]],
    const device int* rep_offsets [[buffer(5)]],
    const device int* luts [[buffer(6)]],
    const device int* inverses [[buffer(7)]],
    device short* state_a [[buffer(8)]],
    device short* state_b [[buffer(9)]],
    device int* scores [[buffer(10)]],
    constant int &sample_count [[buffer(11)]],
    constant int &text_stride [[buffer(12)]],
    constant int &state_stride [[buffer(13)]],
    constant int &rule_count [[buffer(14)]],
    constant int &genome_count [[buffer(15)]],
    const device int* anchors [[buffer(16)]],
    uint tid [[thread_position_in_grid]]) {
    int id=int(tid);
    if(id>=sample_count*genome_count) return;
    int g=id/sample_count, sample=id%sample_count;
    int L=lengths[sample];
    int out_base=id*state_stride;
    const device short* txt=texts+sample*text_stride;
    const device short* pat=patterns+g*(rule_count*65); // padded, 65 slots per rule
    const device short* rep=replacements+g*(rule_count*65);
    const device int* po=pat_offsets+g*(rule_count+1);
    const device int* ro=rep_offsets+g*(rule_count+1);
    const device int* lut=luts+g*256;
    const device int* inv=inverses+g*256;
    device short* a=state_a+out_base;
    device short* b=state_b+out_base;
    if(L<2) { scores[id]=0; return; }
    a[0]=txt[0]; a[1]=0;
    int n=2, correct=0;
    bool on_a=true; // Persistent ping-pong state: eliminate O(prefix) copy each step.
    for(int t=1;t<L;++t) {
        sweep(pat,rep,po,ro,lut,inv,anchors+g*rule_count,
              a,b,n,rule_count,state_stride,on_a);
        device short* s=on_a?a:b;
        if(n==0) { s[0]=0; n=1; }
        if(s[n-1]==txt[t]) ++correct;
        s[n-1]=txt[t];
        s[n]=0;
        ++n;
    }
    scores[id]=correct;
}
'''

_SHADER = None


def _shader():
    global _SHADER
    if torch is None or not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable")
    if _SHADER is None:
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("PyTorch MPS compile_shader is required")
        _SHADER = torch.mps.compile_shader(_METAL_SOURCE)
    return _SHADER


def _pack_texts(texts: Sequence[Sequence[int]], stride: int | None = None) -> np.ndarray:
    """Encode byte strings and integer sequences to the same MPS int16 layout.

    `numpy_array[:] = b"..."` treats bytes as a scalar byte-string, NOT an
    integer sequence.  frombuffer ensures each raw byte becomes one token.
    """
    if stride is None:
        stride = max(1, max((len(s) for s in texts), default=0))
    packed = np.zeros((len(texts), stride), dtype=np.int16)
    for i, row in enumerate(texts):
        if isinstance(row, (bytes, bytearray, memoryview)):
            numbers = np.frombuffer(row, dtype=np.uint8)
        else:
            numbers = np.asarray(row)
            if numbers.ndim != 1 or np.any(numbers < 0) or np.any(numbers > 255):
                raise ValueError('All tokens must be byte values 0..255')
        packed[i, :len(numbers)] = numbers
    return packed


def _compile_rule_anchors(genomes: Sequence, texts: Sequence[Sequence[int]],
                          rule_count: int) -> np.ndarray:
    """Necessary byte/pair preconditions, ordered by corpus rarity.

    -1: rule is always the identity and can be omitted safely.
    0..255: required byte.
    256..65791: required adjacent byte pair (encoded as 256 + 256*a+b).
    The GPU tests byte membership exactly and pair membership by a no-false-negative
    Bloom filter. False positives merely trigger a normal exact match attempt.
    """
    freq_byte=np.zeros(256,dtype=np.int64)
    freq_pair=np.zeros(65536,dtype=np.int64)
    for row in texts:
        vals=(np.frombuffer(row,dtype=np.uint8)
              if isinstance(row,(bytes,bytearray,memoryview))
              else np.asarray(row,dtype=np.uint8))
        freq_byte+=np.bincount(vals.astype(np.intp),minlength=256)
        if len(vals)>1:
            keys=(vals[:-1].astype(np.int32)<<8)|vals[1:].astype(np.int32)
            freq_pair+=np.bincount(keys.astype(np.intp),minlength=65536)
    anchors=np.empty((len(genomes),rule_count),dtype=np.int32)
    for gi,g in enumerate(genomes):
        for ri,r in enumerate(g.rules):
            p,q=r.pattern,r.replacement
            wc=0
            identity=len(p)==len(q) and len(p)>0
            if identity:
                for pv,qv in zip(p,q):
                    if pv>=0:
                        if pv!=qv:
                            identity=False
                            break
                    else:
                        wc+=1
                        if wc>15 or qv!=-wc:
                            identity=False
                            break
            if identity:
                anchors[gi,ri]=-1
                continue
            best=None
            rarity=None
            for a,b in zip(p,p[1:]):
                if a>=0 and b>=0:
                    key=(a<<8)|b
                    f=int(freq_pair[key])
                    if rarity is None or f<rarity:
                        best=256+key;rarity=f
            if best is None:
                for value in p:
                    if value>=0:
                        f=int(freq_byte[value])
                        if rarity is None or f<rarity:
                            best=value;rarity=f
            # is_nonexpanding forces every rule to have a literal.
            if best is None:
                raise ValueError('A valid rule must contain at least one literal')
            anchors[gi,ri]=best
    return anchors


def evaluate_mps(genomes: Sequence, texts: Sequence[Sequence[int]], rule_count: int,
                 batch_size: int = 64, on_batch_done=None) -> np.ndarray:
    """Return [genomes,samples] integer next-byte match counts (no probabilities)."""
    if not genomes:
        return np.zeros((0, len(texts)), dtype=np.int32)
    if not texts:
        return np.zeros((len(genomes), 0), dtype=np.int32)
    if any(len(g.rules) != rule_count for g in genomes):
        raise ValueError("all genomes must have the same rule count")
    import torch
    device = 'mps'
    length = np.asarray([len(s) for s in texts], dtype=np.int32)
    text_stride = max(1, max(map(len, texts)))
    state_stride = text_stride + 1  # prefix and prediction slot; no arbitrary cap
    text_np = _pack_texts(texts, text_stride)
    text_d = torch.from_numpy(text_np).to(device).reshape(-1)
    lengths_d = torch.from_numpy(length).to(device)
    output = np.zeros((len(genomes), len(texts)), dtype=np.int32)
    shader = _shader()
    all_anchors=_compile_rule_anchors(genomes,texts,rule_count)
    per_rule = 65
    for base in range(0, len(genomes), max(1, batch_size)):
        part = genomes[base:base + batch_size]
        G=len(part)
        pat=np.zeros((G,rule_count*per_rule),dtype=np.int16)
        rep=np.zeros_like(pat)
        po=np.zeros((G,rule_count+1),dtype=np.int32)
        ro=np.zeros_like(po)
        luts=np.empty((G,256),dtype=np.int32)
        invs=np.empty_like(luts)
        for gi,g in enumerate(part):
            luts[gi]=np.asarray(g.embedding,dtype=np.int32)
            inv=np.empty(256,dtype=np.int32)
            inv[luts[gi].astype(np.int32)]=np.arange(256,dtype=np.int32)
            invs[gi]=inv
        # Repack each genome into its own 65*R capacity using true cumulative offsets.
        for gi,g in enumerate(part):
            pcur=0; rcur=0
            for ri,r in enumerate(g.rules):
                po[gi,ri]=pcur; ro[gi,ri]=rcur
                pat[gi,pcur:pcur+len(r.pattern)]=r.pattern
                rep[gi,rcur:rcur+len(r.replacement)]=r.replacement
                pcur+=len(r.pattern); rcur+=len(r.replacement)
            po[gi,rule_count]=pcur
            ro[gi,rule_count]=rcur
        a=torch.empty((G*len(texts)*state_stride,),dtype=torch.int16,device=device)
        b=torch.empty_like(a)
        scored=torch.zeros((G*len(texts),),dtype=torch.int32,device=device)
        shader.autoregressive_accuracy(
            text_d,lengths_d,
            torch.from_numpy(pat).to(device).reshape(-1),
            torch.from_numpy(rep).to(device).reshape(-1),
            torch.from_numpy(po).to(device).reshape(-1),
            torch.from_numpy(ro).to(device).reshape(-1),
            torch.from_numpy(luts).to(device).reshape(-1),
            torch.from_numpy(invs).to(device).reshape(-1),
            a,b,scored,len(texts),text_stride,state_stride,rule_count,G,
            torch.from_numpy(all_anchors[base:base+G].copy()).to(device).reshape(-1),
            threads=[((G*len(texts)+63)//64)*64,1,1],group_size=[64,1,1],
        )
        output[base:base+G]=scored.cpu().numpy().reshape(G,len(texts))
        # The blocking D->H transfer above confirms that this entire batch
        # finished. Avoid synchronizing the GPU for each genome solely for UI.
        if on_batch_done is not None:
            on_batch_done(G)
    return output
