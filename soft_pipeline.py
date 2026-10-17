"""Bounded, mean-field C2 replacement layers for fully uncertain byte tapes.

Not the exact pushforward distribution. Approximations: independent delimiter
hazards, marginal captures, expected-position packing, and per-layer length
coordinate reconstruction. At hard inputs/rules, matches within the declared
capture bound use the corrected discrete semantics. Trailing/adjacent/leading
wildcards are deliberately unsupported in this fast experimental version.
"""
import ctypes
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import subprocess
import tempfile
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from replace_exact import smoothstep5, normalized_entropy

_LIBRARY=None

def _library():
    global _LIBRARY
    if _LIBRARY is None:
        source=Path(__file__).with_name('scan_cpu.c')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        folder=Path(tempfile.gettempdir())/'replace_scan'
        folder.mkdir(exist_ok=True)
        binary=folder/f'scan_{digest}.so'
        if not binary.exists():
            subprocess.run(['cc','-O3','-shared','-fPIC',str(source),'-o',str(binary)],check=True)
        _LIBRARY=ctypes.CDLL(str(binary))
        for name in ('scan32','scan64'):
            fn=getattr(_LIBRARY,name)
            fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int64]*3+[ctypes.c_int]
            fn.restype=None
    return _LIBRARY


def _future(x, d):
    if d >= x.shape[1]:
        return x*0
    return F.pad(x[:,d:],(0,d))

class _Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx,w,s,trans):
        if w.device.type!='cpu' or w.dtype not in (torch.float32,torch.float64):
            raise ValueError('native scan currently supports CPU float32/float64 only')
        w,s=w.contiguous(),s.contiguous()
        r=torch.empty_like(s)
        fn=_library().scan64 if w.dtype==torch.float64 else _library().scan32
        fn(w.data_ptr(),s.data_ptr(),r.data_ptr(),*w.shape,int(trans))
        ctx.save_for_backward(w,r)
        ctx.trans=trans
        return r

    @staticmethod
    def backward(ctx,grad):
        w,r=ctx.saved_tensors
        adj=_Solve.apply(w,grad,not ctx.trans)
        # Calling the same differentiable solver on the transpose implements
        # the implicit adjoint. Saved output r retains its autograd connection,
        # so this backward also supports differentiation a second time.
        if ctx.trans:
            gw=torch.stack([adj*_future(r,d+1) for d in range(w.shape[2])],-1)
        else:
            gw=torch.stack([r*_future(adj,d+1) for d in range(w.shape[2])],-1)
        return gw,adj,None


def reach_scan(edges):
    source=edges.new_zeros(edges.shape[:2])
    source[:,0]=1
    return _Solve.apply(edges,source,False)


def shift(x, offset):
    """y[:,i]=x[:,i+offset], zero beyond capacity."""
    L=x.shape[1]
    if offset>=L:
        return x*0
    if offset==0:
        return x
    tail=x.new_zeros((x.shape[0],offset)+x.shape[2:])
    return torch.cat((x[:,offset:],tail),dim=1)


@dataclass(frozen=True)
class Rule:
    pattern: tuple
    replacement: tuple

@dataclass(frozen=True)
class Event:
    rule: int
    span: int
    literals: tuple  # (offset,byte)
    exclusions: tuple # tuples of (offset,byte) for earlier delimiter blocks
    output: tuple  # ('copy',offset) or ('literal',byte)


def compile_events(rules,min_capture,max_capture):
    events=[]
    if min_capture<0 or max_capture<min_capture:
        raise ValueError('invalid capture bound')
    for ri,rule in enumerate(rules):
        a,b=rule.pattern,rule.replacement
        if not a or a[0]<0 or a[-1]<0 or len(b)>64:
            raise ValueError('fast patterns need literal prefix/suffix; replacement <=64')
        if any(c>=256 for c in a) or any(not -15<=c<256 for c in b):
            raise ValueError('invalid byte/capture opcode')
        parts=[[]]
        for c in a:
            if c<0:
                if not parts[-1]:
                    raise ValueError('adjacent wildcards unsupported')
                parts.append([])
            else: parts[-1].append(c)
        n=len(parts)-1
        if n>3:
            raise ValueError('fast experimental implementation supports up to 3 captures')
        for gaps in product(range(min_capture,max_capture+1),repeat=n):
            off=0; literals=[]; excludes=[]; caps=[]
            literals.extend((j,c) for j,c in enumerate(parts[0])); off=len(parts[0])
            for w,gap in enumerate(gaps):
                start=off
                for earlier in range(min_capture,gap):
                    excludes.append(tuple((start+earlier+j,c) for j,c in enumerate(parts[w+1])))
                caps.append(tuple(range(off,off+gap)))
                off+=gap
                literals.extend((off+j,c) for j,c in enumerate(parts[w+1]))
                off+=len(parts[w+1])
            output=[]
            for c in b:
                if c>=0: output.append(('literal',c))
                elif caps:
                    r=-c-1
                    output.extend(('copy',j) for j in caps[r if r<n else 0])
            events.append(Event(ri,off,tuple(literals),tuple(excludes),tuple(output)))
    return tuple(events)


class SoftReplaceLayer(nn.Module):
    """Learn one rule from a declared finite candidate family; stack for many rules.

    Input/output: dense [B,L,256] distributions and [B] continuous length.
    All input byte probabilities participate in gradients. Rule logits are over
    complete candidates (joint rule distribution), NOT arbitrary independent
    token logits. Fixed candidates and finite capture bounds are explicit limits.
    """
    def __init__(self,rules,*,min_capture=0,max_capture=4,dtype=torch.float32,match_power=2,trainable_literals=False,pattern_prior=10.0):
        super().__init__()
        self.rules=tuple(Rule(tuple(r.pattern),tuple(r.replacement)) for r in rules)
        if not self.rules: raise ValueError('need candidates')
        self.events=compile_events(self.rules,min_capture,max_capture)
        if len(self.events)>512: raise ValueError('more than 512 local events')
        if match_power not in (1,2):
            raise ValueError('match_power must be 1 (raw) or 2 (C2 confidence sharpening)')
        self.match_power=match_power
        self.min_capture,self.max_capture=min_capture,max_capture
        self.max_span=max(e.span for e in self.events)
        self.max_ratio=max(1.,max(len(e.output)/e.span for e in self.events))
        self.logits=nn.Parameter(torch.zeros(len(rules),dtype=dtype))
        self.trainable_literals=trainable_literals
        self.pattern_roles={};self.output_roles={}
        self.pattern_logits=nn.ParameterList();self.output_logits=nn.ParameterList()
        if trainable_literals:
            for ri,r in enumerate(self.rules):
                for c in sorted(set(v for v in r.pattern if v>=0)):
                    self.pattern_roles[(ri,c)]=len(self.pattern_logits)
                    z=torch.zeros(256,dtype=dtype);z[c]=pattern_prior
                    self.pattern_logits.append(nn.Parameter(z))
                for c in sorted(set(v for v in r.replacement if v>=0)):
                    self.output_roles[(ri,c)]=len(self.output_logits)
                    # Replacement byte values are NOT supplied as initialization.
                    self.output_logits.append(nn.Parameter(torch.zeros(256,dtype=dtype)))

    def forward(self,X,length,output_capacity=None):
        B,L,C=X.shape
        if C!=256 or length.shape!=(B,) or L==0:
            raise ValueError('expected nonempty [B,L,256] and [B] length')
        if X.device.type!='cpu' or X.dtype!=self.logits.dtype or length.dtype!=X.dtype:
            raise ValueError('matching CPU float32/64 tensors required')
        K=output_capacity if output_capacity is not None else int(__import__('math').ceil(L*self.max_ratio))
        if K<int(__import__('math').ceil(L*self.max_ratio)):
            raise ValueError('output capacity below conservative no-truncation bound')
        g=smoothstep5(length[:,None]-torch.arange(L,dtype=X.dtype,device=X.device))
        pi=self.logits.softmax(-1)
        # Reuse every shifted scalar character probability across all events.
        # Optional smooth confidence map, applied ONLY to delimiter matching.
        # This deliberately changes the soft extension; it is not Bayesian
        # evidence or exact probability propagation. Hard vertices are fixed.
        match_X=X if self.match_power==1 else X.square()/X.square().sum(-1,keepdim=True)
        shifts={}
        pattern_probs=[(z*self.match_power).softmax(-1) for z in self.pattern_logits]
        output_probs=[z.softmax(-1) for z in self.output_logits]
        base_matches={}
        def char(offset,c,ri):
            role=self.pattern_roles[(ri,c)] if self.trainable_literals else c
            key=(offset,role)
            if key not in shifts:
                if role not in base_matches:
                    base_matches[role]=(match_X@pattern_probs[role] if self.trainable_literals else match_X[:,:,c])
                shifts[key]=shift(base_matches[role],offset)
            return shifts[key]
        qs=[]
        for e in self.events:
            q=pi[e.rule].expand(B,L)
            for offset,c in e.literals: q=q*char(offset,c,e.rule)
            for block in e.exclusions:
                earlier=X.new_ones((B,L))
                for offset,c in block: earlier=earlier*char(offset,c,e.rule)
                q=q*(1-earlier)
            q=q*shift(g,e.span-1)
            qs.append(q)
        q=torch.stack(qs,-1)
        # Sum <=1 mathematically: per rule, first-hit hazards form a subprobability.
        no_match=1-q.sum(-1)
        edges=[]
        for d in range(1,self.max_span+1):
            ids=[j for j,e in enumerate(self.events) if e.span==d]
            value=q[:,:,ids].sum(-1) if ids else no_match*0
            if d==1: value=value+no_match
            edges.append(value)
        reach=reach_scan(torch.stack(edges,-1))
        weights=reach[:,:,None]*q
        copy=reach*no_match*g
        emitted=copy+sum(weights[:,:,j]*len(e.output) for j,e in enumerate(self.events))
        starts=emitted.cumsum(1)-emitted
        out_length=emitted.sum(1)
        data=X.new_zeros((B,K,256)); mass=X.new_zeros((B,K))
        def deposit(tokens,weight):
            nonlocal data,mass
            n=tokens.shape[2]
            if n==0: return
            coords=starts[:,:,None]+emitted[:,:,None]*(torch.arange(n,dtype=X.dtype)+.5)/n-.5
            left=coords.floor()
            frac=coords-left
            t=smoothstep5(frac)
            for side,blend in ((0,1-t),(1,t)):
                index=(left.long()+side).clamp(0,K-1).reshape(B,-1)
                w=(weight[:,:,None]*blend).reshape(B,-1)
                mass=mass.scatter_add(1,index,w)
                v=(tokens*(weight[:,:,None]*blend)[:,:,:,None]).reshape(B,-1,256)
                data=data.scatter_add(1,index[:,:,None].expand(-1,-1,256),v)
        deposit(X[:,:,None,:],copy)
        copied={0:X}
        for j,e in enumerate(self.events):
            if not e.output: continue
            tokens=[]
            for kind,value in e.output:
                if kind=='copy':
                    if value not in copied: copied[value]=shift(X,value)
                    tokens.append(copied[value])
                else:
                    v=(output_probs[self.output_roles[(e.rule,value)]] if self.trainable_literals
                       else F.one_hot(torch.tensor(value),256).to(X.dtype))
                    tokens.append(v.expand(B,L,256))
            deposit(torch.stack(tokens,2),weights[:,:,j])
        # Smooth positive denominator; at hard occupied cells mass=1, no bias.
        # At empty cells uniform padding. Fixed regularization, not detached clip.
        padding=1e-4*(1-mass).square()
        Y=(data+padding[:,:,None]/256)/(mass+padding)[:,:,None]
        return Y,out_length

    def regularizer(self):
        terms=[normalized_entropy(self.logits)]
        terms.extend(normalized_entropy(z) for z in self.pattern_logits)
        terms.extend(normalized_entropy(z) for z in self.output_logits)
        return torch.stack(terms).mean()

    @torch.no_grad()
    def decode(self):
        ri=int(self.logits.argmax());r=self.rules[ri]
        if not self.trainable_literals:return r
        a=tuple(c if c<0 else int(self.pattern_logits[self.pattern_roles[(ri,c)]].argmax()) for c in r.pattern)
        b=tuple(c if c<0 else int(self.output_logits[self.output_roles[(ri,c)]].argmax()) for c in r.replacement)
        return Rule(a,b)


class SoftReplacePipeline(nn.Module):
    def __init__(self,layers):
        super().__init__(); self.layers=nn.ModuleList(layers)
    def forward(self,X,length):
        for layer in self.layers: X,length=layer(X,length)
        return X,length
    def regularizer(self): return torch.stack([l.regularizer() for l in self.layers]).mean()
    def decode(self): return [l.decode() for l in self.layers]
