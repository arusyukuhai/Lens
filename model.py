"""Serial differentiable Lens; pure PyTorch CPU/CUDA/MPS implementation.
1500 rules means 1500 dependent rewrites, never parallel banks. See README.
"""
from dataclasses import dataclass, asdict
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def smooth5(t):
    t = t.clamp(0, 1)
    return t*t*t*(10+t*(-15+6*t))


def shift(x, offset):
    """y[i]=x[i+offset]; zero outside capacity."""
    if offset == 0: return x
    if abs(offset) >= x.shape[1]: return torch.zeros_like(x)
    zeros = x.new_zeros((x.shape[0], abs(offset))+x.shape[2:])
    return torch.cat((x[:, offset:], zeros), 1) if offset > 0 else torch.cat((zeros, x[:, :offset]), 1)


def reach_parallel(events):
    """Ready-state probabilities for nonoverlapping variable-span events.
    events[B,L,S]: probability of consuming d tokens in column d-1.
    Column-stochastic S-state automaton, associative prefix product in O(log L)
    dependent steps. O(B L S^3 log L) work; no per-position Python loop.
    """
    b, length, spans = events.shape
    first = torch.cat((1-events[..., 1:].sum(-1, keepdim=True), events[..., 1:]), -1)
    tail = torch.eye(spans, device=events.device, dtype=events.dtype)[:, :-1]
    tail = tail.expand(b, length, spans, spans-1)
    a = torch.cat((first[..., None], tail), -1)
    stride = 1
    while stride < length:
        a = torch.cat((a[:, :stride], a[:, stride:] @ a[:, :-stride]), 1)
        stride *= 2
    return torch.cat((torch.ones_like(events[:, :1, 0]), a[:, :-1, 0, 0]), 1)


def pack(emission):
    """C2 mean-position packing. Exact ordered deletion at hard 0/1 vertices.
    Soft states discard alignment correlations; not exact probabilistic Replace.
    """
    b, length, channels = emission.shape
    mass = emission.sum(-1)
    coord = mass.cumsum(1)-mass
    base = coord.floor()
    frac = smooth5(coord-base)
    out = torch.zeros_like(emission)
    for side, weight in ((0, 1-frac), (1, frac)):
        index = (base.long()+side).clamp(0, length-1)
        out = out.scatter_add(1, index[..., None].expand(b, length, channels),
                              emission*weight[..., None])
    m = out.sum(-1, keepdim=True)
    eps = 1e-5*(1-m).square()
    return (out+eps/channels)/(m+eps), mass.sum(1)


class SerialRule(nn.Module):
    """Learn [left, middle|STAR, right] over bytes + preceding markers.
    Literal middle consumes 3 symbols. STAR captures 0..max_capture symbols,
    ending at the FIRST right match. RHS = fresh marker + captured symbols.
    """
    def __init__(self, alphabet, max_capture):
        super().__init__()
        self.alphabet, self.max_capture = alphabet, max_capture
        self.outer = nn.Parameter(torch.randn(2, alphabet)*.03)
        self.middle = nn.Parameter(torch.randn(alphabet+1)*.03)

    def distributions(self, temperature=1., hard=False):
        if hard:
            outer = F.one_hot(self.outer.argmax(-1), self.alphabet).to(self.outer.dtype)
            middle = F.one_hot(self.middle.argmax(-1), self.alphabet+1).to(self.middle.dtype)
        else:
            outer = (self.outer/temperature).softmax(-1)
            middle = (self.middle/temperature).softmax(-1)
        return outer, middle

    def forward(self, x, lengths, original_lengths, temperature=1., hard=False):
        outer, middle = self.distributions(temperature, hard)
        # All batch items and positions share one matrix multiplication.
        kernels = torch.cat((outer[:1], middle[None, :-1], outer[1:]), 0)
        matches = x @ kernels.T
        left, center, right = matches.unbind(-1)
        mask = smooth5(lengths[:, None]-torch.arange(x.shape[1], device=x.device, dtype=x.dtype))
        literal = left*shift(center, 1)*shift(right, 2)*mask*shift(mask, 1)*shift(mask, 2)
        wild=[]; survival=torch.ones_like(left)
        for gap in range(self.max_capture+1):
            endpoint=shift(right, gap+1)
            valid=mask*shift(mask, gap+1)
            wild.append(left*middle[-1]*survival*endpoint*valid)
            survival=survival*(1-endpoint)
        spans=max(3, self.max_capture+2)
        columns=[]
        for span in range(1,spans+1):
            value=literal if span==3 else torch.zeros_like(literal)
            if 2 <= span <= self.max_capture+2: value=value+wild[span-2]
            columns.append(value)
        q=torch.stack(columns,-1)
        ready=reach_parallel(q)
        fires=ready*q.sum(-1)
        count=fires.sum(1)/original_lengths
        keep=ready*(1-q.sum(-1))*mask
        # Retained captures in original order; one batched vector op per offset.
        for offset in range(1,self.max_capture+1):
            covering=sum(wild[offset:], torch.zeros_like(literal))
            keep=keep+shift(ready*covering,-offset)
        emission=torch.cat((x*keep[...,None],fires[...,None]),-1)
        y,new_lengths=pack(emission)
        return y,new_lengths,count

    def impurity(self, temperature):
        outer,middle=self.distributions(temperature)
        return ((1-outer.square().sum(-1)).mean()+1-middle.square().sum())/2


class SwiGLU(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__(); self.linear=nn.Linear(n_in,2*n_out)
    def forward(self,x):
        a,gate=self.linear(x).chunk(2,-1)
        return a*F.silu(gate)


@dataclass
class Config:
    rules: int = 1500
    hidden: int = 2048
    max_capture: int = 4
    checkpoint_every: int = 25
    feature_gain: float = 256.


class Lens(nn.Module):
    def __init__(self,config=None):
        super().__init__(); self.config=config or Config()
        c=self.config
        if c.rules < 1 or c.hidden < 1 or c.max_capture < 0 or c.checkpoint_every < 0:
            raise ValueError('invalid architecture')
        self.rules=nn.ModuleList([SerialRule(256+i,c.max_capture) for i in range(c.rules)])
        self.head=nn.Sequential(SwiGLU(c.rules,c.hidden),SwiGLU(c.hidden,c.hidden),nn.Linear(c.hidden,1))

    def _block(self,x,lengths,original,temperature,hard,start,end):
        features=[]
        for i in range(start,end):
            x,lengths,f=self.rules[i](x,lengths,original,temperature,hard)
            features.append(f)
        return x,lengths,torch.stack(features,-1)

    def features(self,tokens,lengths,temperature=1.,hard=False):
        if tokens.ndim==2:
            x=F.one_hot(tokens.long(),256).to(self.rules[0].outer.dtype)
        elif tokens.ndim==3 and tokens.shape[-1]==256:
            x=tokens.to(self.rules[0].outer.dtype)
        else: raise ValueError('expected [B,L] byte IDs or [B,L,256] probability vectors')
        if x.shape[1]<3: raise ValueError('capacity must be >=3')
        lengths=lengths.to(x.dtype); original=lengths
        block=self.config.checkpoint_every or len(self.rules)
        feats=[]
        for start in range(0,len(self.rules),block):
            end=min(start+block,len(self.rules))
            # Bind indices: backward recomputation must not see the last block!
            def run(a,b,c,start=start,end=end):
                return self._block(a,b,c,temperature,hard,start,end)
            if self.config.checkpoint_every and self.training and torch.is_grad_enabled() and not hard:
                x,lengths,f=checkpoint(run,x,lengths,original,use_reentrant=False,preserve_rng_state=False)
            else: x,lengths,f=run(x,lengths,original)
            feats.append(f)
        return torch.cat(feats,-1)

    def forward(self,tokens,lengths,temperature=1.,hard=False,amp=False):
        with (torch.autocast(device_type='cuda',enabled=False) if tokens.device.type=='cuda' else nullcontext()):
            f=self.features(tokens,lengths,temperature,hard)
            f=torch.log1p(self.config.feature_gain*f)
        with (torch.autocast(device_type='cuda',dtype=torch.bfloat16)
              if amp and tokens.device.type=='cuda' else nullcontext()):
            pred=self.head(f).squeeze(-1)
        return pred.float() if pred.dtype!=torch.float64 else pred

    def regularizer(self,temperature=1.):
        return torch.stack([rule.impurity(temperature) for rule in self.rules]).mean()

    @torch.no_grad()
    def initialize_from_training(self,chunks,seed=1729,strength=7.):
        """Training-only ngram initialization; no targets or intermediate labels.
        All existing markers remain learnable, without correct-marker hints.
        """
        import random
        rng=random.Random(seed)
        for rule in self.rules:
            c=rng.choice(chunks); i=rng.randrange(len(c)-2); tri=c[i:i+3]
            rule.outer.normal_(0,.03); rule.middle.normal_(0,.03)
            rule.outer[0,tri[0]]=strength; rule.outer[1,tri[2]]=strength
            mid=rule.alphabet if rng.random()<.25 else tri[1]
            rule.middle[mid]=strength

    def export_rules(self):
        rules=[]
        for i,rule in enumerate(self.rules):
            left,right=rule.outer.detach().argmax(-1).cpu().tolist()
            mid=int(rule.middle.detach().argmax())
            mid=-1 if mid==rule.alphabet else mid
            rules.append({'pattern':[left,mid,right],'marker':256+i,
                          'replacement':[256+i]+([-1] if mid==-1 else [])})
        return {'semantics':'serial leftmost nonoverlap, STAR captures 0..max_capture at first delimiter',
                'config':asdict(self.config),'rules':rules}


def discrete_features(tokens,exported):
    """Pure integer rewrites of this bounded serial grammar (counts normalized last)."""
    state=list(tokens); original=len(state); counts=[]
    cap=exported['config']['max_capture']
    for rule in exported['rules']:
        left,mid,right=rule['pattern']; output=[]; count=0; i=0
        while i < len(state):
            end=None
            if state[i]==left:
                if mid==-1:
                    for gap in range(cap+1):
                        j=i+gap+1
                        if j < len(state) and state[j]==right: end=j; break
                elif i+2 < len(state) and state[i+1]==mid and state[i+2]==right: end=i+2
            if end is None: output.append(state[i]); i+=1
            else:
                output.append(rule['marker'])
                if mid==-1: output.extend(state[i+1:end])
                count+=1; i=end+1
        counts.append(count); state=output
    return [c/original for c in counts],state
