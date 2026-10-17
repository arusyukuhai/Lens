"""Lens with old-Nim greedy wildcards and unbounded differentiable captures.

Exact: CPU integer Replace with original delimiter-first semantics.
Soft: batched C2 first-hit DP + stochastic greedy interval DP + output splat.
Soft is a mean-field relaxation, not the exact joint probability pushforward.
"""
from dataclasses import dataclass, asdict
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
if __package__:
    from .old_replace import serial_features as old_serial_features
    from .unbounded_dp import SerialRule, cat as _categorical
else:
    from old_replace import serial_features as old_serial_features
    from unbounded_dp import SerialRule, cat as _categorical


class SwiGLU(nn.Module):
    def __init__(self,n_in,n_out):
        super().__init__();self.linear=nn.Linear(n_in,2*n_out)
    def forward(self,x):
        a,gate=self.linear(x).chunk(2,-1)
        return a*F.silu(gate)


@dataclass
class Config:
    rules:int=1500
    hidden:int=2048
    max_capture:int=0  # legacy checkpoint metadata; ignored by unbounded engine
    max_stars:int=3
    checkpoint_every:int=25
    max_pattern:int=4
    max_replacement:int=4
    feature_gain:float=256.
    format_version:int=7
    exact_max_output:int=1048576


class Lens(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.config=config or Config()
        c=self.config
        if min(c.rules,c.hidden,c.max_stars,c.max_pattern,c.max_replacement)<1 or c.checkpoint_every<0:
            raise ValueError('invalid architecture')
        if c.format_version!=7:raise ValueError('unbounded wildcard model requires format_version=7')
        if c.exact_max_output<1:raise ValueError('exact_max_output must be positive')
        self.rules=nn.ModuleList([SerialRule(256,c.max_pattern,c.max_replacement,c.max_capture,c.max_stars)
                                  for _ in range(c.rules)])
        self.head=nn.Sequential(SwiGLU(c.rules,c.hidden),SwiGLU(c.hidden,c.hidden),nn.Linear(c.hidden,1))

    def _apply(self,fn,recurse=True):
        """Move the common grammar mask only once for all serial rules."""
        if not getattr(self,'rules',None) or len(self.rules)==0:
            return super()._apply(fn,recurse=recurse)
        self.head._apply(fn)
        first=self.rules[0]
        first._apply(fn)
        shared_buffers=dict(first.named_buffers(recurse=False))
        for rule in self.rules[1:]:
            original_buffers=rule._buffers
            rule._buffers={}  # temporarily exclude identical immutable buffers
            try:
                rule._apply(fn)
            finally:
                rule._buffers=original_buffers
            for name,buffer in shared_buffers.items():
                rule._buffers[name]=buffer
        return self

    def _distributions(self,temperature,hard):
        names=('pattern_bytes','pattern_length','replacement_length','pattern_stars','replacement_tokens')
        tensors=[torch.stack([getattr(r,name) for r in self.rules]) for name in names]
        a=self.rules[0]
        return [_categorical(t,temperature,hard,a.star_choice_mask if name=='pattern_stars' else None)
                if name!='replacement_tokens' else t for name,t in zip(names,tensors)]

    def _block(self,x,lengths,original,dist,start,end):
        features=[]
        for i in range(start,end):
            d=[t[i] for t in dist]
            x,lengths,f=self.rules[i](x,lengths,original,distributions=d)
            features.append(f)
        return x,lengths,torch.stack(features,-1)

    @torch.no_grad()
    def exact_features(self,tokens,lengths,return_lengths=False):
        """Original old-Nim semantics; CPU integer interpreter, GPU ML head.

        No parameter gradients through the integer rules. Pure wildcard LHS
        is a no-op; terminal capture is UNBOUNDED (up to exact_max_output).
        """
        if tokens.ndim==3:
            if tokens.shape[-1]!=256:raise ValueError('expected 256 byte channels')
            tokens=tokens.argmax(-1)
        if tokens.ndim!=2 or lengths.ndim!=1 or tokens.shape[0]!=lengths.shape[0]:
            raise ValueError('expected tokens [B,L] and lengths [B]')
        lengths_cpu=lengths.detach().cpu().tolist()
        if bool(((tokens<0)|(tokens>255)).any()):
            raise ValueError('exact mode expects byte IDs 0..255')
        x_cpu=tokens.detach().to('cpu',dtype=torch.uint8).tolist()
        rules=[(tuple(row['pattern']),tuple(row['replacement']))
               for row in self.export_rules()['rules']]
        result=[];new_lengths=[]
        for row,n in zip(x_cpu,lengths_cpu):
            n_int=int(n)
            if n_int<1 or n_int>len(row) or abs(n-n_int)>1e-5:
                raise ValueError('exact mode requires integral, nonempty lengths within capacity')
            feats,out=old_serial_features(bytes(row[:n_int]),rules,
                                          max_output=self.config.exact_max_output)
            result.append(list(feats));new_lengths.append(len(out))
        f=torch.tensor(result,device=tokens.device,dtype=self.rules[0].pattern_bytes.dtype)
        if return_lengths:
            nl=torch.tensor(new_lengths,device=tokens.device,dtype=f.dtype)
            return f,nl
        return f

    def features(self,tokens,lengths,temperature=1.,hard=False,return_lengths=False):
        if hard:
            return self.exact_features(tokens,lengths,return_lengths)
        if tokens.ndim==2:x=F.one_hot(tokens.long(),256).to(self.rules[0].pattern_bytes.dtype)
        elif tokens.ndim==3 and tokens.shape[-1]==256:x=tokens.to(self.rules[0].pattern_bytes.dtype)
        else:raise ValueError('expected [B,L] byte IDs or [B,L,256] distributions')
        if x.shape[1]<2:raise ValueError('capacity must be >=2')
        lengths=lengths.to(x.dtype)
        if torch.any(lengths>x.shape[1]):raise ValueError('input lengths exceed capacity')
        original=lengths
        dist=self._distributions(temperature,hard)
        block=self.config.checkpoint_every or len(self.rules)
        feats=[]
        for start in range(0,len(self.rules),block):
            end=min(start+block,len(self.rules))
            def run(a,b,c,start=start,end=end):return self._block(a,b,c,dist,start,end)
            if self.config.checkpoint_every and self.training and torch.is_grad_enabled() and not hard:
                x,lengths,f=checkpoint(run,x,lengths,original,use_reentrant=False,preserve_rng_state=False)
            else:x,lengths,f=run(x,lengths,original)
            if hard and bool((lengths>x.shape[1]+1e-5).any()):
                raise OverflowError('expanded text exceeds tensor capacity; increase --capacity-factor')
            feats.append(f)
        f=torch.cat(feats,-1)
        return (f,lengths) if return_lengths else f

    def forward(self,tokens,lengths,temperature=1.,hard=False,amp=False):
        with (torch.autocast(device_type='cuda',enabled=False) if tokens.device.type=='cuda' else nullcontext()):
            f=self.features(tokens,lengths,temperature,hard)
            f=torch.log1p(self.config.feature_gain*f)
        with (torch.autocast(device_type='cuda',dtype=torch.bfloat16) if amp and tokens.device.type=='cuda' else nullcontext()):
            y=self.head(f).squeeze(-1)
        return y.float() if y.dtype!=torch.float64 else y

    def regularizer(self,temperature=1.):
        pb,pl,rl,ps,rt=self._distributions(temperature,False)
        rt=_categorical(rt,temperature,False)
        return torch.stack([(1-v.square().sum(-1)).mean() for v in (pb,pl,rl,ps,rt)]).mean()

    def enable_compiled_rule(self,mode='default'):
        raise RuntimeError('torch.compile for multi-capture rule is not yet validated')

    @torch.no_grad()
    def initialize_from_training(self,chunks,seed=1729,strength=7.):
        import random
        rng=random.Random(seed)
        for rule in self.rules:
            chunk=rng.choice(chunks)
            n=rng.randrange(1,min(rule.max_pattern,len(chunk))+1)
            start=rng.randrange(len(chunk)-n+1)
            tri=chunk[start:start+n]
            rule.pattern_bytes.normal_(0,.03)
            rule.replacement_tokens.normal_(0,.03)
            rule.pattern_length.zero_();rule.pattern_length[n-1]=strength
            rule.replacement_length.zero_();rule.replacement_length[1]=strength
            rule.pattern_stars.zero_()
            subset=()
            # Seed some independently captured wildcards so they are not
            # trapped behind a near-deterministic no-STAR initialization.
            if rng.random()<.30:
                # Old semantics rejects patterns consisting only of STARS.
                k=rng.randrange(1,min(n-1,rule.max_stars)+1) if n>1 else 0
                subset=tuple(sorted(rng.sample(range(n),k))) if k else ()
            choice=rule._choices[n-1].index(subset)
            rule.pattern_stars[n-1,choice]=strength
            for j,byte in enumerate(tri):rule.pattern_bytes[j,byte]=strength
            if subset and rng.random()<.7:
                # RHS uses one of the named captures without duplication,
                # avoiding runaway initial text growth.
                ref=rng.randrange(len(subset))
                rule.replacement_tokens[0,256+ref]=strength
            else:
                rule.replacement_tokens[0,rng.randrange(256)]=strength

    @torch.no_grad()
    def export_rules(self):
        """One batched GPU->CPU transfer per categorical table, not R*P syncs."""
        # Raw argmax is identical to the hard categorical distribution.
        # STAR positions use a per-pattern length mask, and RHS capture
        # references are limited to captures actually present in that rule.
        names=('pattern_bytes','pattern_length','replacement_length',
               'pattern_stars','replacement_tokens')
        pb,pl,rl,ps,rt=(torch.stack([getattr(rule,name) for rule in self.rules])
                        for name in names)
        n=pl.argmax(-1)+1
        m=rl.argmax(-1)
        rcount=len(self.rules)
        idx=torch.arange(rcount,device=n.device)
        choice=ps[idx,n-1].masked_fill(~self.rules[0].star_choice_mask[n-1],-torch.inf).argmax(-1)
        # Bit masks are determined only by geometry and can be materialized
        # once per export. This copy is tiny compared to the model weights.
        pattern_bytes=pb.argmax(-1).detach().cpu().tolist()
        nlist=n.detach().cpu().tolist()
        mlist=m.detach().cpu().tolist()
        choices=choice.detach().cpu().tolist()
        raw_rhs=rt.detach().cpu()
        output=[]
        for i,rule in enumerate(self.rules):
            N=nlist[i];M=mlist[i]
            subset=rule._choices[N-1][choices[i]]
            pat=pattern_bytes[i][:N]
            for k,pos in enumerate(subset):pat[pos]=-k-1
            rhs_logits=raw_rhs[i,:M,:256+len(subset)]
            tok=rhs_logits.argmax(-1).tolist()
            rhs=[v if v<256 else -(v-256+1) for v in tok]
            output.append({'pattern':pat,'replacement':rhs})
        return {'semantics':'old-Nim delimiter-first/leftmost nonoverlap; trailing STAR captures entire suffix; pure STAR pattern is no-op; -i denotes capture i',
                'config':asdict(self.config),'rules':output}


def discrete_features(tokens,exported,capacity=None):
    """Exact old-Nim byte interpreter, with per-rule firing rates.

    Both the old integer executor and the soft DP have no capture length limit.
    """
    rules=exported['rules']
    x=bytes(tokens)
    if not x:raise ValueError('empty input')
    limit=exported.get('config',{}).get('exact_max_output',1048576)
    seq=[(tuple(r['pattern']),tuple(r['replacement'])) for r in rules]
    features,out=old_serial_features(x,seq,max_output=limit)
    if capacity is not None and len(out)>capacity:
        raise OverflowError(f'exact rewrite length {len(out)} exceeds capacity {capacity}')
    return list(features),list(out)
