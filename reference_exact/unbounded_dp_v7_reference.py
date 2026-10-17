"""Unbounded wildcard soft transducer (GPU/MPS PyTorch).

The *grammar* has no capture-length limit. Only the global finite input/output
buffer limits a particular computation. Each delimiter is located by an
associative (log-depth) first-hit dynamic program; no length-tuple expansion.

Continuous execution is an explicit C2 mean-field relaxation, not an exact
joint probability pushforward. In particular we approximate later delimiter
positions by the conditional expected position of earlier delimiters.
"""
from itertools import combinations
import torch
from torch import nn
from torch.nn import functional as F


def smooth(t):
    t=t.clamp(0.,1.)
    return t*t*t*(10. + t*(-15. + 6.*t))


def cat(logits,temp,hard,mask=None):
    if mask is not None:
        logits=logits.masked_fill(~mask,-torch.inf)
    if hard:
        return F.one_hot(logits.argmax(-1),logits.shape[-1]).to(logits.dtype)
    return (logits/temp).softmax(-1)


def split_splat(coord,limit):
    base=coord.floor()
    frac=smooth(coord-base)
    a=base.long(); b=a+1
    wa=(1-frac)*((a>=0)&(a<limit)).to(coord.dtype)
    wb=frac*((b>=0)&(b<limit)).to(coord.dtype)
    return a.clamp(0,limit-1),b.clamp(0,limit-1),wa,wb


def interp_scan(values,coord):
    """C2 interpolation of [batch,positions] array at continuous coordinates."""
    n=values.shape[1]
    a,b,wa,wb=split_splat(coord,n)
    # values includes an explicit t=n sentinel (hit probability zero).
    return wa*values.gather(1,a)+wb*values.gather(1,b)


def first_hit_dp(hit):
    """Associative affine scan for first delimiter: O(log L) depth, O(L log L) work.

    Return P(first match exists), E[first position * exists] at every starting
    cursor. Expected FIRST occurrence is weighted by survival of all earlier
    alternatives; we never enumerate individual capture lengths.
    """
    batch,length=hit.shape
    position=torch.arange(length,device=hit.device,dtype=hit.dtype)[None,:]
    survival=1.-hit
    p=hit
    moment=hit*position
    stride=1
    while stride<length:
        a=survival
        # padding with zeros is the neutral suffix for P and moment.
        qp=F.pad(p[:,stride:],(0,stride))
        qm=F.pad(moment[:,stride:],(0,stride))
        qsurvival=F.pad(survival[:,stride:],(0,stride),value=1.)
        p=p+a*qp
        moment=moment+a*qm
        survival=a*qsurvival
        stride*=2
    # one extra zero sentinel supports boundary interpolation
    return F.pad(p,(0,1)),F.pad(moment,(0,1))


def dense_select(probs,endpoint):
    """Greedy non-overlap as batched differentiable DAG forward DP.

    `probs`: [B,L,H] candidate probabilities from each input position.
    `endpoint`: [B,L,H] their soft end positions >= start+1.
    The triangular system has L+1 states, uses two-neighbor output splatting,
    and is C2 (including in the endpoint coordinates).
    """
    b,l,h=probs.shape
    start=torch.arange(l,device=probs.device)[None,:,None]
    endpoint=torch.maximum(endpoint,start.to(probs.dtype)+1.)
    e0,e1,w0,w1=split_splat(endpoint,l+1)
    # build transitions T[to,from], strictly lower triangular
    t=probs.new_zeros(b,(l+1)*(l+1))
    from_idx=start.expand(b,l,h)
    for e,w in ((e0,w0),(e1,w1)):
        target=e*(l+1)+from_idx
        t=t.scatter_add(1,target.reshape(b,-1),(probs*w).reshape(b,-1))
    fire=probs.sum(-1)
    j=torch.arange(l,device=probs.device)[None,:].expand(b,-1)
    keep_targets=(j+1)*(l+1)+j
    t=t.scatter_add(1,keep_targets,1.-fire)
    t=t.reshape(b,l+1,l+1)
    matrix=torch.eye(l+1,dtype=probs.dtype,device=probs.device)[None]-t
    rhs=probs.new_zeros(b,l+1,1)
    rhs[:,0,0]=1.
    if probs.device.type=='mps':
        # Some MPS/PyTorch combinations do not implement batched triangular
        # solves or their higher-order VJP. The DAG recurrence uses only
        # portable operations; it remains differentiable, but is less fast.
        ready=[torch.ones(b,device=probs.device,dtype=probs.dtype)]
        for i in range(1,l):
            ready.append((t[:,i,:i]*torch.stack(ready,dim=1)).sum(-1))
        return torch.stack(ready,dim=1)
    readiness=torch.linalg.solve_triangular(matrix,rhs,upper=False).squeeze(-1)[:,:-1]
    return readiness


def chunk_hits(x,pat,positions,lengths):
    """Probability of a literal chunk at every position, using mean-field bytes."""
    b,l,v=x.shape
    hit=x.new_ones((b,l))
    for j,position in enumerate(positions):
        score=x@pat[position]
        if j:
            score=F.pad(score[:,j:],(0,j))
        hit=hit*score
    coords=torch.arange(l,device=x.device,dtype=x.dtype)[None,:]
    hit=hit*smooth(lengths[:,None]-(coords+len(positions)-1))
    return hit


class SerialRule(nn.Module):
    """Finite-memory soft grammar with UNBOUNDED greedy captures.

    max_capture is accepted only as an ignored positional/keyword compatibility
    argument for checkpoints and older scripts. It NEVER limits captures.
    """
    def __init__(self,alphabet=256,max_pattern=4,max_replacement=4,max_capture=None,max_stars=3):
        super().__init__()
        if alphabet<2 or max_pattern<1 or max_replacement<1 or max_stars<1:
            raise ValueError('invalid grammar dimensions')
        if max_stars>min(15,max_pattern):
            raise ValueError('max_stars must be <= min(15,max_pattern)')
        self.alphabet=alphabet
        self.max_pattern=max_pattern
        self.max_replacement=max_replacement
        self.max_capture=None  # NO LIMIT; compatibility parameter ignored
        self.max_stars=max_stars
        self.pattern_bytes=nn.Parameter(torch.randn(max_pattern,alphabet)*.03)
        self.pattern_length=nn.Parameter(torch.zeros(max_pattern))
        self.replacement_length=nn.Parameter(torch.zeros(max_replacement+1))
        self.replacement_tokens=nn.Parameter(torch.randn(max_replacement,alphabet+max_stars)*.03)
        choices=[]
        geometries=[]
        for n in range(1,max_pattern+1):
            opts=[]
            for k in range(min(max_stars,n)+1):
                opts+=list(combinations(range(n),k))
            choices.append(opts)
            for oi,stars in enumerate(opts):
                if len(stars)==n:continue  # pure STAR is a no-op
                parts=[[]]
                for pos in range(n):
                    if pos in stars: parts.append([])
                    else: parts[-1].append(pos)
                geometries.append((n-1,oi,tuple(tuple(p) for p in parts)))
        self._choices=choices
        self._geometries=geometries
        star_mask=torch.zeros(max_pattern,max(map(len,choices)),dtype=torch.bool)
        for i,opts in enumerate(choices):star_mask[i,:len(opts)]=True
        self.pattern_stars=nn.Parameter(torch.zeros(max_pattern,star_mask.shape[1]))
        self.register_buffer('star_choice_mask',star_mask,persistent=False)

    def distributions(self,temperature=1.,hard=False):
        return (cat(self.pattern_bytes,temperature,hard),
                cat(self.pattern_length,temperature,hard),
                cat(self.replacement_length,temperature,hard),
                cat(self.pattern_stars,temperature,hard,self.star_choice_mask),
                self.replacement_tokens)

    def _candidates(self,x,lengths,pb,pl,ps):
        b,l,_=x.shape
        positions=torch.arange(l,device=x.device,dtype=x.dtype)[None,:]
        matches=[];ends=[];starts=[];stops=[];priors=[];ks=[]
        hit_cache={}; dp_cache={}
        def get_hit(literal):
            if literal not in hit_cache:
                hit_cache[literal]=chunk_hits(x,pb,literal,lengths)
            return hit_cache[literal]
        def get_first(literal):
            if literal not in dp_cache:
                dp_cache[literal]=first_hit_dp(get_hit(literal))
            return dp_cache[literal]
        # Only combinations of STAR positions, NEVER combinations of capture lengths.
        # For every literal part, first-hit DP is computed once per geometry.
        for n_ix,choice_ix,parts in self._geometries:
            k=len(parts)-1
            prefix=parts[0]
            if prefix:
                p=get_hit(prefix)
            else:
                p=smooth(lengths[:,None]-positions)
            cursor=positions.expand(b,-1)+len(prefix)
            cap_start=[];cap_stop=[]
            for j,literal in enumerate(parts[1:]):
                cap_start.append(cursor)
                if not literal:
                    # Adjacent STARs capture the empty string; terminal STAR
                    # consumes the entire remaining suffix of the input.
                    terminal=(j==k-1)
                    stop=lengths[:,None].expand_as(cursor) if terminal else cursor
                    cap_stop.append(stop)
                    cursor=stop
                else:
                    presence,moment=get_first(literal)
                    found=interp_scan(presence,cursor)
                    weighted=interp_scan(moment,cursor)
                    stop=weighted/(found+1e-12)
                    # The conditional mean is defined arbitrarily where no
                    # match exists; p=0 makes that branch's emission vanish.
                    stop=torch.where(found>1e-9,stop,cursor)
                    cap_stop.append(stop)
                    cursor=stop+len(literal)
                    p=p*found
            # No STAR: initial whole literal must fit in remaining input.
            p=p*smooth(lengths[:,None]-positions)
            matches.append(p)
            ends.append(cursor)
            prior=pl[n_ix]*ps[n_ix,choice_ix]
            priors.append(prior)
            ks.append(k)
            empty=positions.new_zeros(b,l)
            starts.append(torch.stack(cap_start+[empty]*(self.max_stars-k),dim=-1))
            stops.append(torch.stack(cap_stop+[empty]*(self.max_stars-k),dim=-1))
        score=torch.stack(matches,-1)
        endpoints=torch.stack(ends,-1)
        cap_start=torch.stack(starts,2)
        cap_stop=torch.stack(stops,2)
        prior=torch.stack(priors)
        return score*prior[None,None,:],endpoints,cap_start,cap_stop,ks

    def _soft_rhs(self,rl,rt,ks,temperature):
        h=len(ks); s=self.max_stars; v=self.alphabet
        options=torch.arange(v+s,device=rt.device)
        masks=options[None,:] < torch.tensor([v+k for k in ks],device=rt.device)[:,None]
        # same RHS logits across geometries, except invalid capture refs
        log=rt[None,:,:].expand(h,-1,-1)
        prob=cat(log,temperature,False,masks[:,None,:])
        alive=torch.flip(rl[1:].flip(0).cumsum(0),dims=(0,))
        return prob[:,:,:v],prob[:,:,v:],alive

    def forward(self,x,lengths,original_lengths,temperature=1.,hard=False,distributions=None):
        """C2 GPU/MPS transducer; captures extend to the full actual string length.

        In contrast to length-tuple enumeration, cost no longer grows as
        (max_capture+1)**max_stars. A dense B*L*L output map is presently used
        for reordered or duplicated captures. This is the bottleneck for long
        sequences (separate from the first-hit DP).
        """
        pb,pl,rl,ps,rt=(distributions if distributions is not None
                         else self.distributions(temperature,hard))
        b,l,v=x.shape
        if v!=self.alphabet:raise ValueError('wrong alphabet')
        if l<2:raise ValueError('requires >=2 source positions')
        event,end,cstart,cend,ks=self._candidates(x,lengths,pb,pl,ps)
        # Ensure smooth stochastic variants retain nonnegative firing/keep.
        readiness=dense_select(event,end)
        weighted=event*readiness[:,:,None]
        fires=event.sum(-1)
        pos=torch.arange(l,device=x.device,dtype=x.dtype)
        keep=readiness*(1.-fires)*smooth(lengths[:,None]-pos[None,:])
        lit,ref,alive=self._soft_rhs(rl,rt,ks,temperature)
        # alive position probabilities and conditional capture lengths
        caplen=(cend-cstart).clamp_min(0.)
        litlen=lit.sum(-1)
        tokenlen=litlen[None,None,:,:]+torch.einsum('hrs,blhs->blhr',ref,caplen)
        tokenlen=tokenlen*alive[None,None,None,:]
        eventlen=tokenlen.sum(-1)
        perpos=keep+(weighted*eventlen).sum(-1)
        base=perpos.cumsum(1)-perpos
        output_length=perpos.sum(-1)
        # Global output capacity is NOT a capture limit; exceeding it is an
        # explicit error rather than silent truncation.
        if bool((output_length>l+1e-3).any().detach()):
            raise OverflowError('unbounded captures expanded output beyond tensor capacity; increase --capacity-factor')
        h=len(ks);r=self.max_replacement; s=self.max_stars
        # Only a source-to-target coefficient map (not byte-channel-per-event)
        # is constructed. Matrix multiplication recovers all 256 channels.
        source=x.new_zeros((b,l*l))
        literal=x.new_zeros((b,l*h*r))
        ix=torch.arange(l,device=x.device)[None,:].expand(b,l)

        def sparse_add(dst,coordinate,mass,source_index,outwidth):
            t0,t1,w0,w1=split_splat(coordinate,l)
            for t,w in ((t0,w0),(t1,w1)):
                idx=(t*outwidth+source_index).reshape(b,-1)
                dst=dst.scatter_add(1,idx,(mass*w).reshape(b,-1))
            return dst

        source=sparse_add(source,base,keep,ix,l)
        prefix=tokenlen.cumsum(-1)-tokenlen
        # literal RHS emission (small: [B,L,H,R])
        co=base[:,:,None,None]+prefix
        lit_mass=(weighted[:,:,:,None]*alive[None,None,None,:]).expand(-1,-1,h,-1)
        rhs_ix=torch.arange(h*r,device=x.device).reshape(1,1,h,r).expand(b,l,-1,-1)
        literal=sparse_add(literal,co,lit_mass,rhs_ix,h*r)
        # source copying. All capture references can be duplicated, omitted,
        # or reordered. No finite capture-length axis is constructed.
        source_positions=pos[None,:,None,None,None]  # [1,L,1,1,1]
        for reference in range(s):
            lengthmask=torch.tensor([reference<k for k in ks],device=x.device,dtype=x.dtype)
            if not lengthmask.any():continue
            cs=cstart[:,:,:,reference]
            ce=cend[:,:,:,reference]
            # source positions j belong to [capture_start,capture_end)
            # smooth5 gives C2 boundaries, exact at integral vertices.
            j=pos[None,None,None,:]
            membership=smooth(j-cs[:,:,:,None]+1.)*smooth(ce[:,:,:,None]-j)
            # [B,L,H,L] coefficient and output position for each source j.
            src_ix=torch.arange(l,device=x.device)[None,None,None,:].expand(b,l,h,-1)
            for t in range(r):
                weight=(weighted*ref[:,t,reference][None,None,:]*alive[t]
                        *lengthmask[None,None,:])
                coordinate=(base[:,:,None,None]+prefix[:,:,:,t,None]
                            +j-cs[:,:,:,None])
                source=sparse_add(source,coordinate,weight[:,:,:,None]*membership,src_ix,l)
        mapped=source.reshape(b,l,l)@x
        rhs_table=lit.reshape(h*r,v)
        mapped=mapped+literal.reshape(b,l,h*r)@rhs_table
        mass=mapped.sum(-1,keepdim=True)
        eps=1e-5*(1-mass).square()
        result=(mapped+eps/v)/(mass+eps)
        return result,output_length,weighted.sum((1,2))/original_lengths

    def impurity(self,temperature):
        a,b,c,d,e=self.distributions(temperature)
        e=cat(e,temperature,False)
        return torch.stack([(1-v.square().sum(-1)).mean() for v in (a,b,c,d,e)]).mean()
