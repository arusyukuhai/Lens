"""Relational BIPS: synthesize multi-rule paths for correct-vs-wrong edits.

This is a bounded, empirical paired-state program synthesizer, not formal
backward semantics or a proof of cross-corpus generalization. The last rolling
original is never accessed. Candidate *construction* sees only its own source
original, and native counterfactual scoring uses another original.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import lens_bips as bp

@dataclass
class RelationalStats:
    proposals: int = 0
    consumer_proposals: int = 0
    tested: int = 0
    prefix_refined: int = 0
    bridge_refined: int = 0
    pair_checks: int = 0
    witness_positive: int = 0
    consumer_activated: int = 0
    scored: int = 0
    readout_improved: int = 0
    structural_only: int = 0
    adopted: int = 0
    best_margin_gain: float = 0.
    best_pair_accuracy: float = 0.


def paired_edits(slot, max_pairs=6):
    """Counterfactual at the same edited coordinate, both from ONE original.

    Harder than comparing clean with noisy: both variants edit the noisy byte,
    but only the positive one reinstates the correct token. Negative edits are
    deterministic and unequal to both the correct byte and the noisy byte.
    """
    inputs=[bytes(v) for v in slot.inputs]
    target=np.asarray(slot.target,dtype=float)
    if not inputs or len(target)!=len(inputs):return []
    ref=int(np.argmax(target)); clean=inputs[ref]
    if target[ref] < 1-1e-8:return []
    results=[];seen=set()
    for k in np.argsort(-target,kind='stable'):
        noisy=inputs[int(k)]
        if len(noisy)!=len(clean) or noisy==clean:continue
        diff=[i for i,(a,b) in enumerate(zip(noisy,clean)) if a!=b]
        for pos in diff[:5]:
            if pos in seen:continue
            right=bytearray(noisy);right[pos]=clean[pos]
            wrong=bytearray(noisy)
            wrong[pos]=(noisy[pos]+1)%256
            if wrong[pos]==clean[pos]:wrong[pos]=(noisy[pos]+2)%256
            if wrong[pos]==noisy[pos]:continue
            seen.add(pos)
            results.append((bytes(right),bytes(wrong),pos))
            if len(results)>=max_pairs:return results
    return results


def _candidate_consumers(champion, features, cap=4):
    """Consumers are existing positively weighted literal rules.

    Later execution must be checked concretely: positive weight alone never
    proves that a consumer improves prediction.
    """
    weights=np.asarray(champion.readout_weights,dtype=float)
    activity=np.asarray(features,dtype=float)
    if activity.ndim!=2 or activity.shape[1]!=len(champion.rules):
        activity=np.zeros((1,len(champion.rules)))
    selected=[]
    for i,r in enumerate(champion.rules):
        pat=tuple(map(int,r.pattern))
        if not 1<=len(pat)<=3 or not all(0<=z<512 for z in pat):continue
        if i>=len(weights) or weights[i]<=1e-8:continue
        util=float(weights[i])*float(np.mean(activity[:,i]>0))
        if util>0:selected.append((util,i,pat))
    return [(i,p) for _,i,p in sorted(selected,reverse=True)[:cap]]


def mine_relational_paths(rolling, champion, features,
                          *, max_proposals=24, max_pairs=6, depth=4,
                          goal='mixed'):
    """Mine differentially matching multi-rule chains per source original.

    No population-heldout original participates; no validation original is
    accessed for constructing any one proposal. Source pairs are the only
    witness. Goal `consumer` rewrites to an EXISTING downstream rule pattern.
    """
    if goal not in ('mixed','consumer','marker'):
        raise ValueError('invalid relational goal')
    if depth not in (2,3,4):raise ValueError('depth must be 2..4')
    scored=[]
    slots=rolling.slots[:-1] if len(rolling.slots)>=2 else []
    n_training=sum(len(sl.inputs) for sl in slots)
    consumers=_candidate_consumers(champion,features[:n_training])
    for ci,slot in enumerate(slots):
        pairs=paired_edits(slot,max_pairs)
        if not pairs:continue
        count=defaultdict(float)
        for good,bad,pos in pairs:
            for left,right in ((0,1),(1,1),(2,1),(1,2),(2,2)):
                lo=max(0,pos-left);hi=min(len(good),pos+right)
                prefix=good[lo:hi]
                if not prefix or prefix in bad:continue
                for gap in (0,1,3,6,10):
                    begin=hi+gap
                    suffix=good[begin:begin+2]
                    if not suffix:continue
                    if bp._ordered_occurrence(bad,(prefix,suffix)):continue
                    count[(prefix,suffix,(),2)]+=1.
                    if depth>=3 and gap>=2:
                        mid=good[hi+gap//2:hi+gap//2+1]
                        if mid:count[(prefix,suffix,(mid,),3)]+=1.
                    if depth>=4 and gap>=5:
                        m1=good[hi+gap//3:hi+gap//3+1]
                        m2=good[hi+(2*gap)//3:hi+(2*gap)//3+1]
                        if m1 and m2 and hi+gap//3+1<=hi+(2*gap)//3:
                            count[(prefix,suffix,(m1,m2),4)]+=1.
        for (prefix,suffix,mid,d),cnt in count.items():
            anchors=(prefix,)+tuple(mid)+(suffix,)
            structural=sum(int(bp._ordered_occurrence(g,anchors))-
                           int(bp._ordered_occurrence(w,anchors))
                           for g,w,_ in pairs)
            if structural<=0:continue
            rel=float(structural)/len(pairs)
            value=(0.6*rel+0.4*cnt/len(pairs))/(1+0.015*sum(map(len,anchors)))
            kwargs=dict(score=value,prefix=prefix,suffix=suffix,
                        middle=mid[0] if mid else b'',bridges=tuple(mid[1:]),
                        origin_case=ci,goal_kind='marker')
            if goal!='consumer':scored.append(bp.PathProposal(**kwargs))
            if goal!='marker':
                for consumer_id,pattern in consumers:
                    if len(pattern)>1+len(suffix):continue
                    scored.append(bp.PathProposal(**{**kwargs,'goal_kind':'consumer',
                              'consumer_index':consumer_id,'consumer_pattern':pattern}))
    # Require distinct goals, original sources and lengths, not just n-gram repeats.
    scored.sort(key=lambda p:(-p.score,p.origin_case,p.depth,p.anchors,p.goal_kind))
    lanes=defaultdict(list)
    for p in scored:lanes[(p.origin_case,p.goal_kind,p.depth)].append(p)
    ordered=[]
    keys=sorted(lanes,key=lambda k:(k[0],k[1]!='consumer',k[2]))
    for ix in range(max((len(lanes[k]) for k in keys),default=0)):
        for key in keys:
            if ix<len(lanes[key]):ordered.append(lanes[key][ix])
    seen=set();unique=[]
    for p in ordered:
        key=(p.anchors,p.origin_case,p.goal_kind,p.consumer_index)
        if key not in seen:
            seen.add(key);unique.append(p)
        if len(unique)>=max_proposals:break
    return unique


def refine_relational_offspring(mod, champion, proposals, features, baseline,
                                target, rolling, count, max_output, phases,
                                cpu_threads, *, budget=8, pairs_per_source=6,
                                min_margin_gain=0.001, allow_structural=True):
    """Verify paired *state executions*, not just the terminal marker firing.

    Producer rules must fire preferentially on correct edits, optionally
    reaching a pre-existing consumer. The score uses a *source-trained* OLS
    readout measured on a different original's paired edits. No synthetic
    targets are added to the main OLS objective or the inference GA.
    """
    import lens_bips_cegar as cg
    stats=RelationalStats(proposals=len(proposals),consumer_proposals=sum(
        p.goal_kind=='consumer' for p in proposals))
    if count<=0 or budget<=0 or len(rolling.slots)<3 or not proposals:return [],stats
    rows,groups,x,b,y,w,_=cg._training_context(rolling,features,baseline,target,champion)
    n=len(champion.rules);candidates=[]
    pair_cache={}
    for group in np.unique(groups):
        pair_cache[int(group)]=paired_edits(rolling.slots[int(group)],pairs_per_source)
    # Build a disjoint validation cohort for each source. Cached parent
    # features reduce costly all-genome calls across candidates.
    parent_pairs={}
    for ci in pair_cache:
        edit=[list(s) for pair in pair_cache[ci] for s in pair[:2]]
        if edit:
            parent_pairs[ci]=cg._concrete_features(
                mod,champion,edit,max_output,phases,cpu_threads)
    per_group={}
    for ci in pair_cache:
        train=np.flatnonzero(groups==ci)
        if len(train)<8:continue
        per_group[ci]=(train, x[train], b[train], y[train])
    tested=set()
    prefix_cache={}
    for p in proposals:
        if stats.tested>=int(budget):break
        source=int(p.origin_case)
        if source not in per_group:continue
        other=[ci for ci in pair_cache if ci!=source and len(pair_cache[ci])>=1]
        if not other:continue
        validation=other[0]
        pairs=pair_cache[validation]
        edit=[list(s) for pair in pairs for s in pair[:2]]
        # Three placements, with consumer after all newly written rules.
        starts=cg._diverse_windows(champion,x,p.depth,3)
        for at in starts:
            if stats.tested>=int(budget):break
            current=p
            # CEGAR on the SOURCE original only: capture the actual state
            # immediately before the insertion boundary. Neither validation
            # bytes nor their oracle targets can influence synthesis.
            source_good=[pair[0] for pair in pair_cache[source][:3]]
            if source_good:
                cache_key=(source,at)
                if cache_key not in prefix_cache:
                    try:
                        from native_cpu_replacer import prefix_states_native
                        prefix_cache[cache_key]=prefix_states_native(
                            [list(v) for v in source_good],champion,at,max_output,
                            phases,cpu_threads=max(1,min(int(cpu_threads),4)))
                    except (ImportError,OSError,RuntimeError):
                        prefix_cache[cache_key]=[]
                states=prefix_cache[cache_key]
                if states and all(st is not None for st in states):
                    current,changed=cg._repair_path_from_prefix(
                        current,states,source_good,champion.embedding,len(states))
                    stats.prefix_refined+=int(changed)
                    newp,nbr,proved=cg._repair_chain_bridges(
                        mod,champion,current,at,states,source_good,max_output)
                    if proved:
                        current=newp;stats.bridge_refined+=nbr
            if current.goal_kind=='consumer' and p.consumer_index<at+p.depth:continue
            key=(p.path_key,at)
            if key in tested:continue
            tested.add(key)
            child=cg._make_child(mod,champion,current,at)
            if child is None:continue
            stats.tested+=1
            fx,bx=cg._concrete_features(mod,child,edit,max_output,phases,cpu_threads)
            stats.pair_checks+=len(pairs)
            stage=fx[:,at+p.depth-1].reshape(-1,2)
            # For consumer goals, delivery has to be consumed by the existing
            # reader, not merely detected at a newly introduced marker.
            stage_diff=stage[:,0]-stage[:,1]
            if not np.any(stage_diff>0):continue
            stats.witness_positive+=1
            reader_diff=0.
            if current.goal_kind=='consumer':
                k=current.consumer_index
                oldf,_=parent_pairs[validation]
                reader_delta=(fx[:,k]-oldf[:,k]).reshape(-1,2)
                reader_diff=float(np.mean(reader_delta[:,0]-reader_delta[:,1]))
                if reader_diff<=1e-8:continue
                stats.consumer_activated+=1
            train,tx,tb,ty=per_group[source]
            oldf,oldb=parent_pairs[validation]
            fparent=np.concatenate((tx,oldf),axis=0)
            bparent=np.concatenate((tb,oldb))
            # Train readout on the SAME source original, with NO val edits.
            train_samples=[list(rows[i]) for i in train]
            fc,bc=cg._concrete_features(mod,child,train_samples,
                                        max_output,phases,cpu_threads)
            fchild=np.concatenate((fc,fx),axis=0)
            bchild=np.concatenate((bc,bx))
            yvec=np.concatenate((ty,np.zeros(len(edit))))
            pred0=cg._micro_ols_predict(fparent,bparent,yvec,len(train))[-len(edit):]
            pred1=cg._micro_ols_predict(fchild,bchild,yvec,len(train))[-len(edit):]
            oldmargin=np.tanh(10*(pred0[0::2]-pred0[1::2]))
            newmargin=np.tanh(10*(pred1[0::2]-pred1[1::2]))
            if not (np.isfinite(oldmargin).all() and np.isfinite(newmargin).all()):continue
            gain=float(np.mean(newmargin-oldmargin))
            accuracy=float(np.mean(pred1[0::2]>pred1[1::2]))
            stats.best_margin_gain=max(stats.best_margin_gain,gain)
            stats.best_pair_accuracy=max(stats.best_pair_accuracy,accuracy)
            stats.scored+=1
            if gain>min_margin_gain:stats.readout_improved+=1
            # Marker goals can still enter a small representation-learning
            # exploration lane if their pair-dependent *execution* is useful.
            elif not allow_structural or p.goal_kind=='consumer':continue
            else:stats.structural_only+=1
            s=gain+0.015*min(1.,float(np.mean(np.maximum(0,stage_diff))))
            s+=0.04*min(1.,reader_diff)
            s-=0.003*p.depth
            if gain<=min_margin_gain:s-=0.03
            candidates.append((s,gain,current,child))
    candidates.sort(key=lambda k:(-k[0],-k[1]))
    selected=[];used=set();structural_budget=max(1,count//4)
    structural_count=0
    for _,gain,p,child in candidates:
        key=(p.origin_case,p.anchors,p.goal_kind,p.consumer_index)
        if key in used:continue
        if gain<=min_margin_gain:
            if structural_count>=structural_budget:continue
            structural_count+=1
        used.add(key);selected.append(child)
        if len(selected)>=count:break
    stats.adopted=len(selected)
    return selected,stats
