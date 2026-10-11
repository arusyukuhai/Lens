"""Bounded counterexample-guided *concrete* refinement for BIPS paths.

The abstract predicate is ordered presence of literal anchors in raw input.
The concrete predicate is execution of the entire modified Replacer genome,
including all prior rules, multiple sweeps and Hidden Memory phases. If an
abstractly feasible chain fails in the real program, its placement is a
counterexample and the search switches to a different rule window. Exact
mini-probe correlations only rank children; ordinary next-generation evaluation
still supplies the actual population fitness. No held-out source is inspected.

This is deliberately NOT a complete/formally sound CEGAR verifier: the
abstraction is a ranking heuristic, the finite probes are not a proof, and no
completeness claim is made for the bounded DSL.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, replace
import numpy as np
import lens_bips as bp


@dataclass
class RefinementStatistics:
    concrete_evaluations: int = 0
    rejected_windows: int = 0
    refined_windows: int = 0
    abstract_false_positives: int = 0
    abstract_false_negatives: int = 0
    exact_accepted: int = 0
    marker_shortages: int = 0
    probed_inputs: int = 0
    best_probe_gain: float = 0.0
    failed_stage_1: int = 0
    failed_later_stage: int = 0
    validation_checked: int = 0
    validation_rejected: int = 0
    validation_accepted: int = 0
    best_heldout_pearson_gain: float = 0.0
    best_heldout_mse_gain: float = 0.0
    probe_case_count: int = 0
    exploration_accepted: int = 0
    prefix_snapshots: int = 0
    rewritten_prefixes: int = 0
    rewritten_bridges: int = 0
    source_chain_proved: int = 0
    source_gate_rejected: int = 0
    source_abstract_hits: int = 0
    selected_front_windows: int = 0
    snapshot_failures: int = 0
    adaptive_expansions: int = 0
    adaptive_extra_rows: int = 0
    causal_macros_checked: int = 0
    causal_stage_interventions: int = 0
    causal_all_stages_helpful: int = 0
    causal_redundant_stages: int = 0
    causal_best_min_stage_gain: float = 0.0
    confirmation_checked: int = 0
    confirmation_accepted: int = 0
    confirmation_rejected: int = 0
    confirmation_insufficient: int = 0
    confirmation_probe_rows: int = 0
    confirmation_best_pearson_gain: float = 0.0
    confirmation_best_mse_gain: float = 0.0
    confirmation_best_mse_gain_ratio: float = 0.0
    validation_no_fires: int = 0
    edit_contrast_pairs: int = 0
    edit_contrast_best_gain: float = 0.0
    edit_contrast_positive: int = 0


def _training_context(rolling, features, baseline, target, genome):
    """Return aligned training rows and ORIGINAL-CHUNK group IDs.

    The final rolling source remains untouched: it is for population fitness,
    NOT candidate generation or CEGAR screening.  Here we can additionally
    reserve the preceding source as an *inner* cross-source validation group.
    """
    slots = rolling.slots[:-1] if len(rolling.slots) >= 2 else rolling.slots
    rows = [bytes(x) for case in slots for x in case.inputs]
    groups = [i for i, case in enumerate(slots) for _ in case.inputs]
    n = min(len(rows), len(target), len(baseline), len(features))
    x = np.asarray(features[:n], dtype=np.float64)
    b = np.asarray(baseline[:n], dtype=np.float64)
    y = np.asarray(target[:n], dtype=np.float64)
    weights = np.asarray(genome.readout_weights, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError('features must be rank two')
    if weights.shape != (x.shape[1],):
        weights = np.zeros(x.shape[1], dtype=np.float64)
    residual = y - (b + x @ weights)
    return rows[:n], np.asarray(groups[:n], dtype=np.int32), x, b, y, weights, residual


def _training_rows(rolling, features, baseline, target, genome):
    rows, groups, x, b, y, w, r = _training_context(
        rolling, features, baseline, target, genome)
    r = r.copy()
    if len(r):
        r -= r.mean()
    return rows, r

def _probes(proposal,rows,residual,cap):
    """Stratify both sides of the abstract predicate, with residual priorities.

    Never draw an example from the final held-out original. The first half
    explicitly tests the path's predicted positives; the rest are negatives.
    """
    if not rows or cap<=0:
        return []
    pos=[];neg=[]
    for i,s in enumerate(rows):
        (pos if bp._ordered_occurrence(s,proposal.anchors) else neg).append(i)
    pos.sort(key=lambda i:-abs(float(residual[i])))
    neg.sort(key=lambda i:-abs(float(residual[i])))
    each=max(1,cap//2)
    chosen=pos[:each]+neg[:each]
    remaining=sorted(set(pos[each:]+neg[each:]),key=lambda i:-abs(float(residual[i])))
    chosen+=remaining[:max(0,cap-len(chosen))]
    return chosen[:cap]


def _grouped_probes(proposal, rows, residual, groups, cap):
    """Balance positive/negative anchor examples across *original chunks*.

    For at least two training originals, the last is held out inside CEGAR.
    No label, prediction, or residual of this inner source is used to fit the
    new feature coefficient.  Only its concrete out-of-source performance is
    used to screen children, before the entirely separate population holdout.
    """
    cap = max(0, int(cap))
    if cap < 4 or len(rows) < 4:
        return [], [], False
    distinct = np.unique(groups)
    if len(distinct) >= 2:
        origin = int(getattr(proposal, 'origin_case', -1))
        if origin in distinct:
            # A crossfit proposal was built using only this original.
            train_pool = np.flatnonzero(groups == origin)
            val_pool = np.flatnonzero(groups != origin)
        else:
            validation_group = distinct[-1]
            train_pool = np.flatnonzero(groups != validation_group)
            val_pool = np.flatnonzero(groups == validation_group)
        independent = True
    else:
        # Limited one-original compatibility. Do NOT claim generalization.
        train_pool = np.arange(0, len(rows), 2)
        val_pool = np.arange(1, len(rows), 2)
        independent = False
    nval = min(len(val_pool), max(2, cap//2))
    ntrain = min(len(train_pool), max(2, cap-nval))
    if ntrain < 2 or nval < 2:
        return [], [], independent
    # Train can use residual importance; validation is independent of residual
    # magnitude to avoid cherry-picking the held-out error tails.
    train_local = _probes(proposal, [rows[i] for i in train_pool],
                          residual[train_pool], ntrain)
    val_local = _probes(proposal, [rows[i] for i in val_pool],
                        np.zeros(len(val_pool), dtype=np.float64), nval)
    return [int(train_pool[i]) for i in train_local], [int(val_pool[i]) for i in val_local], independent


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0
    a = a-a.mean(); b = b-b.mean()
    denom = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a@b / denom) if denom > 1e-12 else 0.0


def _micro_ols_predict(features,baseline,target,ntrain,max_features=6):
    """Refit a bounded *ordinary* linear readout, as in main's OLS scorer.

    Feature selection and coefficient fitting ONLY use the source group. A
    fresh intercept is fitted, and the source-excluded group's Pearson and
    MSE are measured without any refit or selection on its rows.
    """
    x = np.asarray(features,dtype=np.float64)
    b = np.asarray(baseline,dtype=np.float64)
    y = np.asarray(target,dtype=np.float64)
    n=int(ntrain)
    xt=x[:n]
    residual=y[:n]-b[:n]
    means=xt.mean(axis=0)
    xc=xt-means
    yc=residual-residual.mean()
    colnorm=np.sum(xc**2,axis=0)
    cols=np.flatnonzero(colnorm > 1e-10)
    k=min(int(max_features),max(0,n-2),len(cols))
    if k==0:
        return b+float(residual.mean())
    importance=np.abs(xc[:,cols].T@yc)/np.sqrt(colnorm[cols])
    chosen=cols[np.argsort(-importance,kind='stable')[:k]]
    coef=np.linalg.lstsq(xc[:,chosen],yc,rcond=1e-7)[0]
    return b + (x[:,chosen]-means[chosen])@coef+float(residual.mean())


def _validate_concrete_candidate(train_ids, validation_ids, child_feat,
                                 child_baseline, ancestor_feat, ancestor_baseline,
                                 weights, target, at, depth):
    """Source-disjoint **readout-refit** proxy, not last-rule-only covariance.

    Existing untouched feature weights cannot be assumed constant once a
    rewriting chain changes upstream/downstream execution. Refit small OLS
    readouts for BOTH parent and child using exactly the same source subset,
    then compare Pearson and calibration error on another original. This
    better approximates the actual full-population scoring semantics.
    """
    ntrain=len(train_ids)
    if ntrain<3 or len(validation_ids)<3:
        return None
    if not np.isfinite(child_feat).all():
        return None
    parent_pred=_micro_ols_predict(ancestor_feat,ancestor_baseline,target,ntrain)
    candidate_pred=_micro_ols_predict(child_feat,child_baseline,target,ntrain)
    val=slice(ntrain,None)
    pearson_gain=_pearson(target[val],candidate_pred[val])-_pearson(target[val],parent_pred[val])
    old_mse=float(np.mean((target[val]-parent_pred[val])**2))
    new_mse=float(np.mean((target[val]-candidate_pred[val])**2))
    mse_gain=old_mse-new_mse
    if not (math.isfinite(pearson_gain) and math.isfinite(mse_gain)):
        return None
    return pearson_gain,mse_gain,old_mse

def _proposal_identity(proposal):
    """Stable source-path identity across stage repairs and placements.

    The goal is to keep the *best* concrete implementation of a source
    proposal, regardless of where it is placed or how its internal anchors
    are repaired.  Do not lookup using ``path_key`` and save using a different
    tuple; that silently replaced superior children with worse ones.
    """
    return (proposal.anchors, int(proposal.origin_case))


def _heldout_mse(features, baseline, target, ntrain):
    """Ordinary micro-OLS fit on train original; score ONLY validation rows."""
    pred = _micro_ols_predict(features, baseline, target, ntrain)
    return float(np.mean((np.asarray(target)[ntrain:] - pred[ntrain:]) ** 2))


def _intervention_stage_gains(mod, ancestor, child, at, depth, samples,
                              target, ntrain, max_output, phases, cpu_threads,
                              *, full_feat=None, full_baseline=None):
    """Full-semantics rule-level interventions for a cooperating chain.

    Restore exactly one modified row to its antecedent value, *run the whole
    Replacer*, and refit the same OLS proxy using only train originals. Unlike
    a firing-correlation score this captures destructive upstream effects,
    follow-on rewrites and memory interactions. Positive score: removing that
    stage worsens heldout MSE, so the stage contributes to the full macro.
    """
    if full_feat is None:
        full_feat, full_baseline = _concrete_features(
            mod, child, samples, max_output, phases, cpu_threads)
    full_loss = _heldout_mse(full_feat, full_baseline, target, ntrain)
    gains = []
    for stage in range(depth):
        intervention = mod.clone_genome_shallow(child)
        intervention.rules[at + stage] = ancestor.rules[at + stage]
        mod._mark_genome_pack_rows(intervention, [at + stage])
        mod._clear_evaluation_state(intervention)
        fx, bx = _concrete_features(
            mod, intervention, samples, max_output, phases, cpu_threads)
        loss = _heldout_mse(fx, bx, target, ntrain)
        gains.append(loss - full_loss)
    return np.asarray(gains, dtype=np.float64), full_loss


def _rerank_with_interventions(mod, ancestor, candidates, rows, y,
                               max_output, phases, cpu_threads,
                               budget, stats):
    """Conservatively prefer validated macros whose stages are all useful.

    Only intervene on up to ``budget`` leading candidates; the expensive
    execution is deliberately bounded by budget * max path depth. All scoring
    uses train/validation originals already picked by the CEGAR probe; no
    population heldout source may enter this method.
    """
    if not candidates or budget <= 0:
        return candidates
    ranked = sorted(candidates, key=lambda c: -c[0])
    for index, (score, child, meta) in enumerate(ranked[:int(budget)]):
        indices, ntrain, at, depth, fx, bx = meta
        samples = [list(rows[i]) for i in indices]
        targets = np.asarray(y[indices], dtype=np.float64)
        if ntrain < 3 or len(indices) - ntrain < 3:
            continue
        gains, full_loss = _intervention_stage_gains(
            mod, ancestor, child, at, depth, samples, targets, ntrain,
            max_output, phases, cpu_threads, full_feat=fx, full_baseline=bx)
        stats.causal_macros_checked += 1
        stats.causal_stage_interventions += len(gains)
        if not np.isfinite(gains).all():
            continue
        # Normalize to a stable scale. The adjustment never dominates the
        # primary cross-source predictive improvement and doesn't hard-reject
        # a helpful macro because of redundant logic.
        normalized = gains / max(1e-8, abs(full_loss))
        helpful = int(np.count_nonzero(normalized > 1e-5))
        stats.causal_all_stages_helpful += int(helpful == depth)
        stats.causal_redundant_stages += int(depth - helpful)
        minimum = float(np.min(normalized))
        stats.causal_best_min_stage_gain = max(stats.causal_best_min_stage_gain, minimum)
        bounded = np.clip(normalized, -2.0, 2.0)
        adjustment = (0.04 * float(np.mean(np.tanh(bounded)))
                      - 0.02 * float(np.mean(bounded < -0.01)))
        ranked[index] = (score + adjustment, child, meta)
    return sorted(ranked, key=lambda c: -c[0])


def _fresh_confirmation_groups(proposal, screened_ids, groups, cap, min_rows=8):
    """Disjoint *row* check inside the original-disjoint training/validation split.

    The source original must remain the source of the rule proposal.  Separate
    rows from that original are used to refit; different-original rows are
    used for predictions.  Nothing from the population's final heldout source
    is ever accessed.  This is not a statistically untouched corpus across
    all generations, only a deliberately distinct confirmation subset.
    """
    cap=max(0,int(cap)); used=set(map(int,screened_ids))
    origin=int(getattr(proposal,'origin_case',-1))
    all_groups=np.unique(groups)
    if len(all_groups)<2:
        return [],[]
    if origin not in all_groups:
        origin=int(all_groups[0])
    train=[i for i in range(len(groups)) if int(groups[i])==origin and i not in used]
    val=[i for i in range(len(groups)) if int(groups[i])!=origin and i not in used]
    each=min(len(train),len(val),cap//2)
    if each<min_rows:
        return [],[]
    # Prefer a spread across the severity ladder instead of taking the first
    # N adjacent corruptions (which can be near-duplicates).  No labels used.
    def spread(v,count):
        if len(v)<=count:return v
        ix=np.linspace(0,len(v)-1,count).astype(int)
        return [v[i] for i in ix]
    return spread(train,each),spread(val,each)


def _edit_contrast_on_other_original(mod, champion, child,
                                     train_ids, val_ids, rows, groups, y,
                                     x, b, child_features, child_baseline,
                                     max_output, phases, cpu_threads,
                                     max_pairs):
    """Counterfactual byte-edit preferences of the actual inner-GA scorer.

    The rule path is synthesized from the training original.  For every
    other-original example we build a correct one-byte edit and a *wrong*
    one-byte edit at the same location. This tests information needed for
    inference, not merely global corruption severity. Neither the clean
    validation original nor the edited inputs are used to fit coefficients.
    The supervised oracle is used ONLY for this bounded scoring probe.

    This is a soft candidate-ranking diagnostic, not part of the outer OLS
    fitness or the production inference GA; model readout remains OLS.
    """
    if max_pairs <= 0 or not train_ids or not val_ids:
        return None
    validation_groups=sorted(set(int(groups[k]) for k in val_ids))
    edits=[]
    for grp in validation_groups:
        domain=[k for k in range(len(rows)) if int(groups[k])==grp]
        if not domain:
            continue
        clean_idx=max(domain,key=lambda k:float(y[k]))
        if float(y[clean_idx]) < 1.0-1.e-8:
            # No reliable untouched control for this original.
            continue
        clean=rows[clean_idx]
        alternatives=sorted((k for k in val_ids if int(groups[k])==grp),
                            key=lambda k:-float(y[k]))
        seen_positions=set()
        for k in alternatives:
            corrupt=rows[k]
            if len(clean)!=len(corrupt):
                continue
            mismatched=[j for j,(a,c) in enumerate(zip(clean,corrupt)) if a!=c]
            if not mismatched:
                continue
            # Across pairs, cover different positions instead of repeating the
            # same earliest typo.  This is deterministic and label-independent
            # apart from the clean-reference target.
            pos=next((j for j in mismatched if j not in seen_positions),mismatched[len(mismatched)//2])
            if pos in seen_positions:
                continue
            seen_positions.add(pos)
            correct=bytearray(corrupt);correct[pos]=clean[pos]
            wrong=bytearray(corrupt)
            wrong_token=(int(corrupt[pos])+1)%256
            if wrong_token==int(clean[pos]):
                wrong_token=(int(corrupt[pos])+2)%256
            wrong[pos]=wrong_token
            if wrong==correct:
                continue
            edits.extend([list(correct),list(wrong)])
            if len(edits)//2>=max_pairs:
                break
        if len(edits)//2>=max_pairs:
            break
    n_pairs=len(edits)//2
    if not n_pairs:
        return None
    child_edit,child_edit_base=_concrete_features(
        mod,child,edits,max_output,phases,cpu_threads)
    parent_edit,parent_edit_base=_concrete_features(
        mod,champion,edits,max_output,phases,cpu_threads)
    ntrain=len(train_ids)
    train_target=np.asarray(y[train_ids],dtype=np.float64)
    zeros=np.zeros(len(edits),dtype=np.float64)
    child_pred=_micro_ols_predict(
        np.concatenate((child_features[:ntrain],child_edit),axis=0),
        np.concatenate((child_baseline[:ntrain],child_edit_base)),
        np.concatenate((train_target,zeros)),ntrain)[ntrain:]
    parent_pred=_micro_ols_predict(
        np.concatenate((x[train_ids],parent_edit),axis=0),
        np.concatenate((b[train_ids],parent_edit_base)),
        np.concatenate((train_target,zeros)),ntrain)[ntrain:]
    # Smooth bounded preference for correct-vs-incorrect same-position edits.
    # The margin scale is chosen for 0..1 cleanliness predictions.
    good=np.tanh(12.0*(child_pred[0::2]-child_pred[1::2]))
    old=np.tanh(12.0*(parent_pred[0::2]-parent_pred[1::2]))
    if not np.isfinite(good).all() or not np.isfinite(old).all():
        return None
    # 1) does the macro improve the choice over the parent; 2) does it favor
    # actually correct edits? The second term prevents reward for merely
    # reducing an extreme preference for the wrong edit.
    gain=float(np.mean(0.7*(good-old)+0.3*good))
    return gain,n_pairs


def _confirm_macro_candidates(mod, champion, candidates, rows, groups, x, b,
                              y, max_output, phases, cpu_threads,
                              *, budget, max_probes, min_mse_ratio, stats, origin_by_child=None,
                              edit_contrast_weight=0.0, edit_contrast_pairs=3):
    """Bounded independent-row confirmation before a macro is admitted.

    An initial micro Pearson can appear close to +2 because its validation
    set may contain as few as 6-8 correlated noise levels. Candidate ranking
    never uses the confirmation subset before this function; expensive exact
    Native evaluation is spent on a small number of finalists only.
    """
    if budget<=0 or not candidates:
        return candidates
    # Avoid making the macro lane entirely empty when the small corpus cannot
    # provide >=8 new train+validation examples. The mode is an explicit
    # screening upgrade, not an unmarked change to original training fitness.
    ranked=sorted(candidates,key=lambda t:-t[0]); retained=[]
    for idx,(oldscore,child,meta) in enumerate(ranked):
        if idx>=budget:
            # Not confirmed: do not label this as a validated winner. The
            # separate exploration lane can still provide diverse children.
            continue
        used,ntrain,at,depth,_,_=meta
        # Stable origin is stored on the generated child to prevent repaired
        # placements from changing the source group split.
        origin=int((origin_by_child or {}).get(id(child),-1))
        class _Source: pass
        src=_Source();src.origin_case=origin
        train,val=_fresh_confirmation_groups(src,used,groups,max_probes)
        if len(train)<8 or len(val)<8:
            stats.confirmation_insufficient+=1
            # Don't call it *confirmed*; retain to the limited exploration lane
            # only, rather than bypassing the stronger confidence criterion.
            continue
        ids=train+val;ss=[list(rows[k]) for k in ids]
        feat,base=_concrete_features(mod,child,ss,max_output,phases,cpu_threads)
        evidence=_validate_concrete_candidate(train,val,feat,base,
                     x[ids],b[ids],None,y[ids],at,depth)
        stats.confirmation_checked+=1
        stats.confirmation_probe_rows+=len(ids)
        if evidence is None:
            stats.confirmation_rejected+=1
            continue
        gain,msegain,oldmse=evidence
        ratio=msegain/max(1e-10,oldmse)
        if (gain<=0.0 or ratio<float(min_mse_ratio)):
            stats.confirmation_rejected+=1
            continue
        stats.confirmation_accepted+=1
        stats.confirmation_best_pearson_gain=max(
            stats.confirmation_best_pearson_gain,gain)
        stats.confirmation_best_mse_gain=max(
            stats.confirmation_best_mse_gain,msegain)
        stats.confirmation_best_mse_gain_ratio=max(
            stats.confirmation_best_mse_gain_ratio,ratio)
        # Avoid spectacular tiny-probe Pearson gains swamping rankings.
        # Reserve confirmed values for the final ordering, not old gain ~2.
        confirmed_score=(0.65*min(gain,0.5)
                         +0.35*min(max(ratio,0.0),0.2)) / (1+0.04*depth)
        if edit_contrast_weight > 0.0 and edit_contrast_pairs > 0:
            edit_evidence=_edit_contrast_on_other_original(
                mod,champion,child,train,val,rows,groups,y,x,b,feat,base,
                max_output,phases,cpu_threads,int(edit_contrast_pairs))
            if edit_evidence is not None:
                edit_gain,n_edits=edit_evidence
                stats.edit_contrast_pairs+=n_edits
                stats.edit_contrast_best_gain=max(stats.edit_contrast_best_gain,edit_gain)
                stats.edit_contrast_positive+=int(edit_gain>0)
                confirmed_score+=float(edit_contrast_weight)*float(edit_gain)
        # Subsequent causal interventions should use the larger, fresh-row
        # confirmation cohort rather than reverting to the original tiny
        # screening subset (a subtle source of optimistic intervention scores).
        confirmed_meta=(ids,len(train),at,depth,feat,base)
        retained.append((confirmed_score,child,confirmed_meta))
    return sorted(retained,key=lambda t:-t[0])


def _diverse_windows(genome,features,depth,limit):
    """Test widely separated placements, not three adjacent tail windows.

    A chain blocked by an early destructive rewrite cannot be repaired by
    moving it from e.g. row 1497 to row 1496. The sparse model needs positions
    throughout the program: tail, middle, *front*, then additional quantiles.
    """
    n=max(0,len(genome.rules)-depth+1)
    if n==0 or limit<=0:
        return []
    best=bp._candidate_windows(genome,features,depth,limit=1)[0]
    quantiles=[best, (n-1)//2, 0, 3*(n-1)//4, (n-1)//4, n-1]
    for i in range(max(0,int(limit)-len(quantiles))):
        quantiles.append(round((i+1)*(n-1)/(int(limit)+1)))
    out=[]
    for p in quantiles:
        if p not in out:
            out.append(p)
        if len(out)>=limit:
            break
    return out


def _path_children(mod,champion,features,proposals,windows_per_depth=3):
    """Fair (depth × placement) first pass, then candidate diversity.

    Under budget=12 and depth=2,3,4 with 3 windows, all NINE coverage
    cells are visited before a second path at the same window is considered.
    The previous loop exhausted the budget on one/two placements per depth.
    """
    depths=sorted({p.depth for p in proposals})
    by_depth={d:[p for p in proposals if p.depth==d] for d in depths}
    windows={d:_diverse_windows(champion,features,d,windows_per_depth)
             for d in depths}
    max_paths=max((len(v) for v in by_depth.values()),default=0)
    # First pass: each depth's highest-ranked proposal at *every* placement.
    # Later passes: alternative proposals, distributed over placements.
    for index in range(max_paths):
        for win_rank in range(windows_per_depth):
            for depth in depths:
                lane=by_depth[depth]
                if index>=len(lane) or win_rank>=len(windows[depth]):
                    continue
                p=lane[index]; at=windows[depth][win_rank]
                yield p,at


def _make_child(mod,champion,proposal,at):
    if proposal.goal_kind == "consumer" and not (at + proposal.depth <= proposal.consumer_index < len(champion.rules)):
        return None
    markers=bp.available_latent_tokens(champion,proposal.depth,
                     excluding_rule_rows=range(at,at+proposal.depth))
    if len(markers)<proposal.depth:
        return None
    try:
        rules=bp.compile_path(mod,champion,proposal,markers)
    except ValueError:
        return None
    child=mod.clone_genome_shallow(champion)
    for j,r in enumerate(rules):
        child.rules[at+j]=r
    mod._mark_genome_pack_rows(child,range(at,at+proposal.depth))
    mod._clear_evaluation_state(child)
    mod.mark_offspring_origin(child,champion,f'bips:d{proposal.depth}',proposal.depth)
    return child


def _stage_occurs(state, internal_anchors):
    """Necessary witness in the *embedded state immediately before insertion*."""
    seq=tuple(state)
    cursor=0
    for part in internal_anchors:
        n=len(part)
        where=-1
        for j in range(cursor,len(seq)-n+1):
            if seq[j:j+n]==tuple(part):
                where=j;break
        if where<0:
            return False
        cursor=where+n
    return True


def _stage_compile_anchors(proposal,embedding):
    if proposal.internal_anchors:
        return proposal.internal_anchors
    return tuple(tuple(int(embedding[c]) for c in piece) for piece in proposal.anchors)


def _repair_path_from_prefix(proposal, prefix_states, raw_inputs, embedding, train_count):
    """Counterexample-driven path repair using TRAIN-source prefix states only.

    Preserve anchors when they still exist; when an earlier rewrite destroyed
    them, recover tiny (1-2 token) anchors from actual intermediate tokens at
    approximately the same relative position. This never examines validation
    text/labels for constructing a repaired program.
    """
    anchors=_stage_compile_anchors(proposal,embedding)
    if any(_stage_occurs(state,anchors) for state in prefix_states[:train_count]):
        return proposal,False
    # Attempt multiple source witnesses: a path should never be 'repaired'
    # from the held-out original, even if its stage tokens happen to match.
    for state,raw in zip(prefix_states[:train_count],raw_inputs[:train_count]):
        if len(state)<2*len(anchors):
            continue
        raw=bytes(raw)
        starts=[];at=0
        for part in proposal.anchors:
            found=raw.find(part,at)
            if found<0:
                starts=[];break
            starts.append(found)
            at=found+len(part)
        if not starts:
            continue
        picked=[];cursor=0
        for pos,old in zip(starts,anchors):
            # Exact survival first; otherwise choose a new source-stage anchor.
            where=next((i for i in range(cursor,len(state)-len(old)+1)
                        if tuple(state[i:i+len(old)])==old),-1)
            if where>=0:
                piece=old
            else:
                estimate=round(pos*len(state)/max(1,len(raw)))
                pos2=min(max(cursor,estimate),len(state)-1)
                if pos2>=len(state):
                    picked=[];break
                # One token is more transferable than a long accidental match.
                piece=(int(state[pos2]),)
                where=pos2
            if where<cursor:
                picked=[];break
            picked.append(tuple(piece));cursor=where+len(piece)
        if len(picked)==len(anchors) and _stage_occurs(state,picked):
            return replace(proposal,internal_anchors=tuple(picked)), True
    return proposal,False



def _repair_chain_bridges(mod, champion, proposal, at,
                          source_prefix, source_raw, max_output):
    """Refine downstream anchor constraints after concrete upstream rewrites.

    The first-sweep prefix witness alone does not prove a chain: a producer
    may delete a consumer anchor (e.g., the producer also matches the same
    literal downstream). Execute the prospective rules *on source prefixes*
    and replace only the earliest missing consumer anchor with a witnessed
    intermediate token. Never consult a held-out source during synthesis.
    """
    markers=bp.available_latent_tokens(champion,proposal.depth,
                 excluding_rule_rows=range(at,at+proposal.depth))
    if len(markers)<proposal.depth:
        return proposal,0,False
    internal=list(_stage_compile_anchors(proposal,champion.embedding))
    candidates=[(list(st),bytes(raw)) for st,raw in zip(source_prefix,source_raw)
                if st is not None]
    if not candidates:
        return proposal,0,False
    changed=0
    for stage in range(proposal.depth):
        trial=replace(proposal,internal_anchors=tuple(internal))
        try:
            rules=bp.compile_path(mod,champion,trial,markers)
        except ValueError:
            return proposal,0,False
        outcomes=[]
        for current,raw in candidates:
            next_state,matched,overflow,_=mod.replace_once_cpu(
                current,rules[stage],max_output)
            if matched and not overflow:
                outcomes.append((next_state,raw))
        if not outcomes and stage>0 and candidates:
            # Regress the next condition through the ACTUAL preceding rule.
            # Require a downstream literal from the state *after* that rule.
            marker=markers[stage-1]
            for current,raw in candidates:
                if marker not in current:
                    continue
                after=current.index(marker)+1
                if after>=len(current):
                    continue
                original=proposal.anchors[stage]
                raw_at=raw.find(original)
                approx=(round(raw_at*len(current)/max(1,len(raw)))
                        if raw_at>=0 else after)
                # Test both the estimated target region and the closest
                # surviving token following the information-bearing marker.
                poss=[min(len(current)-1,max(after,approx)),after]
                for pos in poss:
                    if current[pos] in markers or current[pos]==marker:
                        continue
                    alt=(int(current[pos]),)
                    if alt==internal[stage]:
                        continue
                    replacement=replace(proposal,internal_anchors=tuple(
                        internal[:stage]+[alt]+internal[stage+1:]))
                    try:
                        new_rules=bp.compile_path(mod,champion,replacement,markers)
                        next_state,matched,overflow,_=mod.replace_once_cpu(
                            current,new_rules[stage],max_output)
                    except ValueError:
                        continue
                    if matched and not overflow:
                        internal[stage]=alt
                        outcomes=[(next_state,raw)]
                        changed+=1
                        break
                if outcomes:
                    break
        if not outcomes:
            return proposal,0,False
        candidates=outcomes
    return replace(proposal,internal_anchors=tuple(internal)),changed,True


def _concrete_features(mod,child,samples,max_output,phases,cpu_threads):
    try:
        import native_cpu_replacer as native
        feat,baseline,*_=native.trajectory_features_native(
            samples,child,max_output,hidden_rewrite_phases=phases,
            cpu_threads=max(1,min(4,int(cpu_threads))),return_states=False)
    except (ImportError, OSError, RuntimeError):
        feat,baseline,*_=mod.trajectory_features_cpu(
            samples,child,max_output,hidden_rewrite_phases=phases)
    return np.asarray(feat,dtype=np.float64), np.asarray(baseline,dtype=np.float64)


def _concrete_path_fires(mod,child,at,depth,samples,max_output,phases,cpu_threads):
    feat, _ = _concrete_features(mod,child,samples,max_output,phases,cpu_threads)
    return np.asarray(feat[:,at:at+depth]>0,dtype=np.bool_)


def _concrete_last_rule_fired(mod,child,at,depth,samples,max_output,phases,cpu_threads):
    return _concrete_path_fires(mod,child,at,depth,samples,max_output,phases,cpu_threads)[:,-1].astype(np.float64)


def refine_path_offspring(mod,champion,proposals,features,baseline,target,rolling,
                          count,max_output,phases,cpu_threads,
                          *, exact_budget=12,probe_samples=12,windows_per_depth=3,
                          max_discrepancy=0.35, min_pearson_gain=0.0,
                          exploration_rate=0.25,source_gate=True,
                          causal_budget=2, adaptive_probes=True,
                          adaptive_max_probes=24, adaptive_max_expansions=2,
                          confirmation_budget=0, confirmation_probes=64,
                          confirmation_min_mse_ratio=0.01,
                          edit_contrast_weight=0.0, edit_contrast_pairs=3,
                          require_validation_firing=False):
    """Source-held-out CEGAR screening with **concrete full-program** effects.

    - Screen stages for real intermediate-message transmission.
    - If concrete != abstract, reject this placement and try another window.
    - Fit a last-feature coefficient on one set of original chunks and judge
      full-child predicted scores on a separate training original.
    - Never touch the population's held-out original. This is NOT a proof of
      generalization and is still subject to candidate-selection effects.
    """
    stats = RefinementStatistics()
    if count<=0 or exact_budget<=0 or not proposals:
        return [], stats
    rows,groups,x,b,y,w,residual = _training_context(
        rolling,features,baseline,target,champion)
    if len(rows)<4:
        return [],stats
    proposals=sorted(proposals,key=lambda p:-p.score)
    depths=sorted(set(p.depth for p in proposals))
    # Use most of the concrete budget to explore *different* sequences before
    # spending the remainder relocating failed sequences. The old formula
    # evaluated just one path per depth under the default 12-evaluation cap.
    per_depth=max(1,math.ceil(int(exact_budget)*0.70/len(depths)))
    proposals=[p for d in depths for p in [q for q in proposals if q.depth==d][:per_depth]]
    tested=set(); best_by_path={}; exploratory_paths={}; path_trials={}; origin_by_child={}
    prefix_cache={}  # (boundary, source row) -> exact embedded first-sweep prefix
    budget=max(0,int(exact_budget))
    for p,at in _path_children(mod,champion,features,proposals,
                               max(1,int(windows_per_depth))):
        if stats.concrete_evaluations>=budget:
            break
        train_ids,val_ids,independent=_grouped_probes(
            p, rows, residual, groups, max(4,int(probe_samples)))
        if len(train_ids)<2 or len(val_ids)<2:
            continue
        indices=train_ids+val_ids
        samples=[list(rows[k]) for k in indices]
        # Native first-sweep snapshots share the *full parent* rule program,
        # not a truncated subset (which changes hidden phase boundaries).
        missing=[i for i in indices if (int(at),i) not in prefix_cache]
        if missing:
            try:
                from native_cpu_replacer import prefix_states_native
                states=prefix_states_native(
                    [list(rows[k]) for k in missing],champion,at,max_output,phases,
                    cpu_threads=max(1,min(int(cpu_threads),4)))
                prefix_cache.update({(int(at),i):st for i,st in zip(missing,states)})
                stats.prefix_snapshots+=len(missing)
            except (ImportError,OSError,RuntimeError):
                stats.snapshot_failures+=1
                prefix_cache.update({(int(at),i):None for i in missing})
        prefix=[prefix_cache[int(at),i] for i in indices]
        if all(st is not None for st in prefix):
            p,repaired=_repair_path_from_prefix(
                p,prefix,[rows[k] for k in indices],champion.embedding,len(train_ids))
            if repaired:
                stats.rewritten_prefixes+=1
            internal=_stage_compile_anchors(p,champion.embedding)
            abstract=np.asarray([_stage_occurs(st,internal) for st in prefix],dtype=bool)
            stats.source_abstract_hits+=int(np.count_nonzero(abstract[:len(train_ids)]))
        else:
            abstract=np.asarray([bp._ordered_occurrence(rows[k],p.anchors)
                                 for k in indices],dtype=bool)
        key=(p.path_key,at)
        if key in tested:
            continue
        tested.add(key)
        # Relocations count against the original proposal, not its freshly
        # reconstructed stage-specific pattern.
        trial_key=_proposal_identity(p)
        if trial_key in path_trials:
            stats.refined_windows+=1
        path_trials[trial_key]=path_trials.get(trial_key,0)+1
        child=_make_child(mod,champion,p,at)
        if child is None:
            stats.marker_shortages+=1
            continue
        if at==0:
            stats.selected_front_windows+=1
        if prefix is not None and all(st is not None for st in prefix):
            repaired_path,nbridge,source_chain=_repair_chain_bridges(
                mod,champion,p,at,prefix[:len(train_ids)],
                [rows[k] for k in train_ids],max_output)
            if nbridge:
                p=repaired_path
                child=_make_child(mod,champion,p,at)
                stats.rewritten_bridges+=nbridge
                # A repaired path must use its new constraints in the abstract
                # witness. Train and validation states are both evaluated,
                # but only the train source informed the actual rewrite.
                new_anchors=_stage_compile_anchors(p,champion.embedding)
                abstract=np.asarray([_stage_occurs(st,new_anchors) for st in prefix],dtype=bool)
            if source_chain:
                stats.source_chain_proved+=1
            elif source_gate:
                # First-sweep source-chain proof is a bounded preflight, NOT
                # proof of invalidity in later rounds. Disable the gate for
                # completeness experiments. Skip only expensive full-genome
                # probes, do not consume the CEGAR exact-evaluation budget.
                stats.source_gate_rejected+=1
                continue
        if child is None:
            stats.marker_shortages+=1
            continue
        feat,child_baseline=_concrete_features(
            mod,child,samples,max_output,phases,cpu_threads)
        stages=np.asarray(feat[:,at:at+p.depth]>0,dtype=bool)
        actual=stages[:,-1]
        if require_validation_firing and not actual[len(train_ids):].any():
            # A chain which only fires on the source text is extremely likely
            # to memorize its specific n-grams, not transmit transferable info.
            stats.validation_no_fires+=1
            stats.concrete_evaluations+=1
            stats.probed_inputs+=len(indices)
            stats.rejected_windows+=1
            continue
        stats.concrete_evaluations+=1
        stats.probed_inputs+=len(indices)
        stats.probe_case_count=max(stats.probe_case_count,len(set(groups[indices])))
        fn=int(np.count_nonzero(abstract & ~actual))
        fp=int(np.count_nonzero(~abstract & actual))
        stats.abstract_false_positives+=fn
        stats.abstract_false_negatives+=fp
        if fn:
            blocked=abstract & ~actual
            stats.failed_stage_1+=int(np.count_nonzero(blocked & ~stages[:,0]))
            stats.failed_later_stage+=int(np.count_nonzero(blocked & stages[:,0]))
        mismatch=(fn+fp)/len(indices)
        if mismatch>float(max_discrepancy) or not actual.any() or actual.all():
            stats.rejected_windows+=1
            continue
        evidence=_validate_concrete_candidate(
            train_ids,val_ids,feat,child_baseline,x[indices],b[indices],w,y[indices],at,p.depth)
        # Progressive sampling: a tiny probe can produce spectacular but
        # unstable Pearson gains (near 2.0). Spend at most a few *extra* Native
        # evaluations on promising/ambiguous cases and verify them with more
        # rows of the same disjoint source groups, before deciding admission.
        if (adaptive_probes and evidence is not None and
            stats.adaptive_expansions < max(0, int(adaptive_max_expansions)) and
            stats.concrete_evaluations < budget):
            gain0, mse0, _ = evidence
            ambiguous = (gain0 > 0.65 or
                         (gain0 > float(min_pearson_gain) and
                          gain0 - float(min_pearson_gain) < 0.10 and mse0 > 0))
            if ambiguous:
                large_train, large_val, _ = _grouped_probes(
                    p, rows, residual, groups,
                    max(int(probe_samples) + 4, int(adaptive_max_probes)))
                large_indices = large_train + large_val
                if (len(large_train) >= 3 and len(large_val) >= 3 and
                    len(large_indices) > len(indices)):
                    big_samples = [list(rows[k]) for k in large_indices]
                    big_f, big_b = _concrete_features(
                        mod, child, big_samples, max_output, phases, cpu_threads)
                    big_evidence = _validate_concrete_candidate(
                        large_train, large_val, big_f, big_b,
                        x[large_indices], b[large_indices], w,
                        y[large_indices], at, p.depth)
                    stats.concrete_evaluations += 1
                    stats.adaptive_expansions += 1
                    stats.adaptive_extra_rows += len(large_indices)
                    stats.probed_inputs += len(large_indices)
                    stats.probe_case_count = max(
                        stats.probe_case_count, len(set(groups[large_indices])))
                    # Conservative: an attempted independent recheck which
                    # cannot fit the OLS proxy is not an accept.
                    evidence = big_evidence
                    indices, train_ids, val_ids = large_indices, large_train, large_val
                    feat, child_baseline = big_f, big_b
        stats.validation_checked+=1
        if evidence is None:
            stats.validation_rejected+=1
            stats.rejected_windows+=1
            continue
        pearson_gain,mse_gain,prior_mse=evidence
        # When an entire training original is available for validation,
        # require improvement on THAT source instead of rewarding training
        # residual alignment. With only one source, reject sign-reversed
        # Pearson even though the row split cannot certify generalization.
        if pearson_gain<=float(min_pearson_gain) or mse_gain<=0.0:
            stats.validation_rejected+=1
            # An underpowered mini-probe is not a valid proof that a macro is
            # useless: still reserve a SMALL explicit exploration lane for
            # concrete working paths without strong evidence of harm.
            if (exploration_rate>0 and pearson_gain>=-0.20 and
                mse_gain >= -0.30*max(1e-9,prior_mse)):
                explored_score=(pearson_gain+0.20+0.05*min(1.,max(0.,mse_gain)/max(1e-9,prior_mse))) * \
                               (1.0-0.55*mismatch)/(1.0+0.04*p.depth)
                path_id = _proposal_identity(p)
                prior_explore = exploratory_paths.get(path_id)
                if prior_explore is None or explored_score > prior_explore[0]:
                    exploratory_paths[path_id] = (
                        explored_score, child,
                        (indices, len(train_ids), at, p.depth, feat, child_baseline))
            stats.rejected_windows+=1
            continue
        stats.validation_accepted+=1
        stats.best_heldout_pearson_gain=max(stats.best_heldout_pearson_gain,pearson_gain)
        stats.best_heldout_mse_gain=max(stats.best_heldout_mse_gain,mse_gain)
        # Favor stable out-of-source correlation; acknowledge cost, overfitting
        # risk from repeated selection, and discrepancy from the abstraction.
        score=(pearson_gain + 0.1*min(1.,mse_gain/max(1e-9,prior_mse))) * \
              (1.0-0.55*mismatch) / (1.0+0.04*p.depth)
        if not independent:
            score*=0.5
        path_id = _proposal_identity(p)
        origin_by_child[id(child)]=int(p.origin_case)
        prior = best_by_path.get(path_id)
        if prior is None or score > prior[0]:
            best_by_path[path_id] = (
                score, child,
                (indices, len(train_ids), at, p.depth, feat, child_baseline))
        stats.best_probe_gain=max(stats.best_probe_gain,score)
    winners = _confirm_macro_candidates(
        mod, champion, list(best_by_path.values()), rows, groups, x, b, y,
        max_output, phases, cpu_threads,
        budget=max(0,int(confirmation_budget)),
        max_probes=max(0,int(confirmation_probes)),
        min_mse_ratio=max(0.,float(confirmation_min_mse_ratio)),stats=stats,
        origin_by_child=origin_by_child,
        edit_contrast_weight=max(0.,float(edit_contrast_weight)),
        edit_contrast_pairs=max(0,int(edit_contrast_pairs)))
    winners = _rerank_with_interventions(
        mod, champion, winners, rows, y,
        max_output, phases, cpu_threads, causal_budget, stats)
    explore = sorted((v for k,v in exploratory_paths.items() if k not in best_by_path),
                     key=lambda v:-v[0])
    slots=min(count, max(0,round(count*float(exploration_rate))))
    if explore and exploration_rate>0 and count>=1:
        slots=max(1,slots)
    selected=winners[:max(0,count-slots)]
    # Give back unused exploration slots rather than shrinking the exploitation
    # lane when CEGAR finds no candidates eligible for exploration.
    if len(explore)<slots:
        selected=winners[:max(0,count-len(explore))]
    selected_explore=explore[:max(0,min(slots,count-len(selected)))]
    result=[child for _,child,_ in selected+selected_explore]
    stats.exploration_accepted=len(selected_explore)
    stats.exact_accepted=len(result)
    return result,stats
