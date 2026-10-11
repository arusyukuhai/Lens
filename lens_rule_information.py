"""Information-directed rule attribution for Lens/Replacer.

Distinguishes three things which should NOT be collapsed into weight magnitude:
  1. Predictive contribution on frozen firing features after OLS refit.
  2. Pairwise phi / coactivation, which is not a causal linkage objective.
  3. Possible producer-to-consumer dependencies in the sequential rewrite program.

All TRAINING-guided signals exclude the last entire source case.  Outer heldout
case is neither inspected nor used to rank replacement locations.
"""
from __future__ import annotations
from dataclasses import dataclass
import csv
from pathlib import Path
import numpy as np


@dataclass
class InformationReport:
    importance: np.ndarray
    replacement_cost: np.ndarray
    bridge_cost: np.ndarray
    firing: np.ndarray
    ids: np.ndarray
    phi: np.ndarray
    train_mse: float
    diagnostic: str = 'frozen-feature OLS refit; not a causal program ablation'


def _linear_fit_mse(z, y, baseline):
    # Training-only OLS with intercept; no Ridge, no rank transform.
    y = np.asarray(y, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if z.shape[1] == 0:
        residual = y-y.mean()
    else:
        zz = np.asarray(z, dtype=np.float64)
        xc = zz-zz.mean(axis=0)
        yc = y-y.mean()
        beta = np.linalg.lstsq(xc, yc, rcond=1e-7)[0]
        residual = yc - xc @ beta
    return float(np.mean(residual**2))


def _bridge_risk(rules, predictive_importance):
    """Conservative static protection for rule-produced literal tokens.

    This is intentionally not a causal attribution.  It protects potential
    downstream consumers, even if the producer itself has zero readout weight.
    Reports this as 'bridge cost', not as predictive contribution.
    """
    n = len(rules)
    consumers = {}
    # Map each input literal token to its later consumers.
    for i,r in enumerate(rules):
        for token in set(v for v in r.pattern if isinstance(v,int) and v>=0):
            consumers.setdefault(token, []).append(i)
    scores = np.asarray(predictive_importance, dtype=np.float64).copy()
    bridges = np.zeros(n, dtype=np.float64)
    # Backward propagation through the sequential pipeline.
    for i in range(n-1,-1,-1):
        # Ignore substitution operands (-1, -2, ...) to avoid pretending the
        # static approximation understands their runtime generated tokens.
        seen=set()
        for tok in set(v for v in rules[i].replacement if isinstance(v,int) and v>=0):
            for j in consumers.get(tok,()):
                if j<=i or j in seen: continue
                seen.add(j)
                # A decayed pass approximates reachability; do not inflate to
                # infinity for cyclic multi-pass interactions.
                bridges[i] = max(bridges[i], .6*scores[j])
        scores[i] += bridges[i]
    return bridges


def analyze(genome, features, baseline, target, case_ids, max_pairs=32, max_refits=48):
    x=np.asarray(features, dtype=np.float64)
    b=np.asarray(baseline,dtype=np.float64)
    y=np.asarray(target,dtype=np.float64)
    cases=np.asarray(case_ids)
    if x.ndim != 2 or x.shape[0]!=len(y) or len(y)!=len(b):
        raise ValueError('feature/target dimension mismatch')
    n,p=x.shape
    if p!=len(genome.rules):
        raise ValueError('expected one feature column per rule')
    train = (cases!=np.max(cases)) if len(np.unique(cases))>1 else np.ones(n, dtype=bool)
    xx, yy, bb=x[train],y[train],b[train]
    weights=np.asarray(genome.readout_weights, dtype=np.float64)
    if len(weights)!=p:
        weights=np.zeros(p,dtype=np.float64)
    firing=(xx!=0)
    freq=firing.mean(axis=0) if len(xx) else np.zeros(p)
    active=np.flatnonzero((np.abs(weights)>1e-12) & (xx.var(axis=0)>1e-12))
    if len(active)>max_refits:
        active=active[np.argsort(-np.abs(weights[active]))[:max_refits]]
    full=_linear_fit_mse(xx[:,active],yy,bb)
    contribution=np.zeros(p, dtype=np.float64)
    for i, r in enumerate(active):
        without=np.concatenate((active[:i],active[i+1:]))
        reduced=_linear_fit_mse(xx[:,without], yy, bb)
        contribution[r]=max(0.0,reduced-full)
    bridges=_bridge_risk(genome.rules, contribution)
    costs=contribution+bridges

    # Map covers useful low-frequency predictors and representative firing rules;
    # no phi regularizer is imposed.  Phi = Pearson of Bernoulli firing flags.
    eligible=np.flatnonzero((freq>0.0)&(freq<1.0))
    if len(eligible):
        ranking=np.lexsort((-freq[eligible], -costs[eligible]))
        ids=eligible[ranking[:max(2,int(max_pairs))]]
    else:
        ids=np.empty(0,dtype=np.int32)
    if len(ids):
        a=firing[:,ids].astype(np.float64)
        a-=a.mean(axis=0)
        den=np.sqrt((a*a).sum(axis=0))
        phi=a.T@a/np.maximum(1e-15,den[:,None]*den[None,:])
        np.clip(phi,-1.0,1.0,out=phi)
    else:
        phi=np.zeros((0,0),dtype=np.float64)
    return InformationReport(contribution,costs,bridges,freq,ids,phi,full)


def replacement_ids(report, count):
    # Lowest forecasted damage first, but don't force all slots with nonzero
    # contribution out of play. An inactive rule can still have bridge cost.
    cost=report.replacement_cost
    return [int(i) for i in np.argsort(cost,kind='stable')[:min(len(cost),max(0,int(count)))]]


def intervention_ids(report, limit):
    """Diagnostic priority: strongest unique information, bridge, weak/unused.

    Selection uses training originals only. Actual ablation may inspect the
    held-out original but MUST NOT feed its result into replacement choices.
    """
    if limit <= 0:return []
    choices=[]
    for order in (np.argsort(-report.importance),
                  np.argsort(-report.bridge_cost),
                  np.argsort(report.replacement_cost)):
        for i in order:
            j=int(i)
            if j not in choices and (len(choices)<limit):
                choices.append(j)
                break
    return choices[:limit]

def export(report, prefix, generation, save_plot=True):
    base=Path(prefix)
    base.parent.mkdir(parents=True,exist_ok=True)
    stem=f'{base.name}_g{generation:06d}'
    root=base.parent
    imp_path=root/(stem+'_importance.csv')
    with imp_path.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f)
        w.writerow(['rule','train_refit_mse_increase','potential_bridge_risk','replacement_cost','firing_rate'])
        for j in range(len(report.importance)):
            w.writerow([j,float(report.importance[j]),float(report.bridge_cost[j]),float(report.replacement_cost[j]),float(report.firing[j])])
    phi_path=root/(stem+'_phi.csv')
    with phi_path.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f)
        w.writerow(['rule']+[int(i) for i in report.ids])
        for ri,line in zip(report.ids,report.phi):
            w.writerow([int(ri)]+[float(c) for c in line])
    img_path=None
    if save_plot and len(report.ids):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(7.0,6.2))
            pic=ax.imshow(report.phi,vmin=-1,vmax=1,cmap='coolwarm')
            labels=[str(int(i)) for i in report.ids]
            ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=6)
            ax.set_yticks(range(len(labels)),labels,fontsize=6)
            ax.set_title('Rule firing phi (not an optimization target)')
            fig.colorbar(pic,ax=ax,label='phi')
            fig.tight_layout()
            img_path=root/(stem+'_phi.png')
            fig.savefig(img_path,dpi=140)
            plt.close(fig)
        except ImportError:
            pass
    return imp_path,phi_path,img_path
