"""Fresh-row BIPS verification tests. Never use the external population holdout."""
from types import SimpleNamespace
import numpy as np
import lens_bips_cegar as cg


def test_confirmation_rows_exclude_initial_probes_and_other_group():
    groups=np.repeat(np.arange(3),32)
    probe=list(range(7))+list(range(32,39))
    src=SimpleNamespace(origin_case=0)
    train,val=cg._fresh_confirmation_groups(src,probe,groups,48)
    assert len(train)>=8 and len(val)>=8
    assert not (set(probe)&set(train+val))
    assert all(groups[i]==0 for i in train)
    assert all(groups[i]!=0 for i in val)
    assert max(train)<32 and min(val)>=32


def test_confirmation_requires_available_fresh_rows():
    groups=np.repeat([0,1],10)
    src=SimpleNamespace(origin_case=0)
    tr,va=cg._fresh_confirmation_groups(src,list(range(7)),groups,64)
    assert tr==[] and va==[]


def test_confirmation_rejects_initially_amazing_but_reversed_fresh_gain(monkeypatch):
    n=100
    rows=[bytes([65+(i%23)]) for i in range(n)]
    groups=np.array([0]*50+[1]*50)
    x=np.zeros((n,4),dtype=float)
    b=np.zeros(n); y=np.linspace(0,1,n)
    child=object();meta=([0,1,2,3,50,51,52,53],4,10,2,None,None)
    calls=[]
    def fake_features(mod,genome,samples,*other):
        calls.append(len(samples))
        return np.zeros((len(samples),4)),np.zeros(len(samples))
    monkeypatch.setattr(cg,'_concrete_features',fake_features)
    monkeypatch.setattr(cg,'_validate_concrete_candidate',
                        lambda *a:(-0.2,-0.01,0.05))
    stats=cg.RefinementStatistics()
    out=cg._confirm_macro_candidates(None,None,[(1.9,child,meta)],rows,groups,x,b,y,
                512,2,1,budget=1,max_probes=64,min_mse_ratio=.01,
                origin_by_child={id(child):0},stats=stats)
    assert not out and stats.confirmation_rejected==1
    assert stats.confirmation_checked==1 and calls[0]>16


def test_confirmation_can_admit_and_clamps_amazing_pearson(monkeypatch):
    n=100;rows=[bytes([65+(i%23)]) for i in range(n)]
    groups=np.array([0]*50+[1]*50);x=np.zeros((n,4));b=np.zeros(n);y=np.linspace(0,1,n)
    child=object();meta=([0,1,2,3,50,51,52,53],4,10,2,None,None)
    monkeypatch.setattr(cg,'_concrete_features',lambda *args:(np.zeros((64,4)),np.zeros(64)))
    monkeypatch.setattr(cg,'_validate_concrete_candidate',lambda *a:(1.9,.01,.1))
    stats=cg.RefinementStatistics()
    out=cg._confirm_macro_candidates(None,None,[(1.9,child,meta)],rows,groups,x,b,y,
                 512,2,1,budget=1,max_probes=64,min_mse_ratio=.01,
                 origin_by_child={id(child):0},stats=stats)
    assert len(out)==1 and out[0][0]<1.9
    assert stats.confirmation_accepted==1
