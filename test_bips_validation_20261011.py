"""Tests for source-disjoint BIPS proposal construction and prediction-aware CEGAR."""
from types import SimpleNamespace
import numpy as np
import main as m
import lens_bips as bp
import lens_bips_cegar as cegar


def _rolling_three():
    originals = [b'abcdefghijABCDEFGHIJ', b'abcdefghijKLMNOPQRST',
                 b'holdout-should-never-be-observed']
    slots = []
    for text in originals:
        damaged=[]
        for j in range(1,11):
            row = bytearray(text)
            row[2+j%4] = ord('!')
            damaged.append(list(row))
        slots.append(SimpleNamespace(inputs=[list(text)] + damaged,
                                     target=np.linspace(1., .1, 11)))
    return SimpleNamespace(slots=slots)


def test_crossfit_anchor_mining_separates_source_originals():
    rolling = _rolling_three()
    g=m.Genome([m.Rule([255],[255]) for _ in range(16)], readout_weights=[0.]*16)
    feats=np.zeros((33,16),dtype=np.float32)
    baseline=np.zeros(33)
    target=np.concatenate([x.target for x in rolling.slots])
    one=bp.mine_paths(rolling,g,feats,baseline,target,
                     max_depth=4,max_proposals=24,max_sources=16,crossfit=True)
    assert one and all(p.origin_case in (0,1) for p in one)
    # Leave-one-original-out: changing source 1 or the final heldout cannot
    # change proposals synthesized strictly from source 0.
    source_zero=[p for p in one if p.origin_case==0]
    assert source_zero
    rolling.slots[1].inputs=[list(b'ZZZZZZZZZZZZZZZZZZZZ') for _ in range(11)]
    rolling.slots[1].target=np.linspace(.9,0.0,11)
    rolling.slots[2].inputs=[list(b'new-unseen-original') for _ in range(11)]
    rolling.slots[2].target=np.linspace(.01,.9,11)
    target=np.concatenate([x.target for x in rolling.slots])
    changed=bp.mine_paths(rolling,g,feats,baseline,target,
                          max_depth=4,max_proposals=24,max_sources=16,crossfit=True)
    assert source_zero == [p for p in changed if p.origin_case==0]


def test_heldout_validation_rejects_anti_generalizing_rule():
    tr=np.arange(4)
    va=np.arange(4,8)
    x=np.zeros((8,2),dtype=np.float64)
    base=np.zeros(8)
    weights=np.zeros(2)
    z=np.array([0,0,1,1,0,0,1,1],dtype=float)
    child=x.copy();child[:,1]=z
    # Fits training original perfectly, but reverses correlation on another.
    targets=np.array([0,0,1,1,1,1,0,0],dtype=float)
    res=cegar._validate_concrete_candidate(tr,va,child,base,x,base,weights,targets,0,2)
    assert res is not None and res[0] < 0 and res[1] < 0
    targets[va]=[0,0,1,1]
    res=cegar._validate_concrete_candidate(tr,va,child,base,x,base,weights,targets,0,2)
    assert res is not None and res[0] > 0 and res[1] > 0


def test_grouped_probes_hold_out_actual_original():
    p=bp.PathProposal(1.0,b'ab',b'cd',origin_case=1)
    rows=[b'abXYZcd',b'none',b'abcd',b'none2', b'ab22cd',b'nothing',b'abcd',b'none3']
    groups=np.array([0]*4+[1]*4)
    tr,va,indep=cegar._grouped_probes(p,rows,np.ones(8),groups,8)
    assert indep
    assert tr and va and set(groups[tr])=={1} and set(groups[va])=={0}
    assert set(tr).isdisjoint(va)


def test_one_byte_bips_anchor_preserves_native_parity():
    import native_cpu_replacer as native
    g=m.Genome([m.Rule([255],[255]) for _ in range(5)])
    p=bp.PathProposal(1.0,b'a',b'c')
    rules=bp.compile_path(m,g,p,bp.available_latent_tokens(g,2))
    child=m.clone_genome_shallow(g)
    child.rules[1:3]=rules
    seq=[list(b'abbbbbc'),list(b'xayc'),list(b'zzz')]
    a=m.trajectory_features_cpu(seq,child,100,hidden_rewrite_phases=2)
    b=native.trajectory_features_native(seq,child,100,hidden_rewrite_phases=2)
    np.testing.assert_array_equal(a[0],b[0])
    assert a[0][0,2] > 0
