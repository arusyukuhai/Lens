"""Bounded symbolic paths, concrete counterexamples, source-level isolation."""
from types import SimpleNamespace
import numpy as np
import main as m
import lens_bips as bp
import lens_bips_cegar as cg


def test_four_stage_markers_and_nonexpanding_cpu_native():
    g=m.Genome([m.Rule([255],[255]) for _ in range(8)])
    p=bp.PathProposal(2.0,b'ab',b'gh',b'cd',(b'ef',))
    assert p.depth==4
    assert p.anchors==(b'ab',b'cd',b'ef',b'gh')
    markers=bp.available_latent_tokens(g,4)
    chain=bp.compile_path(m,g,p,markers)
    assert len(chain)==4
    assert all(m.is_rule_nonexpanding(r) for r in chain)
    assert all(-2 not in r.pattern and -3 not in r.pattern for r in chain)
    state=m.embed_inputs([b'ab11cd22ef33gh!'],g.embedding)[0]
    for rule in chain:
        state,matched,overflow,changed=m.replace_once_cpu(state,rule,150)
        assert matched and not overflow
    assert state[0]==markers[-1]
    child=m.clone_genome_shallow(g)
    child.rules[1:5]=chain
    samples=[list(b'ab11cd22ef33gh!'), list(b'abcd ef gh'), list(b'xyzxyz')]
    a=m.trajectory_features_cpu(samples,child,150,hidden_rewrite_phases=2)
    import native_cpu_replacer as native
    b=native.trajectory_features_native(samples,child,150,hidden_rewrite_phases=2,cpu_threads=2)
    np.testing.assert_array_equal(a[0],b[0])
    assert a[0][0,4]>0


def test_counterexample_detection_catches_interfering_early_rule():
    # The abstract anchored sequence exists, but another earlier rule destroys
    # the intermediate sink before the private-marker reader can observe it.
    g=m.Genome([m.Rule([ord('c'),ord('d')],[ord('c')])]
               +[m.Rule([255],[255]) for _ in range(8)])
    p=bp.PathProposal(2.0,b'ab',b'cd')
    assert bp._ordered_occurrence(b'abXXcd',p.anchors)
    markers=bp.available_latent_tokens(g,2,excluding_rule_rows=(4,5))
    path=bp.compile_path(m,g,p,markers)
    child=m.clone_genome_shallow(g)
    child.rules[4:6]=path
    samples=[list(b'abXXcd'),list(b'xyz'),list(b'abYYcd')]
    actual=cg._concrete_last_rule_fired(m,child,4,2,samples,100,2,2)
    assert actual.tolist()==[0.,0.,0.]
    # Move path to before the interfering rule; the same goal now works.
    child2=m.clone_genome_shallow(g)
    child2.rules[0:2]=path
    # This displaces the original interfering rule; it is a different window.
    actual2=cg._concrete_last_rule_fired(m,child2,0,2,samples,100,2,2)
    assert actual2[0]>0 and actual2[2]>0


def _example():
    raw=[b'abXYcdend',b'abYYcdend',b'hello',b'abZZcdend',b'unrelated']
    slots=[SimpleNamespace(inputs=[list(x) for x in raw],target=np.array([.0,.2,.8,.1,.95])),
           SimpleNamespace(inputs=[list(x) for x in raw],target=np.array([.2,.1,.7,.05,.8])),
           SimpleNamespace(inputs=[list(b'holdoutabXXcd')]*5,target=np.array([.2]*5))]
    g=m.Genome([m.Rule([255],[255]) for _ in range(12)],readout_weights=[0.]*12)
    f=np.zeros((15,12),dtype=np.float32)
    base=np.zeros(15)
    y=np.concatenate([c.target for c in slots])
    return SimpleNamespace(slots=slots),g,f,base,y


def test_source_level_holdout_never_in_cegar_probes():
    roll,g,f,base,y=_example()
    rows,r=cg._training_rows(roll,f,base,y,g)
    assert len(rows)==10 and b'holdout' not in b' '.join(rows)
    p=[bp.PathProposal(1.0,b'ab',b'cd')]
    x,s=cg.refine_path_offspring(m,g,p,f,base,y,roll,3,100,2,2,
                                 exact_budget=6,probe_samples=8,windows_per_depth=2)
    assert s.concrete_evaluations<=6
    assert s.probed_inputs<=6*8
    assert all(child._origin_operator=='bips:d2' for child in x)
    # Changing held-out labels/strings must not alter the training-guided probe.
    roll.slots[-1].inputs=[list(b'completelynew')]*5
    roll.slots[-1].target=np.ones(5)
    y[-5:]=.9
    x2,s2=cg.refine_path_offspring(m,g,p,f,base,y,roll,3,100,2,2,
                                   exact_budget=6,probe_samples=8,windows_per_depth=2)
    assert s2==s
    assert [z.as_program() for z in x2]==[z.as_program() for z in x]


def test_four_depth_source_mining():
    # a corrupted byte precedes all three downstream requirement anchors.
    clean=b'abcdefghij1234567890'
    noisy=bytearray(clean);noisy[2]=ord('Q')
    cases=[SimpleNamespace(inputs=[list(clean),list(noisy),list(noisy)],
                           target=np.asarray([1.,.4,.4])) for _ in range(3)]
    g=m.Genome([m.Rule([255],[255]) for _ in range(16)],readout_weights=[0.]*16)
    f=np.zeros((9,16),dtype=np.float32)
    b=np.zeros(9)
    proposals=bp.mine_paths(SimpleNamespace(slots=cases),g,f,b,
        np.concatenate([c.target for c in cases]),max_proposals=30,max_sources=12,max_depth=4)
    assert any(p.depth==4 for p in proposals)


def test_counterexamples_refine_rule_placement_not_just_proxy_score():
    seq=[b'abXXcd',b'no anchors',b'abYYcd',b'no anchors again']
    slots=[SimpleNamespace(inputs=[list(x) for x in seq],
                           target=np.array([0.,.95,.1,.85])) for _ in range(3)]
    g=m.Genome([m.Rule([ord('c'),ord('d')],[ord('c')])]
               +[m.Rule([255],[255]) for _ in range(9)],
               readout_weights=[0.]*10)
    f=np.zeros((12,10),dtype=np.float32);base=np.zeros(12)
    y=np.concatenate([c.target for c in slots])
    proposal=bp.PathProposal(3.0,b'ab',b'cd')
    children,stats=cg.refine_path_offspring(
        m,g,[proposal],f,base,y,SimpleNamespace(slots=slots),
        count=2,max_output=100,phases=2,cpu_threads=2,
        exact_budget=5,probe_samples=8,windows_per_depth=3)
    assert stats.concrete_evaluations==3
    # Source-stage repair can now fix the early destructive rewrite instead
    # of repeatedly rejecting the same raw-input anchor.
    assert stats.rewritten_prefixes>=1
    assert stats.prefix_snapshots>=8
    assert stats.selected_front_windows>=1
    assert children
    # With score-preserving placement selection, the best window need not be
    # the front window. Verify that the repaired chain survives selection.
    assert any(any(r.pattern == [ord('a'), ord('b')] for r in c.rules)
               for c in children)
