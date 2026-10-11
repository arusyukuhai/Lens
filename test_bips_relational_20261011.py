"""Bounded paired-state program synthesis & consumer endpoint regressions."""
from types import SimpleNamespace
import numpy as np
import main as m
import lens_bips as bp
import lens_bips_cegar as cg
import lens_bips_relational as br


def _make_case(text):
    return SimpleNamespace(inputs=[text]+[text.replace(b'x',b'q')]*12+
                           [text.replace(b'x',b'a')]*9,
                           target=[1.]+[.8]*21)


def test_paired_edits_both_change_same_byte_and_only_one_reverts():
    c=_make_case(b'AABxaade')
    pairs=br.paired_edits(c,5)
    assert len(pairs)==1
    correct,wrong,pos=pairs[0]
    assert pos==3 and correct==c.inputs[0]
    assert wrong!=correct and wrong[pos]!=ord('q')
    assert correct[:pos]+correct[pos+1:]==wrong[:pos]+wrong[pos+1:]


def test_consumer_goal_is_executable_without_hidden_memory():
    # Distinct internal tokens -> downstream *pre-existing* consumer row.
    g=m.Genome([m.Rule([300],[300]) for _ in range(12)])
    g.rules[8]=m.Rule([ord('Z')],[ord('Z')])
    p=bp.PathProposal(1.0,b'x',b'de',origin_case=0,goal_kind='consumer',
                      consumer_index=8,consumer_pattern=(ord('Z'),))
    child=cg._make_child(m,g,p,0)
    assert child is not None
    assert all(m.is_rule_nonexpanding(v) for v in child.rules[:2])
    f,_,_,_=m.trajectory_features_cpu([list(b'xaade'),list(b'qaade')],child,128)
    assert f[0,0]>0 and f[0,1]>0 and f[0,8]>0
    assert np.all(f[1,[0,1,8]]==0)
    assert -2 not in child.rules[1].pattern and -3 not in child.rules[1].pattern
    # Ensure parent consumer survives and cannot be overwritten by the chain.
    assert cg._make_child(m,g,p,8) is None


def test_relational_mining_never_uses_population_heldout_original():
    g=m.Genome([m.Rule([300],[300]) for _ in range(16)])
    g.readout_weights=[0.]*16
    slots=[_make_case(b'AABxaade'),_make_case(b'BBQxaade'),
           _make_case(b'PQRsecretZonly')]
    roll=SimpleNamespace(slots=slots)
    f=np.zeros((sum(len(c.inputs) for c in slots),16))
    p=br.mine_relational_paths(roll,g,f,max_proposals=32)
    assert p and all(v.origin_case in (0,1) for v in p)
    assert all(b'secretZonly' not in b''.join(v.anchors) for v in p)
    assert all(v.goal_kind=='marker' for v in p)


def test_relational_gain_on_other_original_and_correct_pair():
    g=m.Genome([m.Rule([300],[300]) for _ in range(16)])
    g.readout_weights=[0.]*16
    slots=[_make_case(b'AABxaade'),_make_case(b'BBQxaade'),_make_case(b'PQRxaade')]
    roll=SimpleNamespace(slots=slots)
    f,b,_,_=m.trajectory_features_cpu(
        [list(s) for c in slots for s in c.inputs],g,128)
    y=np.array([v for c in slots for v in c.target])
    p=br.mine_relational_paths(roll,g,f,max_proposals=24,depth=3)
    children,st=br.refine_relational_offspring(
        m,g,p,f,b,y,roll,3,128,2,2,budget=16,pairs_per_source=3)
    assert st.tested<=16 and st.witness_positive>0
    assert st.readout_improved>0 and st.best_pair_accuracy==1
    assert children and all(any(-1 in rr.pattern for rr in ch.rules) for ch in children)
    assert all(getattr(ch,'_origin_operator','').startswith('bips:') for ch in children)


def test_consumer_mining_requires_positive_existing_readout_and_no_heldout_features():
    g=m.Genome([m.Rule([300],[300]) for _ in range(16)])
    g.rules[14]=m.Rule([ord('Z')],[ord('Z')])
    g.readout_weights=[0.]*14+[2.]+[0.]
    slots=[_make_case(b'AABxaade'),_make_case(b'BBQxaade'),_make_case(b'PQRxaade')]
    roll=SimpleNamespace(slots=slots)
    f=np.zeros((sum(len(sl.inputs) for sl in slots),16)); f[:44,14]=1.
    proposal=br.mine_relational_paths(roll,g,f,max_proposals=40,goal='consumer')
    assert proposal and all(p.goal_kind=='consumer' for p in proposal)
    assert all(p.consumer_index==14 for p in proposal)
    assert all(p.consumer_pattern==(ord('Z'),) for p in proposal)
    # Only the *outer* heldout original has consumer activity -> cannot mine.
    f[:44,14]=0.;f[44:,14]=1.
    assert br.mine_relational_paths(roll,g,f,max_proposals=40,goal='consumer')==[]
