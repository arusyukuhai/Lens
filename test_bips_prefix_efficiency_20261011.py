"""Native first-sweep prefix semantics and source-stage counterexample repair."""
from types import SimpleNamespace
import numpy as np
import main as mod
import lens_bips as bp
import lens_bips_cegar as cg
import native_cpu_replacer as native


def reference_prefix(inputs, genome, stop, max_output=512, phases=2):
    outputs=[]
    ends=mod._hidden_phase_ends(len(genome.rules),phases)
    for s in inputs:
        cur=mod.embed_inputs([s],genome.embedding)[0]
        mem=mod.HiddenMemory(token_capacity=max(1,len(s)))
        for i, rule in enumerate(genome.rules[:stop]):
            cur, _, overflow, _=mod.replace_once_cpu(cur,rule,max_output,mem)
            if overflow:
                break
            if i+1 in ends:
                cur,_,overflow=mod.hidden_rewrite_cpu(cur,mem,max_output)
                if overflow:
                    break
        outputs.append(tuple(cur))
    return outputs


def test_native_exact_boundary_snapshots_plain_hidden_and_special_opcodes():
    g=mod.Genome([
        mod.Rule([ord('a'),ord('b')],[ord('z')]),
        mod.Rule([ord('z'),-2,ord('d')],[ord('z'),-1]),
        mod.Rule([ord('z')],[ord('y')]),
        mod.Rule([ord('x')],[ord('x')]),
        mod.Rule([ord('y')],[ord('t')]),
        mod.Rule([ord('d')],[ord('d')]),
    ])
    inp=[list(b'abQd'),list(b'abXd'),list(b'xyzabc'),list(b'abcd')]
    for phases in (1,2,3):
        for boundary in range(6):
            got=native.prefix_states_native(inp,g,boundary,256,phases,2)
            expected=reference_prefix(inp,g,boundary,256,phases)
            assert got==expected, (phases,boundary,got,expected)


def test_source_stage_resynthesis_after_early_rule_uses_actual_token():
    g=mod.Genome([mod.Rule([ord('a'),ord('b')],[ord('Q')])]
                 +[mod.Rule([255],[255]) for _ in range(8)])
    proposal=bp.PathProposal(1.0,b'ab',b'cd',origin_case=0)
    rows=[b'abXXcd',b'abYYcd',b'no anchors',b'other']
    states=native.prefix_states_native([list(x) for x in rows],g,2,256)
    fixed,did=cg._repair_path_from_prefix(proposal,states,rows,g.embedding,3)
    assert did
    assert fixed.internal_anchors[0]==(ord('Q'),)
    assert fixed.internal_anchors[1]==(ord('c'),ord('d'))
    child=cg._make_child(mod,g,fixed,2)
    assert child is not None
    features,_,_,_=native.trajectory_features_native(
        [list(x) for x in rows],child,256,2,2)
    assert features[0,3] and features[1,3]
    assert not features[2,3]


def test_depth_x_placement_fairness_before_extra_candidates():
    g=mod.Genome([mod.Rule([255],[255]) for _ in range(150)])
    feat=np.zeros((8,150))
    proposals=[bp.PathProposal(1.,b'ab',b'cd'),
               bp.PathProposal(.8,b'ab',b'cd',b'ef'),
               bp.PathProposal(.7,b'ab',b'cd',b'ef',(b'gh',)),
               bp.PathProposal(.3,b'xy',b'zw')]
    schedule=list(cg._path_children(mod,g,feat,proposals,3))
    first=schedule[:9]
    assert len({(p.depth,at) for p,at in first})==9
    assert {p.depth for p,at in first}=={2,3,4}
    assert all(any(at==0 for p,at in first if p.depth==d) for d in (2,3,4))


def test_resynthesis_never_reads_val_source():
    g=mod.Genome([mod.Rule([ord('a'),ord('b')],[ord('Q')])]
                 +[mod.Rule([255],[255]) for _ in range(7)])
    p=bp.PathProposal(1.,b'ab',b'cd',origin_case=0)
    source=[b'abXXcd',b'abYYcd']
    validation=[b'val-abcdbbb',b'val-zzzz']
    s=native.prefix_states_native([list(x) for x in source+validation],g,2,256)
    p0,_=cg._repair_path_from_prefix(p,s,source+validation,g.embedding,2)
    # Modifying *all* validation examples must not change compiled synthesis.
    validation2=[b'anything and everything',b'abcdefxyzz']
    s2=native.prefix_states_native([list(x) for x in source+validation2],g,2,256)
    p1,_=cg._repair_path_from_prefix(p,s2,source+validation2,g.embedding,2)
    assert p0==p1


def test_downstream_anchor_destroyed_by_producer_is_resynthesized():
    g=mod.Genome([mod.Rule([255],[255]) for _ in range(9)])
    # Initial producer 'ab -> M1' also consumes the second 'ab'.
    p=bp.PathProposal(1.,b'ab',b'ab')
    src=[b'abXXabY',b'abYYabZ']
    states=native.prefix_states_native([list(x) for x in src],g,3,100)
    repaired,nfix,proved=cg._repair_chain_bridges(mod,g,p,3,states,src,100)
    assert proved and nfix>=1
    assert repaired.internal_anchors[1]!=(ord('a'),ord('b'))
    child=cg._make_child(mod,g,repaired,3)
    f,*_=native.trajectory_features_native([list(x) for x in src],child,100,2,1)
    assert f[0,4]>0
