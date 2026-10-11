"""Regression and concrete semantics tests for BIPS causal/adaptive search."""
from types import SimpleNamespace
import numpy as np
import main as mod
import lens_bips as bp
import lens_bips_cegar as cg


def _originals():
    texts = [b'abXXcd', b'nothing', b'abYYcd', b'nothing else',
             b'abZZcd', b'elsewhere', b'abDDcd', b'missing']
    return SimpleNamespace(slots=[SimpleNamespace(inputs=[list(x) for x in texts])
                                  for _ in range(3)])


def _fixture():
    genome = mod.Genome([mod.Rule([255], [255]) for _ in range(10)],
                        readout_weights=[0.]*10)
    rolling = _originals()
    f = np.zeros((24, 10), dtype=np.float64)
    baseline = np.zeros(24)
    target = np.tile([1., 0., 1., 0., 1., 0., 1., 0.], 3)
    proposal = bp.PathProposal(1.0, b'ab', b'cd')
    return genome, rolling, f, baseline, target, proposal


def test_causal_restore_each_stage_measures_whole_chain():
    import native_cpu_replacer as native
    genome, rolling, *_rest, proposal = _fixture()
    child = cg._make_child(mod, genome, proposal, 3)
    assert child is not None
    rows = [list(x) for x in rolling.slots[0].inputs] * 2
    y = np.tile([1., 0., 1., 0., 1., 0., 1., 0.], 2)
    f, b, *_ = native.trajectory_features_native(rows, child, 120, 2, 2)
    assert (f[:, 4] > 0).any()
    gains, loss = cg._intervention_stage_gains(
        mod, genome, child, 3, 2, rows, y, 8, 120, 2, 2,
        full_feat=f, full_baseline=b)
    assert np.isfinite(gains).all() and len(gains) == 2
    # The first stage carries predictive content, whereas the second stage
    # merely relays a marker in this toy case: detect the redundancy.
    assert gains[0] > 0.10 and abs(gains[1]) < 1e-6, (gains, loss)


def test_candidate_key_stable_across_internal_repairs():
    proposal = bp.PathProposal(1.0, b'ab', b'cd', origin_case=0)
    repaired = bp.PathProposal(1.0, b'ab', b'cd', origin_case=0,
                               internal_anchors=((12,), (13,)))
    assert proposal.path_key != repaired.path_key
    assert cg._proposal_identity(proposal) == cg._proposal_identity(repaired)


def test_keep_best_placement_instead_of_later_worse_one(monkeypatch):
    genome, rolling, f, b, y, proposal = _fixture()
    monkeypatch.setattr(cg, '_path_children', lambda *a, **kw: iter([(proposal, 0), (proposal, 4)]))
    monkeypatch.setattr(cg, '_repair_path_from_prefix', lambda p,*a: (p, False))
    monkeypatch.setattr(cg, '_repair_chain_bridges', lambda m,g,p,at,*a: (p, 0, True))
    monkeypatch.setattr(cg, '_validate_concrete_candidate',
                        lambda tr,va,ft,ba,xf,xb,w,yy,at,d: (0.7 if at==0 else 0.1, 0.1, 1.0))
    children, stats = cg.refine_path_offspring(
        mod, genome, [proposal], f, b, y, rolling, 2, 120, 2, 2,
        exact_budget=2, probe_samples=8, windows_per_depth=2,
        max_discrepancy=1.0, source_gate=False,
        adaptive_probes=False, causal_budget=0)
    assert stats.concrete_evaluations == 2
    assert len(children) == 1
    # The higher-scoring first placement (row 0) must survive.
    assert children[0].rules[0].pattern == [ord('a'), ord('b')]


def test_adaptive_probe_expansion_bounded_and_not_leaking(monkeypatch):
    genome, rolling, f, b, y, proposal = _fixture()
    monkeypatch.setattr(cg, '_path_children', lambda *a, **kw: iter([(proposal, 0)]))
    monkeypatch.setattr(cg, '_repair_chain_bridges', lambda m,g,p,at,*a: (p,0,True))
    called = []
    def fake_validation(train,val,ft,ba,xf,xb,w,yy,at,depth):
        called.append((len(train),len(val)))
        # The small sample strongly overstates the benefit; the expanded
        # disjoint-original estimate correctly rejects the candidate.
        return (1.2, .01, 1.) if len(train) <= 4 else (-.2,-.1,1.)
    monkeypatch.setattr(cg, '_validate_concrete_candidate', fake_validation)
    result, stats = cg.refine_path_offspring(
        mod, genome, [proposal], f, b, y, rolling, 2, 120, 2, 2,
        exact_budget=3, probe_samples=8, windows_per_depth=1,
        max_discrepancy=1.0, source_gate=False,
        adaptive_probes=True,adaptive_max_probes=16,
        adaptive_max_expansions=1, causal_budget=0,
        exploration_rate=0)
    assert stats.adaptive_expansions == 1
    assert stats.concrete_evaluations == 2
    assert called == [(4,4),(8,8)]
    assert not result
