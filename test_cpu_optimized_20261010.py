"""CPU optimization differential and performance-safety regression coverage."""
import random
import numpy as np
import pytest
import main
import native_cpu_replacer as native


def model(seed, rules, alphabet, wildcard_rate=0.02):
    rng=random.Random(seed)
    result=[]
    for i in range(rules):
        x=rng.randrange(alphabet)
        y=rng.randrange(alphabet)
        z=rng.randrange(alphabet)
        typ=rng.randrange(12)
        if typ<=3:
            result.append(main.Rule([x,y], [z]))
        elif typ<=6:
            result.append(main.Rule([x], [y]))
        elif typ<=8:
            result.append(main.Rule([x,y,z],[x]))
        elif typ==9:
            result.append(main.Rule([x,-1,y],[y,-1]))
        elif typ==10:
            result.append(main.Rule([-2,x,-2], [-128,-1]))
        else:
            result.append(main.Rule([x,x],[x]))
    return main.Genome(result)

@pytest.mark.parametrize('seed',range(6))
@pytest.mark.parametrize('alphabet', [32,512])
def test_indexed_and_dense_rule_population_matches_python(seed, alphabet):
    g=model(seed, rules=650, alphabet=alphabet)
    rng=random.Random(seed+100)
    candidates=[[rng.randrange(256) for _ in range(rng.randrange(12,80))]
                for _ in range(5)]
    if seed==0:
        candidates.append([256,512,999,1,0,11,35])
    phases=1+seed%4
    ref=main.trajectory_features_cpu(candidates,g,256,hidden_rewrite_phases=phases)
    got=native.trajectory_features_native(candidates,g,256,hidden_rewrite_phases=phases,cpu_threads=4)
    for r,n in zip(ref[:3],got[:3]):
        np.testing.assert_array_equal(r,n)
    assert ref[3]==got[3]

@pytest.mark.parametrize('seed',range(4))
def test_indexed_fused_inference_reused_context(seed):
    gs=[model(seed+i, rules=550, alphabet=512) for i in range(3)]
    rng=random.Random(seed)
    for g in gs:
        g.readout_weights=[rng.uniform(-.2,.2) for _ in g.rules]
    with native.NativeInferenceContext(gs, 192, hidden_rewrite_phases=3,cpu_threads=3) as ctx:
        for _ in range(3):
            candidates=[[rng.randrange(256) for _ in range(40)] for _ in range(8)]
            expected=main.evaluate_inference_candidates(gs,candidates,'cpu',192,hidden_rewrite_phases=3)
            got=ctx.score(candidates)
            np.testing.assert_allclose(expected,got,atol=1e-10,rtol=1e-11)

def test_many_unmatched_rule_phase_boundary_preserves_hidden_state():
    g=main.Genome([
        main.Rule([-2,2,-2],[-128,-1]),
        *[main.Rule([500,i%257],[1]) for i in range(1050)],
        main.Rule([2],[3]),
        *[main.Rule([489,i%257],[7]) for i in range(300)],
    ])
    inp=[[2,1,2,1,2],[0,1,2,1,2,1]]
    ref=main.trajectory_features_cpu(inp,g,128,hidden_rewrite_phases=5)
    got=native.trajectory_features_native(inp,g,128,hidden_rewrite_phases=5,cpu_threads=2)
    for r,n in zip(ref[:3],got[:3]):np.testing.assert_array_equal(r,n)
    assert ref[3]==got[3]

def test_identity_duplicate_models_evaluated_once_and_broadcast():
    g=model(31, rules=530, alphabet=512)
    g.readout_weights=[0.04]*len(g.rules)
    h=model(33, rules=530, alphabet=512)
    h.readout_weights=[0.02]*len(h.rules)
    models=[g,h,g,g,h]
    candidates=[[1,3,4,5,6],[8,9,11],[1,3,4,5,6]]
    with native.NativeInferenceContext(models,128,cpu_threads=3) as ctx:
        assert ctx.model_count==5
        assert ctx._native_model_count==2
        score=ctx.score(candidates)
        np.testing.assert_array_equal(score[0],score[2]);np.testing.assert_array_equal(score[0],score[3]);np.testing.assert_array_equal(score[1],score[4])
    reference=main.evaluate_inference_candidates(models,candidates,'cpu',128)
    np.testing.assert_allclose(reference,score,atol=1e-12,rtol=1e-12)
