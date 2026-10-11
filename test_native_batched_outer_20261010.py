"""Differential tests for fused population evaluation and shared rank graph."""
import random
import numpy as np
import pytest
import main
import native_cpu_replacer as native
from test_cpu_optimized_20261010 import model


@pytest.mark.parametrize('phases', [1,2,3,5])
@pytest.mark.parametrize('num_rules', [17,180,1050])
def test_batched_outer_is_bitwise_same_as_single_model_native(phases, num_rules):
    gs = [model(30+i, num_rules, 256 if i%2 else 512) for i in range(3)]
    gs[1].embedding = list(gs[1].embedding)
    random.Random(42).shuffle(gs[1].embedding)
    groups = gs+[gs[0]]
    rng=random.Random(num_rules+phases)
    inputs=[[rng.randrange(256) for _ in range(rng.randint(15,130))] for j in range(6)]
    packed=native.prepare_inputs(inputs)
    with native.NativeInferenceContext(groups, 240, hidden_rewrite_phases=phases, cpu_threads=3) as ctx:
        features, baseline, stats = ctx.features(packed)
        assert features.shape == (4, 6, num_rules)
        for i,g in enumerate(groups):
            expected=native.trajectory_features_native(packed, g, 240, hidden_rewrite_phases=phases,
                                                       cpu_threads=1, return_states=False)
            np.testing.assert_array_equal(features[i], expected[0]); np.testing.assert_array_equal(baseline[i], expected[1]); np.testing.assert_array_equal(stats[i], expected[2]);
        np.testing.assert_array_equal(features[0], features[3]);np.testing.assert_array_equal(baseline[0],baseline[3])


def test_readout_cached_pairs_are_identical():
    rng=np.random.default_rng(55)
    x=rng.integers(0,4,(35,200),dtype=np.int32).astype(np.float32)
    y=rng.random(35); b=rng.standard_normal(35)/8
    cases=np.asarray([i//7 for i in range(35)],np.int32)
    g=model(9,200,256)
    idx=np.arange(35,dtype=np.int32)
    clone=main.clone_genome_shallow(g)
    prepared=main.case_adjacent_rank_pairs(y,cases)
    a=main.score_genome_features(g,x,b,y,cases,idx,0.1)
    z=main.score_genome_features(clone,x,b,y,cases,idx,0.1,prepared_pairs=prepared)
    assert a==pytest.approx(z,abs=1e-11)
    np.testing.assert_allclose(g.readout_weights,clone.readout_weights,rtol=1e-11,atol=1e-11)
