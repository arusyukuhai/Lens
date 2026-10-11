"""Regression tests: fused native inference has no Python per-genome fallback."""
import random
import numpy as np
import pytest
import main
import native_cpu_replacer as native


def make_models(seed, count=5, maxrules=14):
    r = random.Random(seed)
    bank = [
        main.Rule([1], [2]), main.Rule([2], [1]), main.Rule([1, 2], [2]),
        main.Rule([3, -1, 4], [3, -32]), main.Rule([-2, 9, -2], [-128, -1]),
        main.Rule([-2, 9, -2], [-129, -1]), main.Rule([9, -1], [-16]),
        main.Rule([9, -2], [-48]), main.Rule([9, -3], [-64]),
        main.Rule([8, -2, 9], [-131, -1]),
        main.Rule([1, 1], [1]), main.Rule([0], [0]),
    ]
    models = []
    for i in range(count):
        rules = r.choices(bank, k=r.randrange(1, maxrules))
        model = main.Genome(rules)
        if i % 3 != 0:
            model.readout_weights = [r.uniform(-3, 3) for _ in rules]
        if i % 2 == 0:
            model.embedding[:12] = r.sample(model.embedding[:12], k=12)
        models.append(model)
    return models


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('phases', [1, 2, 4])
def test_fused_inference_matches_reference(seed, phases):
    r = random.Random(seed)
    models = make_models(seed)
    candidates = [[r.randrange(12) for _ in range(r.randrange(0, 35))]
                  for _ in range(18)]
    expected = main.evaluate_inference_candidates(models, candidates, 'cpu', 96,
                                                   hidden_rewrite_phases=phases)
    got = main.evaluate_inference_candidates(models, candidates, 'cpu-native', 96,
                                              hidden_rewrite_phases=phases)
    np.testing.assert_allclose(expected, got, atol=1.e-11, rtol=1.e-12)
    assert np.isfinite(got).all()


def test_context_reuse_across_candidate_generations():
    models = make_models(124, count=4)
    rng = random.Random(321)
    with_model = native.NativeInferenceContext(models, 96, hidden_rewrite_phases=3, cpu_threads=3)
    try:
        for n in (13, 33, 9):
            candidates = [[rng.randrange(10) for _ in range(rng.randrange(1, 24))]
                          for _ in range(n)]
            a = main.evaluate_inference_candidates(models, candidates, 'cpu', 96,
                                                   hidden_rewrite_phases=3)
            b = with_model.score(candidates)
            np.testing.assert_allclose(a, b, atol=1.e-11, rtol=1.e-12)
    finally:
        with_model.close()
    with pytest.raises(RuntimeError):
        with_model.score([[1]])


def test_fused_native_inference_must_not_call_trajectory_features(monkeypatch):
    models = make_models(5, count=3)
    def failure(*_args, **_kwargs):
        raise AssertionError('legacy per-genome native call used')
    monkeypatch.setattr(native, 'trajectory_features_native', failure)
    score = main.evaluate_inference_candidates(models, [[1,2,3], [4,5,6]],
                                               'cpu-native', 96)
    assert score.shape == (3, 2)


def test_fused_native_handles_no_active_readout():
    models = [main.Genome([main.Rule([1], [2])]), main.Genome([])]
    candidates = [[1,2,3], [], [999, 1]]
    a = main.evaluate_inference_candidates(models, candidates, 'cpu', 96)
    b = main.evaluate_inference_candidates(models, candidates, 'cpu-native', 96)
    np.testing.assert_array_equal(a,b)


def test_inference_ga_reports_native_calls():
    random.seed(5)
    corpus = [b'abcabcabcabcabcabc', b'abcdabcdabcdabcd', b'xyxyxyxyxyxyxyxy']
    sampler = main.CorpusSampler(corpus, ngram_pool=8, byte_buffer_size=4096)
    infset = main.RollingInferenceSet(corpus, sampler, cases=2, span=12, noise_rate=.12)
    models = make_models(3, count=5)
    for i,m in enumerate(models):
        m.fitness=1.0-i*.01
    result = main.run_string_inference_ga(models, infset, sampler,
        population_size=4, generations=3, ensemble_size=3,
        backend='cpu-native', max_output=128, case_count=2,
    )
    assert result['native_calls'] >= 2
    assert result['native_eval_seconds'] >= 0.
    assert result['mps_calls'] == 0


def test_inference_readout_active_weight_threshold_and_nan():
    model = main.Genome([main.Rule([1], [2]), main.Rule([2], [3])])
    model.readout_weights = [1e-14, 0.75]
    cand = [[1,2], [2,2], [1], []]
    np.testing.assert_allclose(
        main.evaluate_inference_candidates([model], cand, 'cpu', 64),
        main.evaluate_inference_candidates([model], cand, 'cpu-native', 64),
        rtol=0, atol=1e-14,
    )
    model.readout_weights = [float('nan'), .75]
    np.testing.assert_allclose(
        main.evaluate_inference_candidates([model], cand, 'cpu', 64),
        main.evaluate_inference_candidates([model], cand, 'cpu-native', 64),
        rtol=0, atol=1e-14,
    )
