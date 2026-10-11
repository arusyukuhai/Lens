"""Differential CPU-native / Python-reference regression and smoke tests."""
import math
import random
import time
import numpy as np
import main
import native_cpu_replacer as native


def equal_trajectory(genome, inputs, max_output=128, phases=2, threads=3):
    py = main.trajectory_features_cpu(inputs, genome, max_output, hidden_rewrite_phases=phases)
    cpp = native.trajectory_features_native(inputs, genome, max_output, hidden_rewrite_phases=phases, cpu_threads=threads)
    np.testing.assert_array_equal(py[0], cpp[0])
    np.testing.assert_array_equal(py[1], cpp[1])
    np.testing.assert_array_equal(py[2], cpp[2])
    assert py[3] == cpp[3], (py[3], cpp[3])
    return py


def test_wildcards_capture_operations_and_hidden_memory_exact():
    memops = [-128, -129, -130, -131, -132, -133, -134, -135, -136,
              -137, -138, -139, -140, -141, -142, -143, -144, -145,
              -146, -147, -148, -149, -150, -151]
    rulebank = [
        main.Rule([1], [2]),
        main.Rule([1, 2], [1]),
        main.Rule([-2, 9, -2], [-128, -1]),
        main.Rule([-3, 9, -3], [-129, -1]),
        main.Rule([-1, 4], [-1]),
        main.Rule([3, -1, 4], [3, -32]),
        main.Rule([-1, 4, -1], [-1, -2]),
        main.Rule([-2, 1, -2, 4], [-2]),
        main.Rule([4, -1], [-16]),
        main.Rule([7, -2, 8], [-48]),
        main.Rule([8, -3, 7], [-64]),
        main.Rule([9, -1], [-80]),
        main.Rule([9, -3], [-96]),
        main.Rule([-1], [-1]),
    ] + [main.Rule([-2, 9, -2], [op]) for op in memops]
    for seed in range(30):
        rng = random.Random(seed)
        rules = rng.choices(rulebank, k=rng.randint(1, 12))
        g = main.Genome(rules=rules)
        if seed % 4 == 0:
            g.embedding[:10] = list(reversed(g.embedding[:10]))
        inputs = [[rng.randrange(10) for _ in range(rng.randint(0, 18))]
                  for _ in range(6)]
        equal_trajectory(g, inputs, phases=1+(seed%5))


def test_native_embedding_oov_and_output_exhaustive_shapes():
    g = main.Genome([main.Rule([-2,3,-3], [-129,-2]), main.Rule([1], [5])])
    equal_trajectory(g, [[], [256, 999, 1, 3], [1,2,3,4,5]], max_output=2)
    equal_trajectory(g, [[1,3,2,3],[1,1],[3]], max_output=128)


def test_native_readout_and_inference_scores_equal():
    g = main.Genome([main.Rule([1], [2]), main.Rule([-2,3,-2], [-1])])
    inputs = [[1,2,3], [3,4], [1,2,3], [0,1,2]]
    g.readout_weights = [0.3, -0.1]
    a = main.evaluate_inference_candidates([g], inputs, 'cpu', 128)
    b = main.evaluate_inference_candidates([g], inputs, 'cpu-native', 128)
    np.testing.assert_array_equal(a,b)


def test_native_cached_pack_detects_rule_replacement():
    g = main.Genome([main.Rule([1], [2])])
    inputs = [[1,1,2,3]]
    equal_trajectory(g, inputs)
    g.rules[0] = main.Rule([1], [3])
    equal_trajectory(g, inputs)


def test_native_handles_expanding_debug_rules_and_out_of_vocab_literal():
    # Although evolved rules are repaired to be nonexpanding, diagnostics may
    # construct larger outputs. Native debug state buffer must not truncate.
    g = main.Genome([main.Rule([2], [2, 2, 2])])
    equal_trajectory(g, [[2], [1, 2], [], [2, 2]], max_output=64)
    g2 = main.Genome([main.Rule([512, 1], [9])])
    equal_trajectory(g2, [[999, 1, 4], [512, 1, 2]], max_output=64)
