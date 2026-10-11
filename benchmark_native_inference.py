"""Compare legacy per-genome-native inference and fused-native inference.

Example: python benchmark_native_inference.py --models 64 --rules 750
Includes native-context setup as a separate timing line.
"""
import argparse
import random
import statistics
import time

import numpy as np

import main
import native_cpu_replacer as native


def bench(fn, repeats):
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        samples.append(time.perf_counter() - t)
    return out, statistics.median(samples)


def main_bench():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=int, default=64)
    parser.add_argument('--rules', type=int, default=750)
    parser.add_argument('--candidates', type=int, default=28)
    parser.add_argument('--length', type=int, default=96)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    a = parser.parse_args()
    r = random.Random(42)
    rulebank = [main.Rule([i], [i + 1]) for i in range(30)]
    rulebank += [main.Rule([4, -1, 5], [-1, 4]), main.Rule([6, -2, 7], [-128, -1])]
    models = []
    for _ in range(a.models):
        g = main.Genome(r.choices(rulebank, k=a.rules))
        g.readout_weights = [r.uniform(-0.05, 0.05) for _ in g.rules]
        models.append(g)
    cand = [[r.randrange(40) for _ in range(a.length)] for _ in range(a.candidates)]
    prep = native.prepare_inputs(cand)
    t = time.perf_counter()
    ctx = native.NativeInferenceContext(models, 256, cpu_threads=a.threads)
    setup = time.perf_counter() - t
    try:
        def legacy():
            result = np.empty((len(models), len(cand)), dtype=np.float64)
            for i, model in enumerate(models):
                x, b, _, _ = native.trajectory_features_native(
                    prep, model, 256, cpu_threads=a.threads, return_states=False)
                result[i] = main._prediction_scores_from_features(model, x, b)
            return result
        before, t_before = bench(legacy, a.repeats)
        after, t_after = bench(lambda: ctx.score(prep), a.repeats)
        np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-11)
        print(f'Number of models: {a.models} / rules: {a.rules} / candidates: {a.candidates}')
        print(f'Build context: {setup:.4f}s')
        print(f'Old per-genome-native inference: {t_before:.4f}s/batch')
        print(f'Fused native inference: {t_after:.4f}s/batch')
        print(f'Native scoring speedup (batch only): {t_before/t_after:.2f}x')
        print('Correctness: scores agree')
    finally:
        ctx.close()


if __name__ == '__main__':
    main_bench()
