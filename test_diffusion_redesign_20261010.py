"""Regression tests for the diversity-aware, severity-balanced diffusion path."""
import random
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
import main


class DiffusionRedesignTests(unittest.TestCase):
    def make_case(self, n: int, samples: int, noise: float):
        random.seed(191 + n + samples)
        np.random.seed(1771)
        base = (bytes(range(256)) * ((n + 255) // 256))[:n]
        sampler = main.CorpusSampler([base, b"more text data here"],
                                     ngram_pool=8, byte_buffer_size=4096)
        xs, ys, ids = main.make_noise_trajectory(base, samples, sampler, noise)
        return base, xs, ys, ids

    def test_small_chunks_have_no_repeated_nonzero_damage_examples(self):
        for n in (1, 2, 3, 4, 5, 9, 20, 96, 100, 1000):
            with self.subTest(n=n):
                base, xs, ys, ids = self.make_case(n, 100, .35)
                self.assertEqual(len(xs), 100)
                self.assertEqual(ids, list(range(100)))
                self.assertEqual(xs[0], list(base))
                self.assertEqual(xs.count(list(base)), 1)
                self.assertTrue(all(ys[i] >= ys[i + 1] for i in range(99)))
                seen = set()
                for row, clean in zip(xs, ys):
                    actual_k = sum(a != b for a, b in zip(row, base))
                    self.assertAlmostEqual(clean, 1.0 - actual_k / n)
                    self.assertNotIn(bytes(row), seen)
                    seen.add(bytes(row))
                max_k = max(1, min(max(1, n - 1), round(.35 * n)))
                self.assertEqual(max_k, max(sum(a != b for a, b in zip(row, base)) for row in xs))

    def test_multiple_severities_and_both_corruption_structures(self):
        base, xs, ys, _ = self.make_case(60, 210, .5)
        # For any mid-range tier represented many times, some errors form one
        # contiguous run (burst) and others are scattered across the string.
        tier = [(i, row) for i, (row, y) in enumerate(zip(xs, ys))
                if round((1.0 - y) * 60) == 12]
        self.assertGreaterEqual(len(tier), 2)
        shapes = []
        for _, row in tier:
            positions = [i for i, (a, b) in enumerate(zip(row, base)) if a != b]
            shapes.append(max(positions) - min(positions) + 1 == len(positions))
        self.assertIn(True, shapes)
        self.assertIn(False, shapes)

    def test_no_unnecessary_repetitions_when_levels_exceed_samples(self):
        base, xs, ys, _ = self.make_case(500, 20, .5)
        self.assertEqual(len(set(ys)), 20)
        self.assertEqual(len(set(map(bytes, xs))), 20)

    def test_legacy_opt_in_remains_identical_to_previous_schedule(self):
        base, _, _, _ = self.make_case(3, 20, .35)
        random.seed(100)
        np.random.seed(100)
        sampler = main.CorpusSampler([base], ngram_pool=4, byte_buffer_size=4096)
        xs, ys, sids = main.make_noise_trajectory(base, 20, sampler, .35, "legacy")
        self.assertEqual(len(xs), 20)
        self.assertEqual(sids, list(range(20)))
        self.assertLess(len(set(map(bytes, xs))), 20)
        self.assertEqual(ys[0], 1.0)

    def test_balanced_rank_pairs_do_not_waste_edges_on_ties(self):
        # 1 control + 99 separately corrupted examples at k=1; old solver
        # derived 98 zero-margin pairs and only ONE informative comparison.
        ys = np.asarray([1.0] + [.5] * 99, dtype=np.float64)
        cases = np.zeros(100, dtype=np.int32)
        hi, lo, margin = main.case_adjacent_rank_pairs(ys, cases)
        self.assertEqual(len(hi), 99)
        self.assertEqual(set(hi.tolist()) | set(lo.tolist()), set(range(100)))
        self.assertTrue(np.all(margin > 0))
        self.assertTrue(np.all(ys[hi] > ys[lo]))

    def test_legacy_rank_pair_mode_retains_old_zero_margin_ties(self):
        ys = np.asarray([1.0, .5, .5, .5, .5], dtype=np.float64)
        ids = np.zeros(5, dtype=np.int32)
        hi, lo, margin = main.case_adjacent_rank_pairs(ys, ids, "legacy")
        self.assertEqual(len(hi), 4)
        self.assertEqual(int(np.count_nonzero(margin)), 1)

    def test_every_training_example_participates_in_cross_tier_edges(self):
        _, _, ys, _ = self.make_case(20, 100, .5)
        y = np.asarray(ys, dtype=np.float64)
        cases = np.zeros(len(y), dtype=np.int32)
        hi, lo, margins = main.case_adjacent_rank_pairs(y, cases)
        self.assertEqual(set(hi.tolist()) | set(lo.tolist()), set(range(len(ys))))
        self.assertTrue(np.all(margins > 0))
        self.assertTrue(np.all(y[hi] > y[lo]))
        self.assertLessEqual(len(hi), 2 * len(y))

    def test_rolling_and_standalone_use_same_schedule(self):
        base = bytes(range(20))
        random.seed(400)
        np.random.seed(400)
        sampler = main.CorpusSampler([base], ngram_pool=4, byte_buffer_size=4096)
        rolling = main.RollingEvaluationSet([base], sampler, cases=1, samples=100,
                                             max_noise=.2, trajectory_len=0)
        xs, y, cases, sids = rolling.flatten()
        self.assertEqual(len(xs), 100)
        self.assertEqual(len(y), 100)
        self.assertEqual(sids.tolist(), list(range(100)))
        self.assertEqual(len(set(map(bytes, xs))), 100)


if __name__ == "__main__":
    unittest.main()
