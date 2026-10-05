import math
import random
import unittest
import numpy as np

import main as m


class V58SearchTests(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        np.random.seed(7)
        self.sampler = m.CorpusSampler([b"abcdef abcdef\n", b"hello world hello\n"])

    def genome(self, offset=0, rules=32):
        rs = [m.Rule([(i + offset) % 256], [((i + offset) + 1) % 256]) for i in range(rules)]
        g = m.Genome(rs)
        g.fitness = 0.5
        g.case_scores = [0.4, 0.5, 0.6]
        g._eval_case_serials = (10, 11, 12)
        return g

    def test_sparse_linkage_is_bounded(self):
        pop = []
        for i in range(24):
            g = self.genome(i)
            w = np.random.normal(0, 0.01, 32)
            shared = (i + 1) / 24.0
            w[[2, 7, 11, 19]] = shared
            g.readout_weights = w.tolist()
            g.fitness = 0.7 + i * 1e-3
            pop.append(g)
        pop.sort(key=lambda g: g.fitness, reverse=True)
        lm = m.SparseLinkageModel()
        lm.refresh(pop, 0, 8, 24, 16, 20, 8)
        self.assertTrue(lm.modules)
        self.assertLessEqual(len(lm.modules), 20)
        self.assertTrue(all(2 <= len(x) <= 8 for x in lm.modules))
        self.assertTrue(all(0 <= j < 32 for x in lm.modules for j in x))

    def test_linkage_mix_copy_on_write(self):
        base = self.genome(0, 8)
        donor = self.genome(40, 8)
        base.readout_weights = [0.01] * 8
        donor.readout_weights = [1.0] * 8
        lm = m.SparseLinkageModel()
        lm.modules = [(1, 3, 5)]
        lm.module_scores = [1.0]
        child, changed = m.linkage_mix(base, donor, lm, 1)
        self.assertEqual(changed, 3)
        self.assertEqual(child.rules[1].pattern, donor.rules[1].pattern)
        self.assertEqual(base.rules[1].pattern, [1])
        self.assertFalse(math.isfinite(child.fitness))

    def test_delayed_reward_uses_only_matching_serials(self):
        g = self.genome(0, 4)
        g._origin_operator = "local"
        g._parent_case_serials = (1, 2, 3)
        g._parent_case_scores = (0.50, 0.60, 0.70)
        g._eval_case_serials = (2, 3, 4)
        g.case_scores = [0.65, 0.68, 0.10]
        self.assertAlmostEqual(m.offspring_overlap_reward(g), 0.015, places=12)

    def test_qd_grid_keeps_best_per_cell(self):
        a = self.genome(0, 8)
        b = self.genome(1, 8)
        for g in (a, b):
            g.readout_weights = [1.0, 0, 0, 0, 0, 0, 0, 0]
            g.embedding = m.default_embedding()
            g.case_scores = [0.9, 0.2, 0.1]
        a.fitness = 0.7
        b.fitness = 0.8
        grid = m.build_qd_grid([a, b], 4)
        self.assertEqual(len(grid), 1)
        self.assertIs(next(iter(grid.values())), b)

    def test_bandit_observes_without_extra_eval(self):
        g = self.genome(0, 4)
        g._origin_operator = "mix_local"
        g._parent_case_serials = (10, 11, 12)
        g._parent_case_scores = (0.3, 0.4, 0.5)
        g._eval_case_serials = (10, 11, 12)
        g.case_scores = [0.31, 0.42, 0.49]
        b = m.AdaptiveOperatorBandit(("mix_local", "legacy"))
        n, r = b.observe_population([g])
        self.assertEqual(n, 1)
        self.assertAlmostEqual(r, (0.01 + 0.02 - 0.01) / 3, places=12)
        self.assertEqual(b.counts["mix_local"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
