import pytest
import random
import unittest
from types import SimpleNamespace

import main as m



pytestmark = pytest.mark.skip(reason="differential-evolution path intentionally retired; current trainer uses the standard GA")

class V61DifferentialEvolutionTests(unittest.TestCase):
    def setUp(self):
        random.seed(123)
        self.sampler = m.CorpusSampler([b"abcdef abcdef", b"xyz xyz xyz"])

    def g(self, vals):
        return m.Genome([m.Rule(list(p), list(r)) for p, r in vals])

    def test_diff3_applies_b_to_c_delta_onto_a(self):
        a = self.g([([1, 2, 8], [4])])
        b = self.g([([1, 2, 3], [4])])
        c = self.g([([1, 9, 3], [4])])
        child, changed, conflicts = m.differential_trial_diff3(a, b, c, self.sampler)
        self.assertEqual(child.rules[0].pattern, [1, 9, 8])
        self.assertEqual(changed, 1)
        self.assertEqual(conflicts, 0)

    def test_diff3_conflict_resolves_to_whole_parent_gene(self):
        a = self.g([([1, 7, 3], [4])])
        b = self.g([([1, 2, 3], [4])])
        c = self.g([([1, 9, 3], [4])])
        seen = set()
        for seed in range(60):
            random.seed(seed)
            child, _, conflicts = m.differential_trial_diff3(a, b, c, self.sampler)
            self.assertEqual(conflicts, 1)
            seen.add(tuple(child.rules[0].pattern))
        allowed = {(1, 7, 3), (1, 2, 3), (1, 9, 3)}
        self.assertTrue(seen <= allowed)
        self.assertGreaterEqual(len(seen), 2)

    def test_strict_survivor_requires_both_objectives(self):
        p = m.Genome([], fitness=.5, inference_accuracy=.5)
        t = m.Genome([], fitness=.6, inference_accuracy=.5)
        self.assertFalse(m._strict_de_dominates(t, p, False))
        t.inference_accuracy = .6
        self.assertTrue(m._strict_de_dominates(t, p, False))
        t.fitness = .5
        self.assertFalse(m._strict_de_dominates(t, p, False))

    def test_three_case_refresh_is_distinct_slots(self):
        rolling = m.RollingEvaluationSet(
            [b"aaaa", b"bbbb", b"cccc", b"dddd", b"eeee"],
            self.sampler, 4, 3, .2, 4,
        )
        old = rolling.serials
        slots = m._rotate_many_eval_cases(rolling, 3)
        self.assertEqual(len(slots), 3)
        self.assertEqual(len(set(slots)), 3)
        changed = sum(a != b for a, b in zip(old, rolling.serials))
        self.assertEqual(changed, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
