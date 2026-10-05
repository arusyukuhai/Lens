import unittest
import numpy as np
import main


class V59RankReadoutTests(unittest.TestCase):
    def test_3x36_all_samples_participate_in_pair_graph(self):
        cases, samples = 3, 36
        case_ids = np.repeat(np.arange(cases, dtype=np.int32), samples)
        # Include realistic rounded/tied cleanliness values.
        clean = np.concatenate([
            1.0 - np.round(np.linspace(0, 17 + c, samples)) / 100.0
            for c in range(cases)
        ]).astype(np.float64)
        hi, lo, margin = main.case_adjacent_rank_pairs(clean, case_ids)
        self.assertEqual(len(hi), cases * (samples - 1))
        self.assertEqual(len(lo), cases * (samples - 1))
        self.assertEqual(len(margin), cases * (samples - 1))
        touched = set(map(int, hi)) | set(map(int, lo))
        self.assertEqual(touched, set(range(cases * samples)))
        self.assertTrue(np.all(margin >= -1e-15))
        # No pair may cross a case boundary.
        self.assertTrue(np.all(case_ids[hi] == case_ids[lo]))

    def test_pairwise_all_scores_all_36_per_case(self):
        cases, samples, p = 3, 36, 8
        case_ids = np.repeat(np.arange(cases, dtype=np.int32), samples)
        sample_ids = np.tile(np.arange(samples, dtype=np.int32), cases)
        clean = np.tile(np.linspace(1.0, 0.0, samples), cases)
        # One feature follows cleanliness monotonically in every case.
        x = np.zeros((cases * samples, p), dtype=np.float32)
        x[:, 0] = clean.astype(np.float32)
        baseline = np.zeros(cases * samples, dtype=np.float64)
        g = main.Genome([main.Rule([1], [1]) for _ in range(p)])
        f = main.score_genome_features(
            g, x, baseline, clean, case_ids, sample_ids, 4.0, "pairwise-all"
        )
        self.assertEqual(len(g.case_scores), cases)
        self.assertGreater(f, 0.99)
        self.assertTrue(all(v > 0.99 for v in g.case_scores))

    def test_screened_rank_ridge_is_exact_when_under_cap(self):
        rng = np.random.default_rng(7)
        cases, samples, p = 3, 12, 96
        n = cases * samples
        case_ids = np.repeat(np.arange(cases, dtype=np.int32), samples)
        clean = np.tile(np.linspace(1.0, 0.5, samples), cases)
        baseline = np.zeros(n, dtype=np.float64)
        x = rng.poisson(0.15, size=(n, p)).astype(np.float32)
        hi, lo, margin = main.case_adjacent_rank_pairs(clean, case_ids)
        pair_x = x[hi] - x[lo]
        expected = main.fit_dual_ridge(pair_x, margin, 4.0)
        actual = main.fit_pairwise_rank_ridge_all(
            x, baseline, clean, case_ids, 4.0, max_features=256
        )
        self.assertTrue(np.allclose(actual, expected, rtol=1e-10, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
