import contextlib
import csv
import io
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import main as m


class Improvements(unittest.TestCase):
    def test_ridge_matches_original_dense_objective(self):
        rng = np.random.default_rng(817)
        for lam in (0.1, 4., 100.):
            x = rng.integers(0, 10, (36, 1500)).astype(np.float32)
            x[:, 100:] = 0
            x[:, 8] = x[:, 7]  # dependent columns
            y = rng.normal(size=36)
            xd = x.astype(np.float64)
            ss = (xd * xd).sum(axis=0)
            scale = np.zeros(x.shape[1]); scale[ss > 1e-12] = len(x) / ss[ss > 1e-12]
            k = (xd * scale) @ xd.T + lam * np.eye(len(x))
            expected = scale * (xd.T @ np.linalg.solve(k, y))
            np.testing.assert_allclose(m.fit_dual_ridge(x, y, lam), expected, atol=2e-12, rtol=2e-10)
        np.testing.assert_array_equal(m.fit_dual_ridge(np.zeros((36, 1500)), y, 4), 0)

    def test_tie_ranks(self):
        np.testing.assert_equal(m.average_ranks(np.array([3, 1, 3, 2, 1])), [4.5, 1.5, 4.5, 3, 1.5])
        self.assertEqual(len(m.average_ranks(np.array([]))), 0)
        self.assertEqual(m.spearman([1, 1, 1], [3, 2, 1]), 0)

    def test_plateau_ignores_tiny_positive_jitter(self):
        p = m.PlateauTracker(20, 0.0002)
        for i in range(120):
            stalled, gain = p.update(0.995 + (i % 2) * 0.00001)
        self.assertGreaterEqual(stalled, 36)
        self.assertAlmostEqual(gain, 0)
        p = m.PlateauTracker(20, 0.0002)
        for i in range(120):
            stalled, gain = p.update(.8 + i * .0001)
        self.assertEqual(stalled, 0)
        self.assertGreater(gain, .0002)

    def test_specialists_survive(self):
        pop = [m.Genome([], case_scores=s) for s in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, .5])]
        random.seed(42)
        selector = m.CaseSelector(pop)
        picked = {id(selector.pick()) for _ in range(100)}
        self.assertTrue(all(id(g) in picked for g in pop[:3]))

    def test_stale_hof_score_cannot_replace_same_lineage(self):
        g = m.Genome([m.Rule([1], [])], fitness=.99, case_scores=[.99])
        archive = [m.make_hof_entry(g, (10,), 0, 32, .7)]
        candidate = m.clone_genome_deep(g); candidate.fitness = 1; candidate.case_scores = [1]
        admitted, replaced = m.update_hall_of_fame(archive, [candidate], (11,), 1, 8, 32, .7, .015)
        self.assertEqual((admitted, replaced), (0, 0))
        admitted, replaced = m.update_hall_of_fame(archive, [candidate], (10,), 1, 8, 32, .7, .015)
        self.assertEqual(replaced, 1)

    def test_csv_superset_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'history.csv'; p.write_text('generation,old_column\n0,yes\n')
            with patch.object(m.os, 'replace', side_effect=AssertionError('unexpected full rewrite')):
                m.append_history_csv(str(p), {'generation': 1})
            self.assertEqual(len(list(csv.DictReader(io.StringIO(p.read_text())))), 2)

    def test_audit_uses_frozen_readout_and_paired_cases(self):
        import audit_generalization as audit
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus = root / 'audit.txt'; corpus.write_text('abc abc abc\nxyz xyz abc\nabc xyx xyx\n')
            model = root / 'model.json'
            m.save_genome(str(model), m.Genome([m.Rule([97], [])], readout_weights=[1.]), 0)
            args = SimpleNamespace(seed=17, local_corpus=str(corpus), min_chunk=4, max_chunk=24,
                                   cases=3, samples=9, max_noise=.35, models=[str(model), str(model)],
                                   backend='cpu', max_output=128, output='')
            with patch.object(m, 'fit_dual_ridge', side_effect=AssertionError('audit must not fit')), contextlib.redirect_stdout(io.StringIO()):
                report = audit.run(args)
            self.assertEqual(report['paired_difference_second_minus_first']['mean'], 0)
            self.assertEqual(report['models'][0]['case_scores'], report['models'][1]['case_scores'])

    def test_mps_orchestration_matches_cpu_with_stale_archive(self):
        # Execute the actual MPS scheduling/reuse branch. Only kernel execution
        # is substituted with the reference evaluator; this is not a GPU test.
        from types import SimpleNamespace
        class ReferenceEvaluator:
            def __init__(self, sample_count, rule_count, max_raw_len, max_output, **kwargs):
                self.sample_count = sample_count
                self.max_output = max_output
                self._pool_meta_host = []
                self.last_profile = {}
                self.unique_input_count = sample_count
            def __getattr__(self, name):
                return 0
            def set_inputs(self, inputs):
                self.inputs = inputs
            def evaluate_population(self, genomes, **kwargs):
                return [m.trajectory_features_cpu(self.inputs, g, self.max_output)[:3] for g in genomes]
        torch = SimpleNamespace(manual_seed=lambda seed: None,
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); corpus = root / 'corpus.txt'
            corpus.write_text('abc abc xyy xyz\nxyz hello ababcd\nabcd abbc zzzzyx\naaa bbb ccc ddd\n')
            histories = []
            for backend in ('cpu', 'mps'):
                argv = ['main.py', '--backend', backend, '--population', '8', '--rules', '12',
                        '--cases', '3', '--samples', '9', '--min-chunk', '4', '--max-chunk', '24',
                        '--trajectory-len', '24', '--local-corpus', str(corpus), '--generations', '12',
                        '--hof-size', '16', '--hof-eval', '2', '--hof-candidates', '3', '--hof-min-distance', '0',
                        '--elites', '2', '--no-progress', '--no-plot', '--checkpoint', '',
                        '--save', str(root/'best.json'), '--current-save', '', '--history-csv', str(root/(backend+'.csv'))]
                with patch.object(sys, 'argv', argv), patch.object(m, 'torch', torch), \
                     patch.object(m, 'MpsPopulationEvaluator', ReferenceEvaluator), \
                     patch.object(m, '_get_mps_trajectory_lib', lambda: None), contextlib.redirect_stdout(io.StringIO()):
                    m.evolve(m.parse_args())
                histories.append(list(csv.DictReader(io.StringIO((root/(backend+'.csv')).read_text()))))
            for a, b in zip(*histories):
                for field in ('best', 'mean', 'hof_best_current', 'hof_size'):
                    self.assertAlmostEqual(float(a[field]), float(b[field]), places=12, msg=field)
            self.assertTrue(any(int(r['eval_reused']) > 0 for r in histories[1]))

    def test_cpu_training_checkpoint_resume_and_hof_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); corpus = root / 'corpus.txt'
            corpus.write_text('\n'.join(['abc xyz abcd xyzy', 'xxyy zzzy aabb cc', 'hello abc world x', 'abbc ddda xyz xyz']))
            base = ['main.py', '--backend', 'cpu', '--population', '8', '--rules', '12', '--cases', '3', '--samples', '9',
                    '--min-chunk', '4', '--max-chunk', '24', '--trajectory-len', '24', '--local-corpus', str(corpus),
                    '--hof-size', '8', '--hof-eval', '2', '--hof-candidates', '3', '--hof-min-distance', '0',
                    '--elites', '2', '--generations', '5', '--plateau-window', '2', '--no-plot', '--no-progress',
                    '--current-save', str(root/'latest.json'), '--save', str(root/'best.json'), '--checkpoint', str(root/'cp.npz'), '--history-csv', str(root/'h.csv')]
            with patch.object(sys, 'argv', base), contextlib.redirect_stdout(io.StringIO()):
                args = m.parse_args(); m.evolve(args)
            rows = list(csv.DictReader(io.StringIO((root/'h.csv').read_text())))
            self.assertTrue(all(int(r['hof_evaluated']) <= 2 for r in rows))
            self.assertTrue(any(int(r['hof_evaluated']) == 2 for r in rows))
            self.assertTrue(all(-1 <= float(r['best']) <= 1 for r in rows))
            sampler = m.CorpusSampler([b'abc xyz'])
            generation, pop, best, history, config, hof = m.load_checkpoint(str(root/'cp.npz'), sampler)
            self.assertEqual(generation, 5); self.assertEqual(len(pop), 8)
            self.assertTrue(all(g._eval_case_serials == () for g in pop + hof))
            args.load = str(root/'cp.npz'); args.generations = 7; args.hof_size = 1
            with contextlib.redirect_stdout(io.StringIO()):
                m.evolve(args)
            self.assertGreaterEqual(args.hof_size, len(hof))
            self.assertEqual(len(list(csv.DictReader(io.StringIO((root/'h.csv').read_text())))), 7)

if __name__ == '__main__':
    unittest.main(verbosity=2)
