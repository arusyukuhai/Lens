"""Regression tests for observational tqdm progress without changing learning."""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import main as m
import native_cpu


class RecordingProgress:
    def __init__(self):
        self.current = 0
        self.calls = []
        self.status = []

    def update(self, value):
        self.current += value
        self.calls.append(value)

    def set_postfix_str(self, text, refresh=True):
        self.status.append(text)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [b'ABABCCABCABC!', bytes(range(1, 48)), b'alpha\nbeta\ngamma\n']
        self.rng = random.Random(92)
        self.bigrams = m._bigram_seeds(self.corpus)

    def test_native_parallel_per_genome_ticks_exactly_once(self):
        gs = [m.new_genome(14, self.rng, self.corpus, self.bigrams) for _ in range(25)]
        done = RecordingProgress()
        got = native_cpu.evaluate_cpu(gs, self.corpus, 14, workers=4,
                                      on_genome_done=done.update)
        baseline = native_cpu.evaluate_cpu(gs, self.corpus, 14, workers=4)
        np.testing.assert_array_equal(got, baseline)
        self.assertEqual(done.current, len(gs))
        self.assertEqual(done.calls, [1] * len(gs))

    def test_cache_and_duplicates_report_all_genomes(self):
        gs = [m.new_genome(16, self.rng, self.corpus, self.bigrams) for _ in range(6)]
        gs += [m.Genome(gs[0].rules[:], gs[0].embedding[:]),
               m.Genome(gs[1].rules[:], gs[1].embedding[:])]
        progress = RecordingProgress()
        m.score_population(gs, self.corpus, 'cpu', cpu_workers=3,progress=progress)
        self.assertEqual(progress.current, len(gs))
        prior = [(g.fitness, list(g.case_scores)) for g in gs]
        cached = RecordingProgress()
        m.score_population(gs, self.corpus, 'cpu', cpu_workers=3,progress=cached)
        self.assertEqual(cached.current, len(gs))
        self.assertEqual(prior, [(g.fitness, list(g.case_scores)) for g in gs])
        self.assertTrue(any('cache=' in s for s in cached.status))

    def test_reference_python_progress(self):
        gs = [m.new_genome(5, self.rng, self.corpus, self.bigrams) for _ in range(4)]
        progress = RecordingProgress()
        m.score_population(gs, self.corpus, 'python', progress=progress)
        self.assertEqual(progress.current, len(gs))

    def test_mocked_mps_reports_whole_finished_batches(self):
        # Exercise the MPS batch path with an explicitly compatible population.
        # Freshly seeded genomes may contain newly supported CPU-only binary ops.
        gs = [m.Genome([m.Rule([65, 0], [65, 66 + i]) for _ in range(10)])
              for i in range(7)]
        progress = RecordingProgress()
        def fake_mps(models, samples, rules, batch_size, on_batch_done=None):
            results = np.array([[m.autoregressive_rollout(g, s)[0] for s in samples]
                                for g in models], dtype=np.int32)
            for i in range(0, len(models), batch_size):
                if on_batch_done:
                    on_batch_done(len(models[i:i+batch_size]))
            return results
        with patch.object(m.gpu_backend, 'evaluate_mps', side_effect=fake_mps), \
             patch.object(m, '_MPS_CROSSCHECK_DONE', True):
            m.score_population(gs, self.corpus, 'mps',mps_batch=3,progress=progress)
        self.assertEqual(progress.calls, [3, 3, 1])
        self.assertEqual(progress.current, len(gs))

    def test_cli_progress_does_not_change_model(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for label, extra in [('on', []), ('off', ['--no-tqdm'])]:
                model = td / f'{label}_best.json'
                cmd = [sys.executable, str(Path(m.__file__)), '--backend', 'cpu',
                       '--local-corpus', str(Path(m.__file__).with_name('example_corpus.txt')),
                       '--population', '13', '--rules', '12', '--cases', '3',
                       '--generations', '3', '--seed', '73', '--no-plot',
                       '--checkpoint', str(td / f'{label}.pkl'),
                       '--history-csv', str(td / f'{label}.csv'),
                       '--save', str(model)] + extra
                result = subprocess.run(cmd, cwd=td, capture_output=True, text=True,check=True)
                if label == 'on':
                    self.assertIn('評価/cpu', result.stderr)
                    self.assertIn('次世代作成', result.stderr)
                else:
                    self.assertNotIn('評価/cpu', result.stderr)
            with (td / 'on_best.json').open() as f:
                on = json.load(f)
            with (td / 'off_best.json').open() as f:
                off = json.load(f)
            self.assertEqual(on, off)

if __name__ == '__main__':
    unittest.main()
