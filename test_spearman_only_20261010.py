"""Regression tests for the inference-free Spearman training path."""
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import main


def test_spearman_selection_does_not_read_inference():
    # A worse Spearman model must not win because of inference scores/rank.
    low = main.Genome([main.Rule([97], [97])], fitness=0.2, inference_accuracy=1.0,
                      pareto_rank=0, pareto_crowding=math.inf)
    high = main.Genome([main.Rule([98], [98])], fitness=0.9, inference_accuracy=0.0,
                       pareto_rank=9999, pareto_crowding=0.0)
    assert main.select_spearman_elites([low, high], 1) == [high]
    assert main.spearman_sort_key(high) < main.spearman_sort_key(low)
    assert main.build_qd_grid([low, high], 1, fitness_only=True)  # descriptors can differ
    with patch.object(main.random, "sample", return_value=[0, 1]):
        assert main.tournament([low, high], 2, fitness_only=True) is high


def test_train_without_inference_or_pareto():
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        corpus = t / 'corpus.txt'
        corpus.write_text('\n'.join('def function_%d(x): return x * 3 # script testing'%i for i in range(12)))
        argv = [
            'main.py', '--backend', 'cpu-native', '--local-corpus', str(corpus),
            '--min-chunk', '20', '--max-chunk', '120', '--trajectory-len', '90',
            '--population', '6', '--rules', '12', '--cases', '2', '--samples', '3',
            '--generations', '2', '--elites', '2', '--hof-size', '8', '--hof-eval', '0',
            '--qd-inject', '0', '--linkage-elites', '3', '--linkage-loci', '5',
            '--linkage-modules', '2', '--checkpoint-every', '0',
            '--checkpoint', str(t/'checkpoint.npz'), '--save', str(t/'best.json'),
            '--current-save', str(t/'latest.json'), '--history-csv', str(t/'history.csv'),
            '--no-plot', '--no-progress', '--cpu-threads', '2', '--cpu-model-batch', '4',
        ]
        with patch.object(sys, 'argv', argv), \
             patch.object(main, 'run_string_inference_ga', side_effect=AssertionError('inference called')), \
             patch.object(main, 'assign_pareto_metrics', side_effect=AssertionError('Pareto called')), \
             patch.object(main.CorpusSampler, '_rebuild_context_model', side_effect=AssertionError('inference startup called')):
            champion = main.evolve(main.parse_args())
        assert math.isfinite(champion.fitness)
        with (t/'history.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert all(row['objective'] == 'spearman_only' for row in rows)
        assert all(not k.startswith('inference_') for k in rows[0])
        assert 'pareto_front_size' not in rows[0]
        latest = json.loads((t/'latest.json').read_text())
        assert latest['fitness'] == float(rows[-1]['best'])


def test_spearman_only_saturation_plots_ignore_legacy_inference_data():
    # This change is verified structurally in the patched plot implementation:
    # legacy inference_best keys may exist in history but must not be drawn.
    import inspect
    plot_src = inspect.getsource(main.write_plots)
    assert 'inference_best' not in plot_src
