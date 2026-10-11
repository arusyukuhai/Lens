"""2026-10-11 regression tests: numeric OLS/Pearson, guided synthesis, retained inner GA."""
from __future__ import annotations
import numpy as np
import main as m
import lens_direct_learning as guided


def test_ols_uses_entire_held_out_chunk_not_corrupted_rows():
    n = 24
    xx = np.linspace(0, 1, n)
    features = np.vstack([np.c_[xx, np.zeros(n)], np.c_[xx, np.zeros(n)]])
    baseline = np.zeros(2*n)
    # Train relationship positive, independent chunk has inverted relationship.
    target = np.r_[xx, 1 - xx]
    cases = np.r_[np.zeros(n, int), np.ones(n, int)]
    sample_ids = np.tile(np.arange(n), 2)
    w, holdout = m.fit_linear_ols_holdout(features, baseline, target, cases, sample_ids, max_features=8)
    assert np.all(holdout == (cases == 1))
    assert w[0] > 0.9
    assert w[1] == 0
    genome = m.Genome([m.Rule([1], [1]), m.Rule([2], [2])])
    fit = m.score_genome_features(genome, features, baseline, target,
                                  cases, sample_ids, ridge_lambda=12345,
                                  readout_mode='linear-pearson', rank_max_features=8)
    assert fit < -0.999
    assert genome.case_scores[0] > 0.999
    assert genome.case_scores[1] < -0.999


def test_ols_is_not_pairwise_rank_ridge_and_handles_constant_rule():
    x = np.c_[np.linspace(0, 1, 10), np.ones(10)]
    y = np.linspace(0, 1, 10)
    w, mask = m.fit_linear_ols_holdout(x, np.zeros(10), y,
                                       np.zeros(10, int), np.arange(10), 3)
    assert abs(float(w[0]) - 1.0) < 1e-7
    assert w[1] == 0
    assert 0 < mask.sum() < 10


def test_supervised_rule_mining_never_uses_validation_source():
    # Capture rule mining's training-only iteration with a deliberate, unique
    # validation-only corruption signature; source_cap large to cover all rows.
    import random
    from types import SimpleNamespace
    random.seed(123)
    train_clean = b'abcdefghijklmnopqrst'
    val_clean = b'abcdefghijklmnopqrst'
    train_inputs = [list(train_clean), list(b'abcdQfghijklmnopqrst'),
                    list(b'abcdefghiJklmnopqrst'), list(b'abcdefghiJklmnOpqrst')]
    val_inputs = [list(val_clean), list(b'abcdefghiZklmnopqrst')]
    slots = [SimpleNamespace(inputs=train_inputs, target=np.array([1., .95, .95, .9])),
             SimpleNamespace(inputs=val_inputs, target=np.array([1., .95]))]
    rolling = SimpleNamespace(slots=slots)
    feats = np.zeros((6, 2), dtype=np.float64)
    baseline = np.zeros(6, dtype=np.float64)
    target = np.r_[slots[0].target, slots[1].target]
    genome = m.Genome([m.Rule([1], [1]), m.Rule([2], [2])],
                      readout_weights=[0., 0.])
    genome.embedding = list(range(256))
    sampler = m.CorpusSampler([train_clean])
    proposals = guided.mine_repair_rules(rolling, genome, feats, baseline, target,
                                         sampler, m, limit=64, source_cap=100)
    assert proposals, 'expected at least one supervised repair proposal'
    assert all(ord('Z') not in rule.pattern for _, rule in proposals)
    assert all(m.is_rule_nonexpanding(rule) for _, rule in proposals)


def test_directed_offspring_are_independent_and_structurally_modified():
    base = m.Genome([m.Rule([1,2], [1,2]), m.Rule([3,4], [3,4])],
                    readout_weights=[0.0, 0.0])
    proposals = [(1.0, m.Rule([7,8], [7,9]))]
    children = guided.make_directed_offspring(m, base, proposals, [0, 1], 2)
    assert len(children) == 2
    assert base.rules[0].pattern == [1,2]
    assert children[0].rules[0].pattern == [7,8]
    assert children[1].rules[1].pattern == [7,8]
    assert children[0]._origin_operator == 'residual-guided'


def test_default_cli_runs_real_ga_with_guided_outer_learning(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    corpus = tmp_path / 'corpus.txt'
    corpus.write_bytes(b'abcdefghijklmnopqrstu\n===SPLIT===\n'
                       b'zyxwvutsrqponmlkjihg\n===SPLIT===\n'
                       b'the cat sits near the window\n===SPLIT===\n'
                       b'Hello world and welcome here\n')
    p = subprocess.run([
        sys.executable, str(Path(m.__file__)), '--backend', 'cpu-native',
        '--local-corpus', str(corpus), '--min-chunk', '8', '--max-chunk', '100',
        '--trajectory-len', '36', '--cases', '2', '--samples', '9',
        '--rules', '6', '--population', '5', '--elites', '1',
        '--generations', '1', '--hof-size', '3', '--hof-eval', '0',
        '--inference-cases', '1', '--inference-population', '4',
        '--inference-generations', '2', '--inference-ensemble', '2',
        '--inference-budget-mode', 'direct', '--ablation-every', '0',
        '--save', str(tmp_path/'best.json'), '--current-save', str(tmp_path/'current.json'),
        '--history-csv', str(tmp_path/'train.csv'), '--no-plot', '--no-progress',
    ], text=True, capture_output=True, cwd=str(tmp_path), timeout=60)
    assert p.returncode == 0, p.stderr + '\n' + p.stdout
    assert 'directed proposals=' in p.stdout
    assert 'infVote=' in p.stdout
    assert 'infOracle=nan' not in p.stdout
    assert 'infPlan=1x4x2/2r' in p.stdout
