"""Regression tests: diversified guided learning, v60-compatible diff3, GA unchanged."""
from __future__ import annotations
from types import SimpleNamespace
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import main as m
import lens_multistrategy as ms


def _test_set():
    clean = b'abcdefghijklmnopqrstuvwxyzaaa'
    sources = []
    for bias in (0, 1, 2):
        corrupt1 = bytearray(clean); corrupt1[5] = ord('X') + bias
        corrupt2 = bytearray(clean); corrupt2[12] = ord('Q') + bias
        both = bytearray(corrupt1); both[12] = ord('Q') + bias
        cases = [list(clean), list(corrupt1), list(corrupt2), list(both)]
        sources.append(SimpleNamespace(inputs=cases,
                                       target=np.array([1., .966, .966, .932])))
    rolling = SimpleNamespace(slots=sources)
    genome = m.Genome([m.Rule([1,2],[1,2]) for _ in range(12)],
                      readout_weights=[0.]*12)
    genome.embedding = list(range(256))
    sampler = m.CorpusSampler([clean])
    feat = np.zeros((12,12), dtype=float)
    baseline = np.zeros(12, dtype=float)
    target = np.concatenate([s.target for s in sources])
    return rolling, genome, sampler, feat, baseline, target


def test_strategy_registry_weights_and_errors():
    modes = ms.parse_modes('splice,diff3,repair,splice')
    assert modes == ('splice', 'diff3', 'repair')
    p = ms.parse_weights('', modes)
    assert abs(sum(p.values()) - 1.0) < 1e-12
    assert p['diff3'] > p['splice']
    assert ms.parse_weights('0,1,0', modes)['diff3'] == 1.0
    for bad in ('none', '', 'repair,notreal'):
        with pytest.raises(ValueError):
            ms.parse_modes(bad)
    for bad in ('0,0,0', '1,2', '-1,2,3', 'nan,1,2'):
        with pytest.raises(ValueError):
            ms.parse_weights(bad, modes)


def test_diverse_candidates_are_real_nonexpanding_and_diff3_is_three_way():
    random.seed(123)
    rolling, champion, sampler, feats, baseline, target = _test_set()
    proposals = ms.mine_strategies(rolling, champion, feats, baseline,
                                   target, sampler, m, limit=16, source_cap=32)
    assert set(proposals) == set(ms.MODES)
    assert all(proposals.values()), {k:len(v) for k,v in proposals.items()}
    assert any(-1 in r.pattern and -1 in r.replacement for _,r in proposals['diff3'])
    assert any(-1 in r.pattern for _,r in proposals['wildcard'])
    assert any(len(r.pattern) >= 5 for _,r in proposals['ngram'])
    for name, family in proposals.items():
        for score, rule in family:
            assert np.isfinite(score)
            assert rule.pattern != rule.replacement
            assert m.is_rule_nonexpanding(rule), name
            assert len(rule.pattern) <= m.MAX_RULE_TOKENS
            assert len(rule.replacement) <= m.MAX_RULE_TOKENS
    # The diff3 edit must preserve target-only right-side changes, while also
    # replaying a donor wildcard change at the unconflicted left-side anchor.
    merged, n = m.merge_sequence_delta([10, 20, 30], [-1, 20, 30], [10, 20, 30, 40])
    assert merged == [-1,20,30,40] and n == 1


def test_source_holdout_cannot_influence_supervised_proposals():
    random.seed(124)
    rolling, champion, sampler, feats, baseline, target = _test_set()
    random.seed(124)
    a = ms.mine_strategies(rolling, champion, feats, baseline, target,
                           sampler, m, limit=8, source_cap=32,
                           modes=('repair','wildcard','ngram','diff3','splice'))
    # Rewrite held-out source and labels in place, while intentionally holding
    # training rows and their features unchanged. No mining result may change.
    hold = rolling.slots[-1]
    hold.inputs = [list(b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ') for _ in hold.inputs]
    hold.target = np.array([1.,0.,0.,0.])
    target[-4:] = hold.target
    random.seed(124)
    b = ms.mine_strategies(rolling, champion, feats, baseline, target,
                           sampler, m, limit=8, source_cap=32,
                           modes=('repair','wildcard','ngram','diff3','splice'))
    assert {k: [(s, r.pattern, r.replacement) for s,r in v] for k,v in a.items()} == {k: [(s,r.pattern,r.replacement) for s,r in v] for k,v in b.items()}


def test_diverse_offspring_preserve_parent_cache_and_capture_validity():
    random.seed(127)
    rolling, champion, sampler, feats, baseline, target = _test_set()
    strategies = ms.mine_strategies(rolling, champion, feats, baseline,
                                     target, sampler, m, limit=8, source_cap=30)
    parent_before = [(r.pattern.copy(),r.replacement.copy()) for r in champion.rules]
    children, counts = ms.make_multi_offspring(m, champion, strategies,
                                                list(range(8)), 30,
                                                ms.parse_weights('',ms.MODES))
    assert len(children) == 30
    assert all(v >= 1 for v in counts.values()), counts
    assert len(set(tuple((i,tuple(r.pattern),tuple(r.replacement)) for i,r in enumerate(g.rules))
                   for g in children)) == len(children)
    assert all(g._origin_operator.startswith('guided:') for g in children)
    assert parent_before == [(r.pattern,r.replacement) for r in champion.rules]
    assert all(m.is_rule_nonexpanding(r) for g in children for r in g.rules)


def test_guided_multi_and_original_inner_ga_two_generations(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_bytes(b'abcdefghijklmnopqrstuvwxyz\n===SPLIT===\n'
                       b'zyxwvutsrqponmlkjihgfedcba\n===SPLIT===\n'
                       b'hello lovely world abcdefghijklmnop\n===SPLIT===\n'
                       b'good evening today qwertyuiopasdf\n')
    cmd = [sys.executable, str(Path(m.__file__)), '--backend','cpu-native',
           '--local-corpus',str(corpus),'--min-chunk','8','--max-chunk','100',
           '--trajectory-len','36','--cases','3','--samples','10','--rules','12',
           '--population','16','--elites','1','--generations','2',
           '--hof-size','8','--hof-eval','0', '--inference-cases','1',
           '--inference-population','4','--inference-generations','2',
           '--inference-ensemble','2','--inference-budget-mode','direct',
           '--ablation-every','0', '--guided-candidates','8',
           '--guided-source-samples','24', '--no-progress', '--no-plot',
           '--save',str(tmp_path/'b.json'),'--current-save',str(tmp_path/'c.json'),
           '--history-csv',str(tmp_path/'history.csv')]
    p = subprocess.run(cmd,cwd=tmp_path,text=True,capture_output=True,timeout=75)
    assert p.returncode == 0, p.stderr + '\n'+p.stdout
    assert 'lanes=repair:' in p.stdout
    assert 'diff3:' in p.stdout
    assert 'guided=' in p.stdout
    assert 'infPlan=1x4x2/2r' in p.stdout
    assert 'infOracle=nan' not in p.stdout
    import csv
    with open(tmp_path/'history.csv',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==2
    assert sum(int(float(v)) for k,v in rows[-1].items() if k.startswith('guided_children_')) > 0
    assert 'guided_prev_evaluated_diff3' in rows[-1]
    assert 'guided_prev_reward_repair' in rows[-1]
