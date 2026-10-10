"""Regression tests for bounded native evaluation, HOF sharing, and checkpoints."""
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import main
import native_cpu


def simple_g(n=1):
    return main.Genome([main.Rule([65, 0], [65, 66])] * n, list(range(256)))


def test_native_overflow_does_not_fail_batch(monkeypatch):
    monkeypatch.setenv('LENS_MAX_STATE_BYTES', '64')
    # Rule 0->00 has exponentially expanding hidden state.
    runaway = main.Genome([main.Rule([0], [0, 0])], list(range(256)))
    normal = simple_g()
    texts = [b'A' + b'\x00' * 25, b'ABAB']
    out = native_cpu.evaluate_cpu([runaway, normal], texts, 1, workers=2)
    assert out.shape == (2, 2)
    assert out[0, 0] == 0  # oversize candidate, not an exception or truncated state
    assert out[1, 1] == main.autoregressive_rollout(normal, texts[1])[0]


def test_hof_snapshot_structurally_shares_rules():
    g = simple_g(1500)
    snap = main.Genome(g.rules.copy(), g.embedding.copy(), g.fitness, g.case_scores.copy())
    assert snap.rules is not g.rules
    assert all(a is b for a, b in zip(snap.rules,g.rules))
    clone = main.mutate_genome(g, random.Random(7), [b'ABAB'], [(65,66)], False)
    assert all(main.is_valid_rule(r) for r in clone.rules)
    assert all(main.is_valid_rule(r) for r in snap.rules)


def test_checkpoint_roundtrip_legacy_and_shared(tmp_path):
    rng = random.Random(3)
    shared = main.Rule([65, 0], [65, 66])
    a = main.Genome([shared, shared], list(range(256)), .5, [.5])
    b = main.Genome([shared, main.Rule([66, 0],[66, 67])], list(range(256)), .75, [.75])
    dest = tmp_path/'small.pkl'
    main.save_checkpoint(str(dest), 14, [a, b], b, [], rng, [a, b])
    generation,pop,best,history,rng2,archive = main.load_checkpoint(str(dest))
    assert generation == 14 and len(pop)==2 and len(archive)==2
    assert pop[0].rules[0] is pop[1].rules[0] is archive[0].rules[0]
    assert best.fitness == b.fitness
    # Old v4 checkpoints stored nested dicts rather than shared dataclasses.
    with dest.open('rb') as f:
        payload=pickle.load(f)
    payload['population'] = [main.genome_object(g) for g in (a,b)]
    payload['best']=main.genome_object(b)
    payload['archive']=[main.genome_object(a)]
    with dest.open('wb') as f:
        pickle.dump(payload,f)
    generation,pop,best,history,rng2,archive = main.load_checkpoint(str(dest))
    assert generation == 14 and len(pop)==2 and len(archive)==1


def test_hof_capacity_not_lowered():
    import argparse
    import sys
    old=sys.argv
    try:
        sys.argv=['main.py']
        args=main.parse_args()
    finally:
        sys.argv=old
    assert args.hof_size == 4096
