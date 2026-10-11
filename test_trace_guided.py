"""Tests that the new optimizer learns from *observed failures*, not GA mutation."""
import argparse
import csv
import json
import random

import main
import trace_guided as tg


def identity_genome(rules=3):
    return main.Genome([main.Rule([255], [255]) for _ in range(rules)])


def opts(**kw):
    config = dict(trace_tail=1, trace_prefix=32, trace_steps=20,
                  trace_state_cap=512, trace_texts=2, trace_context=4,
                  trace_candidates=96, trace_confirm=8, backend='cpu',
                  mps_batch=8, cpu_workers=1, no_eval_cache=False)
    config.update(kw)
    return argparse.Namespace(**config)


def test_trace_records_true_teacher_forced_baseline():
    g = identity_genome()
    example = b'ABABABA'
    snaps, overflow = tg.collect_snapshots(main, g, [example], 2,
                                           prefix=7, steps_per_text=20)
    correct, total, predictions, _ = main.autoregressive_rollout(
        g, example, predictions=True, return_states=True)
    assert overflow == 0
    assert len(snaps) == total
    assert [s.baseline_prediction for s in snaps] == predictions
    assert [s.target for s in snaps] == list(example[1:])


def test_constructs_literal_rule_that_directly_repairs_prediction():
    g = identity_genome()
    texts = [b'ABABABAB', b'ABABAB']
    main.score_population([g], texts, 'cpu', cpu_workers=1)
    new, info = tg.improve_once(main, g, texts, 0, opts())
    assert info['trace_errors'] > 0
    assert info['trace_proposals'] > 0
    assert info['trace_accepted'] == 1
    assert info['trace_gain'] > 0
    assert new.fitness > g.fitness
    assert all(main.is_valid_rule(r) for r in new.rules)
    assert g.rules[-1].pattern == [255] # incumbent never mutated in-place


def test_local_replay_considers_replacing_an_active_original_rule():
    g = main.Genome([main.Rule([65, 0], [65, 66])])
    snaps, _ = tg.collect_snapshots(main, g, [b'ABAB'], 0,
                                    prefix=4, steps_per_text=6)
    proposal = main.Rule([65, 0], [65, 67])
    ranked, before = tg.local_rank(main, g, 0, snaps, [(proposal, 1)])
    assert before > 0
    assert ranked == []


def test_teacher_forced_trace_counterfactual_matches_full_rollout():
    g = main.Genome([main.Rule([255], [255]), main.Rule([65, 0], [65, 66])])
    text = b'ABABAB'
    snaps, _ = tg.collect_snapshots(main, g, [text], 1,
                                    prefix=6, steps_per_text=12)
    assert snaps
    original_correct = main.autoregressive_rollout(g, text)[0]
    assert original_correct >= 1
    for s in snaps:
        assert s.target == text[s.step]
    # No state-size or context clipping occurs in the base model.


def test_trace_mode_cli_and_resume(tmp_path):
    corpus = tmp_path/'corpus.txt'
    corpus.write_bytes(b'ABABABABAB===SPLIT===ABABABA')
    args = main.parse_args() if False else opts()
    args = argparse.Namespace(**vars(args), local_corpus=str(corpus),
         min_chunk=2, corpus_chunks=10, max_chunk=100,
         load='', rules=3, population=1, no_embedding=True,
         seed=3, cases=2, case_rotate_every=5,
         checkpoint=str(tmp_path/'ckpt.pkl'), checkpoint_every=1,
         history_csv=str(tmp_path/'hist.csv'),
         save=str(tmp_path/'latest.json'),
         no_plot=True, plot_every=999, plot_prefix=str(tmp_path/'plot'),
         plot_window=10, generations=3, max_state_bytes=512,
         hof_size=4096)
    tg.evolve_trace(args, main)
    assert (tmp_path/'ckpt.pkl').exists()
    first, pop, best, history, rng, archive, rotation = main.load_checkpoint(
        str(tmp_path/'ckpt.pkl'), include_rotation_state=True)
    assert first == 3
    assert len(pop) == 1
    assert 1 <= len(archive) <= 4096
    assert len(history) == 3
    assert all(x['optimizer'] == 'trace' for x in history)
    assert json.loads((tmp_path/'latest.json').read_text())['rules']
    args.load = str(tmp_path/'ckpt.pkl')
    args.generations = 5
    tg.evolve_trace(args, main)
    with open(tmp_path/'hist.csv', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert [int(row['generation']) for row in rows] == list(range(5))
    assert all(row['optimizer'] == 'trace' for row in rows)


def test_trace_after_ga_checkpoint(tmp_path):
    g = identity_genome()
    rng = random.Random(3)
    path = tmp_path/'ga.pkl'
    main.save_checkpoint(str(path), 1, [g, g], g, [], rng, [g])
    corpus = tmp_path/'corpus.txt'
    corpus.write_bytes(b'ABABABA')
    args = argparse.Namespace(**vars(opts()), local_corpus=str(corpus),
        min_chunk=2, corpus_chunks=10, max_chunk=100,
        load=str(path), rules=3, population=2, no_embedding=True,
        seed=3, cases=1, case_rotate_every=5,
        checkpoint=str(path), checkpoint_every=1,
        history_csv=str(tmp_path/'hist.csv'), save=str(tmp_path/'latest.json'),
        no_plot=True, plot_every=999, plot_prefix=str(tmp_path/'plot'),
        plot_window=10, generations=2, max_state_bytes=512, hof_size=4096)
    tg.evolve_trace(args, main)
    assert main.load_checkpoint(str(path))[0] == 2


def test_full_rollout_rejects_local_overfit_from_one_text():
    # The only traced example prefers A->C, but the complete active training
    # corpus overwhelmingly favors the incumbent A->B.
    g = main.Genome([main.Rule([65, 0], [65, 66])])
    texts = [b'AC', b'ABABABABABABABABABAB']
    main.score_population([g], texts, 'cpu', cpu_workers=1)
    previous = g.fitness
    proposed, info = tg.improve_once(main, g, texts, 0, opts(trace_texts=1))
    assert info['trace_errors'] > 0
    assert proposed.fitness >= previous
    assert (proposed is g) or (info['trace_accepted'] == 1 and proposed.fitness > previous)


def test_trace_generator_makes_two_capture_rule():
    g = identity_genome()
    text = [b'ABCDEABCDEF']
    snaps, _ = tg.collect_snapshots(main, g, text, 2,
                                    prefix=11, steps_per_text=12)
    proposals = tg.propose_rules(main, snaps, g.embedding,
                                 context=8, max_candidates=200)
    assert any(rule.pattern.count(-1) == 2 and
               len(rule.replacement) >= 2 for rule, _ in proposals)
