"""Regression tests for event co-firing and directed token provenance."""
import argparse
import csv
import random

import main
import rule_dependencies as dep
import trace_guided as tg


def rule(a,b):
    return main.Rule(list(a), list(b))


def test_phi_perfect_positive_negative_independent_and_constant():
    # Samples 0..3: 0011 vs 0011, 1100, 0101
    assert dep.phi_binary(0b1100, 0b1100, 4) == 1.0
    assert dep.phi_binary(0b1100, 0b0011, 4) == -1.0
    assert dep.phi_binary(0b1100, 0b1010, 4) == 0.0
    assert dep.phi_binary(0b1111, 0b0011, 4) == 0.0
    assert dep.phi_binary(0b0, 0b0011, 4) == 0.0
    assert dep.phi_binary(0b111100, 0b100000, 4) == 0.0  # high bits masked


def test_tracks_true_rewrite_dependency_not_just_cofire():
    # First rule writes B; second rule consumes B and changes it to C.
    g = main.Genome([rule([65, 0],[65, 66]), rule([66],[67]),
                     rule([255],[255])])
    graph = dep.collect_dependencies(main, g, [b'ABABABAB'], prefix=8,
                                      steps_per_text=20)
    assert graph.observations == 7
    assert graph.directed[(0, 1)] >= 1
    assert graph.support(0) >= 1 and graph.support(1) >= 1
    assert graph.producer_errors[1] >= 1
    selected, mode, _ = dep.choose_rule(graph, 1, len(g.rules), tail=1,
                                        explore_every=0)
    assert mode == 'dependency'
    assert selected in (0, 1)  # can train a non-tail rule


def test_independent_firing_not_mistaken_for_causal_edge():
    # They fire in the same sweep, but on independent input tokens.
    g = main.Genome([rule([65],[68]), rule([0],[66])])
    graph = dep.collect_dependencies(main, g, [b'ABABAB'], prefix=6,
                                      steps_per_text=20)
    assert graph.cofires(0, 1) >= 1
    assert (0, 1) not in graph.directed


def test_teacher_forcing_resets_prediction_slot_provenance():
    # Rule 0 supplies B to the predicted position, then teacher forcing
    # overwrites it. Rule 1 must not falsely depend on that B next step.
    g = main.Genome([rule([0],[66]), rule([66],[67])])
    graph = dep.collect_dependencies(main, g, [b'ABBBB'], prefix=5,
                                      steps_per_text=20)
    assert graph.directed[(0, 1)] >= 1
    # Graph is collected without manipulating either rule or genome.
    assert g.rules[0].pattern == [0]


def test_reports_phi_and_priorities_as_bounded_csv(tmp_path):
    g = main.Genome([rule([65,0],[65,66]),rule([66],[67])])
    graph = dep.collect_dependencies(main, g, [b'ABABAB'], prefix=6,
                                      steps_per_text=10)
    output = str(tmp_path/'edges')
    dep.export_dependencies(graph, output, 25)
    with open(output + '_edges.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]['source_rule'] == '0'
    assert rows[0]['target_rule'] == '1'
    assert 'phi_effective_fire' in rows[0]
    with open(output + '_rules.csv', newline='') as f:
        rule_rows = list(csv.DictReader(f))
    assert len(rule_rows) == 2
    assert 'dependency_priority' in rule_rows[0]


def test_trace_dependent_training_runs_and_resumes(tmp_path):
    corpus = tmp_path / 'c.txt'
    corpus.write_bytes(b'ABABABABAB===SPLIT===ABABABAB')
    kwargs = dict(trace_tail=1, trace_prefix=20, trace_steps=10,
        trace_state_cap=256, trace_texts=2, trace_context=4,
        trace_candidates=24, trace_confirm=2, backend='cpu',
        mps_batch=4, cpu_workers=1, no_eval_cache=False,
        local_corpus=str(corpus), min_chunk=2, corpus_chunks=10, max_chunk=100,
        load='', rules=3, population=1, no_embedding=True, seed=42,
        cases=2, case_rotate_every=5,
        checkpoint=str(tmp_path / 'ckpt.pkl'), checkpoint_every=1,
        history_csv=str(tmp_path / 'history.csv'),
        save=str(tmp_path / 'latest.json'), no_plot=True, plot_every=100,
        plot_prefix=str(tmp_path / 'plot'), plot_window=10, generations=3,
        max_state_bytes=4096, hof_size=4096, trace_dependencies=True,
        trace_dep_every=2, trace_dep_prefix=20, trace_dep_steps=10,
        trace_dep_texts=2, trace_dep_topk=3, trace_dep_min_support=1,
        trace_dep_explore_every=0,
        trace_dep_graph_prefix=str(tmp_path / 'dep'),
        trace_dep_graph_every=1)
    args = argparse.Namespace(**kwargs)
    tg.evolve_trace(args, main)
    assert (tmp_path / 'dep_edges.csv').is_file()
    assert (tmp_path / 'dep_rules.csv').is_file()
    first, pop, best, history, rng, archive = main.load_checkpoint(args.checkpoint)
    assert first == 3 and len(pop) == 1 and len(archive) <= 4096
    assert all('trace_dep_edges' in row for row in history)
    assert all(0 <= row['trace_row'] < 3 for row in history)
    args.load = args.checkpoint
    args.generations = 5
    tg.evolve_trace(args, main)
    assert main.load_checkpoint(args.checkpoint)[0] == 5


def test_provenance_across_timesteps_even_when_phi_negative():
    # Step 1 only rule 0 changes A -> B. Step 2 only rule 1 changes B B.
    # They never cofire, yet rule 1 consumes a byte produced by rule 0.
    g = main.Genome([rule([65, 0], [66, 0]),
                     rule([66, 66], [67, 66])])
    graph = dep.collect_dependencies(main, g, [b'ABX'], prefix=3,
                                      steps_per_text=2)
    assert graph.observations == 2
    assert graph.cofires(0, 1) == 0
    assert graph.phi(0, 1) == -1.0
    assert graph.directed[(0, 1)] == 1


def test_state_overflow_aborts_trace_without_cropping():
    g = main.Genome([rule([65], [65] * 300)])
    graph = dep.collect_dependencies(main, g, [b'AB'], prefix=2,
                                     steps_per_text=1, state_cap=128)
    assert graph.overflows == 1
    assert graph.observations == 0
    assert len(g.rules[0].replacement) == 300
