import random
from types import SimpleNamespace

import numpy as np

import main as m


def _genome(shared, edits=()):
    rules = list(shared)
    for i, token in edits:
        rules[i] = m.Rule([token], [(token + 1) & 255])
    return m.Genome(rules, embedding=list(range(m.EMBEDDING_ENTRY_COUNT)))


def test_threshold_distance_matches_exact_near_decision():
    shared = [m.Rule([i & 255], [(i + 7) & 255]) for i in range(200)]
    base = _genome(shared)
    random.seed(17)
    for _ in range(100):
        ids = random.sample(range(len(shared)), random.randrange(0, 20))
        g = _genome(shared, [(i, random.randrange(256)) for i in ids])
        if random.random() < 0.25:
            g.embedding[random.randrange(m.EMBEDDING_ENTRY_COUNT)] = m.VOCAB - 1
        exact = m.genome_structural_distance(base, g)
        fast = m.genome_structural_distance_below(base, g, 0.015)
        assert (exact < 0.015) == (fast is not None)
        if fast is not None:
            assert abs(exact - fast) < 1e-12


def test_hof_uses_copy_on_write_rule_sharing():
    shared = [m.Rule([1], [2]), m.Rule([3], [4])]
    g = m.Genome(shared, fitness=0.5, case_scores=[0.5])
    entry = m.make_hof_entry(g, (1,), 0, 8, 0.7)
    assert entry.genome.rules[0] is g.rules[0]
    child = m.clone_genome_shallow(entry.genome)
    child.rules[0] = m.Rule([9], [8])
    assert entry.genome.rules[0].pattern == [1]


def test_checkpoint_v3_deduplicates_rules_and_roundtrips(tmp_path):
    shared = [m.Rule([i & 255], [(i + 1) & 255]) for i in range(32)]
    pop = [_genome(shared) for _ in range(6)]
    for i, g in enumerate(pop):
        g.fitness = i / 10.0
        g.case_scores = [i / 10.0]
        g.readout_weights = [float(j) for j in range(len(shared))]
    arrays = {}
    m._pack_genome_group(pop, "pop_", arrays)
    assert arrays["pop_rule_refs"].shape == (6, 32)
    assert len(arrays["pop_pat_offsets"]) - 1 == 32

    sampler = m.CorpusSampler([b"checkpoint regression corpus"], ngram_pool=4, byte_buffer_size=4096)
    args = SimpleNamespace(cases=1, samples=3, max_noise=0.1, max_output=64, hidden_rewrite_phases=1)
    cp = tmp_path / "cp.npz"
    m.save_checkpoint(str(cp), 1, pop, pop[0], [], sampler, args, [])
    sampler2 = m.CorpusSampler([b"x" * 64], ngram_pool=4, byte_buffer_size=4096)
    _, loaded, _, _, _, _ = m.load_checkpoint(str(cp), sampler2)
    assert len(loaded) == len(pop)
    for a, b in zip(pop, loaded):
        assert m.same_genome_structure(a, b)
        np.testing.assert_allclose(a.readout_weights, b.readout_weights)
    assert loaded[0].rules[0] is loaded[1].rules[0]
