import math
import random
import unittest

import numpy as np

import main


class V60InferenceTests(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        np.random.seed(7)

    def test_budget_stays_below_hard_cap(self):
        cases, pop, gens, work = main.resolve_inference_budget(200, 3, 12, 4, 0.75)
        self.assertEqual(cases, 3)
        self.assertEqual(gens, 4)
        self.assertLessEqual(work, int(200 * 0.85))
        self.assertLessEqual(cases * pop, 200)
        self.assertGreaterEqual(pop, 2)

    def test_string_mutation_only_touches_corrupted_loci(self):
        corpus = [b"abcdefghijklmnopqrstuvwxyz" * 4]
        sampler = main.CorpusSampler(corpus, ngram_pool=16, byte_buffer_size=4096)
        case = main.RollingInferenceSet(corpus, sampler, 1, 64, 0.10).slots[0]
        original = list(case.noisy)
        candidate = list(original)
        for _ in range(20):
            candidate = main.mutate_inference_candidate(candidate, case.mutable_positions, sampler)
        mutable = set(case.mutable_positions)
        for i, (a, b) in enumerate(zip(original, candidate)):
            if i not in mutable:
                self.assertEqual(a, b)

    def test_pareto_rank_keeps_tradeoff_front(self):
        def g(f, a):
            x = main.Genome([main.Rule([1], [1])], fitness=f)
            x.inference_accuracy = a
            return x

        pop = [g(0.90, 0.20), g(0.80, 0.80), g(0.70, 0.10), g(0.85, 0.15)]
        fronts = main.assign_pareto_metrics(pop)
        front0 = set(fronts[0])
        self.assertIn(0, front0)
        self.assertIn(1, front0)
        self.assertNotIn(2, front0)
        self.assertNotIn(3, front0)
        self.assertGreater(pop[2].pareto_rank, 0)

    def test_moving_average_starts_when_new_metric_appears(self):
        x = main._moving_average([float("nan"), float("nan"), 0.2, 0.4], 3)
        self.assertTrue(math.isnan(x[0]))
        self.assertTrue(math.isnan(x[1]))
        self.assertAlmostEqual(x[2], 0.2)
        self.assertAlmostEqual(x[3], 0.3)

    def test_checkpoint_pack_roundtrip_preserves_inference_fields(self):
        g = main.Genome([main.Rule([1], [1])], fitness=0.75, case_scores=[0.7])
        g.readout_weights = [0.25]
        g.inference_accuracy = 0.625
        g.inference_full_accuracy = 0.95
        g.pareto_rank = 2
        g.pareto_crowding = 1.5
        arrays = {}
        main._pack_genome_group([g], "x_", arrays)
        class Z(dict):
            @property
            def files(self):
                return list(self.keys())
        out = main._unpack_genome_group(Z(arrays), "x_")[0]
        self.assertAlmostEqual(out.inference_accuracy, 0.625)
        self.assertAlmostEqual(out.inference_full_accuracy, 0.95)
        self.assertEqual(out.pareto_rank, 2)
        self.assertAlmostEqual(out.pareto_crowding, 1.5)


class V60CheckedRegressionTests(unittest.TestCase):
    def setUp(self):
        random.seed(11)
        np.random.seed(11)

    def test_default_shared_budget_counts_ensemble_and_final_pass(self):
        cases, pop, gens, work = main.resolve_inference_budget(
            200, 3, 12, 4, 0.75,
            genome_count=450, outer_evaluated_genomes=450, ensemble_size=64,
        )
        self.assertEqual((cases, pop, gens), (3, 12, 4))
        expected = math.ceil(3 * 12 * (1.0 + 4 * 64 / 450))
        self.assertEqual(work, expected)
        self.assertLessEqual(work, math.floor(200 * 0.85))

    def test_constant_pareto_axes_do_not_create_arbitrary_infinite_crowding(self):
        pop = []
        for _ in range(5):
            g = main.Genome([main.Rule([1], [1])], fitness=0.5)
            g.inference_accuracy = 0.25
            pop.append(g)
        fronts = main.assign_pareto_metrics(pop)
        self.assertEqual(set(fronts[0]), set(range(5)))
        self.assertTrue(all(g.pareto_crowding == 0.0 for g in pop))

    def test_local_corpus_long_chunks_are_cropped_not_dropped(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.txt"
            path.write_bytes(b"A" * 200 + b"===SPLIT===" + b"B" * 12)
            chunks = main.load_local_corpus(str(path), limit=10, min_len=8, max_len=64)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 64)
        self.assertEqual(len(chunks[1]), 12)


    def test_stream_loader_long_code_is_cropped_without_undefined_name(self):
        import sys
        import types
        from unittest.mock import patch
        fake = types.ModuleType("datasets")
        fake.load_dataset = lambda *args, **kwargs: [{"code": "x" * 120}]
        with patch.dict(sys.modules, {"datasets": fake}):
            chunks = main.stream_github_code(limit=1, min_len=8, max_len=32)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 32)


    def test_effective_span_treats_zero_max_chunk_as_unbounded(self):
        from types import SimpleNamespace
        args = SimpleNamespace(max_chunk=0, trajectory_len=64)
        self.assertEqual(main.effective_trajectory_span(args, [b"x" * 200]), 64)
        args = SimpleNamespace(max_chunk=0, trajectory_len=0)
        self.assertEqual(main.effective_trajectory_span(args, [b"x" * 200]), 200)

    def test_inner_mps_rule_pack_is_reused_across_search_rounds(self):
        class FakeEvaluator:
            def __init__(self):
                self.pack_calls = 0
                self.max_output = 128
                self.inputs = []
                self.last_profile = {}
            def __getattr__(self, name):
                if name == "last_profile":
                    return {}
                return 0
            def _pack_population(self, genomes, pool_live_genomes=None):
                self.pack_calls += 1
                return ("packed", self.pack_calls)
            def set_inputs(self, inputs):
                self.inputs = list(inputs)
            def evaluate_population(self, genomes, **kwargs):
                return [main.trajectory_features_cpu(self.inputs, g, self.max_output)[:3]
                        for g in genomes]

        corpus = [b"abc abc abc abc", b"abd abd abd abd", b"abe abe abe abe"]
        sampler = main.CorpusSampler(corpus, ngram_pool=8, byte_buffer_size=4096)
        infset = main.RollingInferenceSet(corpus, sampler, cases=2, span=16, noise_rate=0.10)
        genomes = []
        for i in range(4):
            g = main.Genome([main.Rule([ord('a')], [ord('a')])], fitness=1.0 - i * 0.01)
            g.readout_weights = [1.0]
            genomes.append(g)
        ev = FakeEvaluator()
        main.run_string_inference_ga(
            genomes, infset, sampler, population_size=4, generations=4,
            ensemble_size=2, backend="mps", max_output=128,
            mps_evaluator=ev, restore_inputs=[list(b"abc")],
            pool_live_genomes=genomes,
        )
        # One pack for the search ensemble, one for final all-genome scoring.
        self.assertEqual(ev.pack_calls, 2)


class V60TieHandlingTests(unittest.TestCase):
    def test_inner_rank_ties_are_neutral(self):
        x = np.asarray([[2.0, 2.0, 2.0], [1.0, 3.0, 3.0]])
        r = main._row_rank01(x)
        np.testing.assert_allclose(r[0], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(r[1], [0.0, 0.75, 0.75])

class V60ParetoVectorizationTests(unittest.TestCase):
    def test_vectorized_pareto_ranks_match_bruteforce(self):
        random.seed(23)
        pop = []
        points = []
        for _ in range(40):
            f = random.randrange(8) / 7.0
            a = random.randrange(6) / 5.0
            g = main.Genome([main.Rule([1], [1])], fitness=f)
            g.inference_accuracy = a
            pop.append(g)
            points.append((f, a))
        main.assign_pareto_metrics(pop)

        remaining = set(range(len(points)))
        expected = [None] * len(points)
        rank = 0
        while remaining:
            front = []
            for i in remaining:
                fi, ai = points[i]
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    fj, aj = points[j]
                    if fj >= fi and aj >= ai and (fj > fi or aj > ai):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            for i in front:
                expected[i] = rank
            remaining.difference_update(front)
            rank += 1
        self.assertEqual([g.pareto_rank for g in pop], expected)



class V60ElitePreservationTests(unittest.TestCase):
    def test_inner_elites_leave_one_offspring_slot(self):
        ids = main._inference_elite_ids([0.9, 0.8], elite_count=99)
        self.assertEqual(ids.tolist(), [0])

    def test_inner_elites_are_stable_and_top_ranked(self):
        ids = main._inference_elite_ids([0.5, 0.9, 0.9, 0.1], elite_count=2)
        # Stable descending order: tied ids 1 and 2 retain source order.
        self.assertEqual(ids.tolist(), [1, 2])

    def test_outer_elites_pin_pareto_knee_and_both_objective_extremes(self):
        def g(f, a):
            x = main.Genome([main.Rule([1], [1])], fitness=f)
            x.inference_accuracy = a
            return x

        pop = [g(0.99, 0.10), g(0.80, 0.80), g(0.10, 0.99), g(0.60, 0.20)]
        main.assign_pareto_metrics(pop)
        elites = main.select_outer_elites(pop, 3)
        self.assertIn(max(pop, key=lambda x: x.fitness), elites)
        self.assertIn(max(pop, key=lambda x: x.inference_accuracy), elites)
        self.assertIn(main.choose_pareto_knee(pop), elites)

    def test_outer_elite_clone_is_structurally_unchanged(self):
        g = main.Genome([main.Rule([1, 2], [2])], fitness=0.9)
        g.inference_accuracy = 0.7
        main.assign_pareto_metrics([g])
        source = main.select_outer_elites([g], 1)[0]
        clone = main.clone_genome_shallow(source)
        self.assertTrue(main.same_genome_structure(source, clone))
        self.assertEqual(source.embedding, clone.embedding)


class V601InferenceZeroRegressionTests(unittest.TestCase):
    def setUp(self):
        random.seed(31)
        np.random.seed(31)

    def test_flat_model_scores_do_not_bias_to_untouched_noisy_candidate(self):
        from unittest.mock import patch
        corpus = [b"abcdefg abcdefg abcdefg"]
        sampler = main.CorpusSampler(corpus, ngram_pool=32, byte_buffer_size=4096)
        infset = main.RollingInferenceSet(corpus, sampler, cases=1, span=16, noise_rate=0.01)
        case = infset.slots[0]
        self.assertEqual(len(case.mutable_positions), 1)
        pos = case.mutable_positions[0]
        g = main.Genome([main.Rule([ord('a')], [ord('a')])], fitness=0.9)
        g.readout_weights = [0.0]

        def always_clean(candidate, p, _sampler):
            return int(case.clean[int(p)])

        def all_tied(genomes, candidates, backend, max_output, **kwargs):
            return np.zeros((len(genomes), len(candidates)), dtype=np.float64)

        with patch.object(main, "_propose_inference_byte", side_effect=always_clean), \
             patch.object(main, "evaluate_inference_candidates", side_effect=all_tied):
            stats = main.run_string_inference_ga(
                [g], infset, sampler, population_size=4, generations=2,
                ensemble_size=1, backend="cpu", max_output=128, case_count=1,
                elite_count=1,
            )

        # Old np.argmax always selected candidate 0 (the untouched noisy string),
        # producing exactly 0.0.  Uniform treatment of equal maxima must not have
        # this list-order bias.
        self.assertGreater(g.inference_accuracy, 0.0)
        self.assertGreater(stats["oracle_accuracy"], 0.0)

    def test_checkpoint_context_model_is_rebuilt_from_persisted_ngrams(self):
        sampler = main.CorpusSampler([b"ab" * 32], ngram_pool=8, byte_buffer_size=4096)
        sampler.ngrams = [[1, 2, 3], [1, 2, 4]]
        sampler._rebuild_context_model()
        self.assertGreater(sampler._context_pairs[1, 2], sampler._context_pairs[1, 7])
        self.assertGreater(sampler._context_pairs[2, 3], sampler._context_pairs[2, 7])


class V602LogUniformInferenceMutationTests(unittest.TestCase):
    def setUp(self):
        random.seed(41)
        np.random.seed(41)

    def test_log_uniform_mutation_count_geometry(self):
        from unittest.mock import patch
        n = 16
        with patch.object(main.random, "uniform", return_value=0.0):
            self.assertEqual(main.sample_inference_mutation_count(n), 1)
        with patch.object(main.random, "uniform", return_value=math.log(float(n)) / 2.0):
            self.assertEqual(main.sample_inference_mutation_count(n), 4)
        with patch.object(main.random, "uniform", return_value=math.log(float(n))):
            self.assertEqual(main.sample_inference_mutation_count(n), n)

    def test_focus_is_included_without_collapsing_log_uniform_radius(self):
        from unittest.mock import patch
        candidate = [10, 20, 30, 40, 50, 60, 70]
        positions = [1, 2, 3, 4, 5]
        sampler = main.CorpusSampler([b"abcdef" * 8], ngram_pool=8, byte_buffer_size=4096)

        def bump(cand, pos, _sampler):
            return (int(cand[int(pos)]) + 1) & 255

        with patch.object(main, "sample_inference_mutation_count", return_value=4), \
             patch.object(main, "_propose_inference_byte", side_effect=bump):
            out = main.mutate_inference_candidate(
                candidate, positions, sampler, focus_position=3
            )

        changed = [i for i, (a, b) in enumerate(zip(candidate, out)) if a != b]
        self.assertEqual(len(changed), 4)
        self.assertIn(3, changed)
        self.assertTrue(set(changed).issubset(set(positions)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
