"""Binary capture operators: encode/decode, Python semantics, native CPU parity.
Run with `python -m unittest -v test_binary_operators.py`.
"""
import random
import unittest
import main as m


class BinaryOperatorTests(unittest.TestCase):
    def setUp(self):
        self.lut = list(range(256))
        self.inv = self.lut.copy()

    def test_encoding_is_unique_and_preserves_unary(self):
        seen = set()
        for kind in range(len(m.BINARY_OP_NAMES)):
            for left in range(1, 5):
                for right in range(1, 5):
                    op = m.binary_opcode(kind, left, right)
                    self.assertNotIn(op, seen)
                    seen.add(op)
                    self.assertEqual(m.decode_binary_opcode(op), (kind, left - 1, right - 1))
                    self.assertGreaterEqual(op, -32768)
                    self.assertLessEqual(op, -112)
        self.assertEqual(len(seen), 144)
        for op in range(-111, 0):
            self.assertIsNone(m.decode_binary_opcode(op))

    def test_legality_and_repair(self):
        for kind in range(len(m.BINARY_OP_NAMES)):
            op = m.binary_opcode(kind, 3, 4)
            valid = m.Rule([65, -1, 66, -1, 67, -1, 68, -1, 0], [op])
            invalid = m.Rule([65, -1, 66, -1, 0], [op])
            self.assertTrue(m.is_valid_rule(valid))
            self.assertFalse(m.is_nonexpanding(valid))
            self.assertFalse(m.is_valid_rule(invalid))
            self.assertEqual(m.repair_rule(valid).replacement, [op])
            self.assertNotIn(op, m.repair_rule(invalid).replacement)

    def test_results(self):
        a, b = [3, 1, 3, 2], [3, 2]
        cases = {
            'INTERSECT': [3, 2],
            'DIFF': [1],
            'ZIP_ADD': [6, 3],
            'ZIP_XOR': [0, 3],
            'EQUAL': [0],
            'FILTER_IN': [3, 3, 2],
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                kind = m.BINARY_OP_NAMES.index(name)
                self.assertEqual(m._emit_binary(kind, a, b, self.lut, self.inv), expected)
        self.assertEqual(m._emit_binary(2, [1, 2], [3, 4], self.lut, self.inv), [3, 10, 8])
        self.assertEqual(m._emit_binary(7, [1, 2], [3, 4], self.lut, self.inv), [4, 11, 6])
        self.assertEqual(m._emit_binary(5, [1, 2], [1, 2], self.lut, self.inv), [1])
        self.assertEqual(m._emit_binary(6, [5, 6, 7], [6, 7, 8], self.lut, self.inv), [5, 6, 7, 8])
        self.assertEqual(m._emit_binary(6, [1, 1, 1], [1, 1, 2], self.lut, self.inv), [1, 1, 1, 2])
        self.assertEqual(m._emit_binary(2, [], [1], self.lut, self.inv), [])
        self.assertEqual(m._emit_binary(8, [3, 3, 1], [3], self.lut, self.inv), [3, 3])

    def test_lut_and_rewrite_with_four_captures(self):
        lut = self.lut.copy()
        lut[1], lut[2] = 2, 1
        inverse = m.inverse_lut(lut)
        op = m.binary_opcode(3, 1, 4)  # zip-add($1,$4)
        rule = m.Rule([65, -1, 66, -1, 67, -1, 68, -1, 0], [op])
        state = [65, 1, 66, 9, 67, 9, 68, 2, 0]
        self.assertEqual(m.replace_once(state, rule, lut, inverse), [3])
        self.assertEqual(m._emit_binary(5, [1], [1], lut, inverse), [1])

    def test_native_cpu_matches_python(self):
        import native_cpu
        rng = random.Random(809)
        texts = [b'AXBBXYZCBAA', b'AXBAXBBCA', b'AXBYC', b'XAAXBYA', b'ABABABAB']
        luts = [self.lut, rng.sample(range(256), 256)]
        genomes = []
        for kind in range(len(m.BINARY_OP_NAMES)):
            for lut in luts:
                op = m.binary_opcode(kind, 1, 2)
                genome = m.Genome([m.Rule([65, -1, 66, -1, 0], [65, op, 0])], lut)
                genomes.append(genome)
                self.assertTrue(m.is_valid_rule(genome.rules[0]))
        expected = [[m.autoregressive_rollout(g, x)[0] for x in texts] for g in genomes]
        actual = native_cpu.evaluate_cpu(genomes, texts, 1, workers=2)
        self.assertEqual(actual.tolist(), expected)

    def test_every_pair_actually_changes_native_prediction(self):
        """Strong parity check: target is a binary operator's emitted *last* byte.

        Unlike arbitrary rollouts, this forces all 9x16 opcode variants to fire
        on four nonempty captures and makes correct dispatch observable in ACC.
        """
        import native_cpu
        prefix = [65, 1, 66, 2, 67, 3, 68, 4, 69]
        pat = [65, -1, 66, -1, 67, -1, 68, -1, 69, 0]
        cap = ([1], [2], [3], [4])
        genomes, texts = [], []
        for lut in (self.lut, random.Random(47).sample(range(256), 256)):
            inverse = m.inverse_lut(lut)
            for kind in range(len(m.BINARY_OP_NAMES)):
                for i in range(1, 5):
                    for j in range(1, 5):
                        op = m.binary_opcode(kind, i, j)
                        vals = m._emit_binary(kind, cap[i-1], cap[j-1], lut, inverse)
                        target = vals[-1] if vals else 65
                        g = m.Genome([m.Rule(pat, [65, op])], lut)
                        text = bytes(prefix + [target])
                        self.assertEqual(m.autoregressive_rollout(g, text)[0], 1)
                        genomes.append(g)
                        texts.append(text)
        native_scores = native_cpu.evaluate_cpu(genomes, texts, 1, workers=4)
        for i in range(len(genomes)):
            self.assertEqual(native_scores[i, i], 1, i)

    def test_mps_binary_falls_back_without_kernel_dispatch(self):
        from unittest.mock import patch
        g = m.Genome([m.Rule([65, -1, 66, -1, 0],
                             [65, m.binary_opcode(2, 1, 2), 0])], self.lut)
        with patch.object(m.gpu_backend, 'evaluate_mps',
                          side_effect=AssertionError('binary op reached GPU')):
            m.score_population([g], [b'A\x01B\x02C'], 'mps', use_cache=False)
        self.assertGreaterEqual(g.fitness, 0)

    def test_mutation_can_generate_binaries(self):
        rng = random.Random(64)
        corpus = [b'AAABBBCCCAAA']
        bigrams = m._bigram_seeds(corpus)
        rule = m.Rule([65, -1, 66, -1, 0], [65, -1, 0])
        found = False
        for _ in range(300):
            mutated = m.mutate_rule(rule, rng, corpus, bigrams)
            self.assertTrue(m.is_valid_rule(mutated))
            if any(m.decode_binary_opcode(op) is not None for op in mutated.replacement):
                found = True
                break
        self.assertTrue(found, 'new binary op mutation family never sampled')


if __name__ == '__main__':
    unittest.main()
