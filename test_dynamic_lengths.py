"""v14 unlimited rule length and growing recurrent state regression suite."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import main as m
import native_cpu


class DynamicLengthTests(unittest.TestCase):
    def setUp(self):
        self.lut=list(range(256))
        self.inv=m.inverse_lut(self.lut)

    def test_repair_keeps_long_rules_and_output(self):
        pat=[65]*170+[0]
        rep=[65]*220+[67]
        rule=m.repair_rule(m.Rule(pat,rep),random.Random(1))
        self.assertEqual(rule.pattern,pat)
        self.assertEqual(rule.replacement,rep)
        self.assertTrue(m.is_valid_rule(rule))
        self.assertFalse(m.is_nonexpanding(rule))
        self.assertEqual(m.replace_once(pat,rule,self.lut,self.inv),rep)

    def test_state_grows_and_preserves_history(self):
        # 450-rule hard limit removed for individual rule token sequences,
        # but the usual model still contains 450 rule rows by default.
        g=m.Genome([m.Rule([65,0],[65]*96+[66]),
                    m.Rule([65]*96+[66,0],[67]*96+[68])],self.lut)
        score,total,preds,traces=m.autoregressive_rollout(g,b'ABC',predictions=True,return_states=True)
        self.assertEqual(total,2)
        self.assertEqual(len(traces[0]),97)
        self.assertEqual(traces[0][-1],66)
        self.assertEqual(len(traces[1]),97)
        self.assertEqual(traces[1][-1],68)
        self.assertEqual(score,1)
        self.assertEqual(preds,[66,68])

    def test_duplicate_capture_actually_expands(self):
        rule=m.Rule([65,-1,0],[65,-1,-1,0])
        self.assertTrue(m.is_valid_rule(rule))
        self.assertFalse(m.is_nonexpanding(rule))
        self.assertEqual(m.replace_once([65,66,67,0],rule,self.lut,self.inv),
                         [65,66,67,66,67,0])

    def test_native_matches_python_with_long_rules_and_growth(self):
        genomes=[
            m.Genome([m.Rule([65,0],[65]*96+[66]),
                      m.Rule([65]*96+[66,0],[67]*96+[68])],self.lut),
            m.Genome([m.Rule([65,-1,0],[65,-1,-1,0])],self.lut),
            m.Genome([m.Rule([65]*70+[0],[65]*80+[66]),
                      m.Rule([65,0],[65]*70+[0])],self.lut),
        ]
        cases=[b'ABC',b'ABBC',b'AABCD',b'ABABAB',b'AZZAZ']
        expected=np.asarray([[m.autoregressive_rollout(g,t)[0] for t in cases]
                             for g in genomes],dtype=np.int32)
        actual=[]
        for g in genomes:
            actual.append(native_cpu.evaluate_cpu([g],cases,len(g.rules),workers=1)[0])
        np.testing.assert_array_equal(np.asarray(actual),expected)

    def test_model_checkpoint_accepts_unbounded_length(self):
        g=m.Genome([m.Rule([65,0],[65]*210+[66])],self.lut)
        with tempfile.TemporaryDirectory() as td:
            model=Path(td)/'best.json'
            check=Path(td)/'check.pkl'
            m.save_model(str(model),g)
            restored=m.genome_from_object(json.loads(model.read_text()))
            self.assertEqual(len(restored.rules[0].replacement),211)
            rng=random.Random(3)
            m.save_checkpoint(str(check),7,[g],g,[],rng)
            _,population,best,*_=m.load_checkpoint(str(check))
            self.assertEqual(best.rules[0].replacement,g.rules[0].replacement)
            self.assertEqual(population[0].rules[0].replacement,g.rules[0].replacement)

    def test_mps_falls_back_to_native_for_expansion(self):
        g=m.Genome([m.Rule([65,0],[65,66,67])],self.lut)
        with patch.object(m.gpu_backend,'evaluate_mps',side_effect=AssertionError('MPS must not run')):
            m.score_population([g],[b'AB'],backend='mps',use_cache=False)
        self.assertEqual(g.fitness,0.)  # predicts 67, correct next byte B=66

    def test_mutated_rules_can_exceed_64(self):
        rng=random.Random(16)
        corpus=[bytes(range(90)),b'A'*90+b'\0']
        seeds=m._bigram_seeds(corpus)
        rule=m.Rule([65]*67+[0],[65]*67+[66])
        longest=len(rule.pattern)
        for _ in range(120):
            rule=m.mutate_rule(rule,rng,corpus,seeds)
            self.assertTrue(m.is_valid_rule(rule))
            longest=max(longest,len(rule.pattern),len(rule.replacement))
        self.assertGreater(longest,64)


if __name__ == '__main__':
    unittest.main()
