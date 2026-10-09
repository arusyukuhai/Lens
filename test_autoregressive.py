"""Run: python -m unittest -v test_autoregressive.py"""
import json
import random
import tempfile
import unittest
from pathlib import Path

import main as m


class ReplacerAutoregressiveTests(unittest.TestCase):
    def setUp(self):
        self.lut=list(range(256))
        self.inv=m.inverse_lut(self.lut)

    def test_only_256_tokens(self):
        assert m.TOKEN_COUNT==256
        rng=random.Random(8)
        corpus=[b'ABCABCDABABBCDDABCD']
        for _ in range(30):
            g=m.new_genome(450,rng,corpus,m._bigram_seeds(corpus))
            self.assertTrue(m.valid_lut(g.embedding))
            self.assertTrue(all(m.is_valid_rule(r) for r in g.rules))
            for r in g.rules:
                self.assertTrue(all(v==-1 or 0<=v<256 for v in r.pattern))
                self.assertTrue(all(v>=m.BINARY_OP_MIN and v<=255 for v in r.replacement))

    def test_operator_lut_only(self):
        lut=list(range(256));lut[1],lut[2]=2,1
        inv=m.inverse_lut(lut)
        self.assertEqual(m.replace_once([8,1,2,9],m.Rule([8,-1,9],[-1]),lut,inv),[1,2])
        self.assertEqual(m.replace_once([8,1,2,9],m.Rule([8,-1,9],[-32]),lut,inv),[2,1])
        self.assertEqual(m.replace_once([8,1,2,9],m.Rule([8,-1,9],[-16]),lut,inv),[2,1])
        self.assertEqual(m.replace_once([8,1,9],m.Rule([8,-1,9],[-48]),lut,inv),[3])
        self.assertEqual(m.replace_once([8,1,9],m.Rule([8,-1,9],[-64]),lut,inv),[2])
        self.assertEqual(m.replace_once([8,1,9],m.Rule([8,-1,9],[-80]),lut,inv),[4])
        self.assertEqual(m.replace_once([8,1,9],m.Rule([8,-1,9],[-96]),lut,inv),[2])

    def test_teacher_forced_prediction(self):
        g=m.Genome([m.Rule([65,0],[65,66])],self.lut)
        n,total,preds,traces=m.autoregressive_rollout(g,b'ABAC',predictions=True,return_states=True)
        self.assertEqual(preds[0],ord('B'))
        self.assertEqual(total,3)
        self.assertEqual(traces[0],[65,66])

    def test_persistent_transformed_history(self):
        # A -> X changes the retained prefix. Next prediction can use X later.
        g=m.Genome([m.Rule([65,0],[88,66]),m.Rule([88,66,0],[88,66,67])],self.lut)
        n,total,preds,traces=m.autoregressive_rollout(g,b'ABC',predictions=True,return_states=True)
        self.assertEqual(preds,[66,67])
        self.assertEqual(n,2)
        self.assertEqual(traces[0],[88,66])
        self.assertEqual(traces[1],[88,66,67])

    def test_no_context_cap(self):
        g=m.Genome([m.Rule([255],[255])],self.lut)
        line=b'A'*520
        c,n,p,traces=m.autoregressive_rollout(g,line,return_states=True)
        self.assertEqual(n,519)
        self.assertEqual(len(traces),519)
        self.assertEqual(len(traces[-1]),520)
        self.assertEqual(len(m.free_generate(g,b'A',520)),520)

    def test_one_sweep_only(self):
        # Rule 2 makes a token that would have matched rule 1, but we NEVER revisit rule 1.
        rules=[m.Rule([66],[67]),m.Rule([65],[66])]
        self.assertEqual(m.sweep([65],rules,self.lut),[66])
        self.assertEqual(m.sweep([66],rules,self.lut),[67])

    def test_random_rule_legality_and_growth(self):
        rng=random.Random(10); corpus=[b'ABABABABAB',b'ABCDEFABCDEF']
        bigrams=m._bigram_seeds(corpus)
        for _ in range(200):
            r=m.repair_rule(m.make_rule(rng,corpus,bigrams),rng)
            self.assertTrue(m.is_valid_rule(r))
            x=[rng.randrange(256) for _ in range(rng.randrange(2,40))]
            y=m.replace_once(x,r,self.lut,self.inv)
            self.assertGreaterEqual(len(y),0)
            self.assertTrue(all(0<=v<256 for v in y))

    def test_checkpoint_model_roundtrip(self):
        corpus=[b'HELLOHELLO',b'ABCABCABC']
        rng=random.Random(5)
        g=m.new_genome(5,rng,corpus,m._bigram_seeds(corpus))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'check.pkl'
            m.save_checkpoint(str(p),4,[g],g,[{'generation':3}],rng)
            gen,pop,best,hist,restored,archive=m.load_checkpoint(str(p))
            self.assertEqual(gen,4)
            self.assertEqual(len(pop),1)
            self.assertEqual(m.genome_object(pop[0]),m.genome_object(g))
            self.assertEqual(restored.random(),rng.random())

    def test_cpu_accuracy(self):
        models=[m.Genome([m.Rule([65,0],[65,66])],self.lut),m.Genome([m.Rule([65,0],[65,67])],self.lut)]
        samples=[b'AB',b'AC',b'AB']
        m.score_population(models,samples,'cpu')
        self.assertEqual(models[0].fitness,2/3)
        self.assertEqual(models[1].fitness,1/3)


if __name__=='__main__':
    unittest.main()
