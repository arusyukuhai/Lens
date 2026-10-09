"""Regression tests for the one-genome, real-state n-gram hill climber."""
import random
import tempfile
import unittest
from pathlib import Path
import main as m
import native_cpu


class NgramGreedyTests(unittest.TestCase):
    def setUp(self):
        self.data=[b'ABCABCABCABC',b'AABBAABBAABB',b'HELLO HELLO']
        self.rng=random.Random(123)

    def test_top5_count_frequency(self):
        out=m.frequent_templates([b'ABABABXY'],(2,),(),self.rng,5)
        self.assertEqual(out[0],[65,66]) # AB occurs 3 times
        self.assertIn([66,65],out)

    def test_general_wildcard_pattern_from_actual_bytes(self):
        state=[bytes(range(75))]
        shape=(2,3,4,1,2)
        gaps=(2,5,1,3)
        choices=m.frequent_templates(state,shape,gaps,self.rng,top_k=5)
        self.assertTrue(choices)
        self.assertEqual(choices[0].count(-1),4)
        self.assertEqual(sum(v>=0 for v in choices[0]),sum(shape))
        self.assertTrue(any(m._match_at(list(state[0]),pat,k) is not None
                            for pat in choices for k in range(len(state[0]))))

    def test_dynamic_grammar_up_to_six_literal_blocks(self):
        seen=set()
        for _ in range(1000):
            shape=m._grammar_lengths(self.rng,max_blocks=6,max_literal=12)
            seen.add(len(shape))
            self.assertTrue(all(x>0 for x in shape))
        self.assertEqual(seen,{1,2,3,4,5,6})

    def test_native_trace_equals_reference(self):
        g=m.new_genome(10,self.rng,self.data,m._bigram_seeds(self.data))
        npre,npost,ncounts=native_cpu.trace_cpu(g,self.data,2,4,64)
        ppre,ppost,pcounts=m._trace_rule_python(g,self.data,2,4,64)
        self.assertEqual(ncounts.tolist(),pcounts.tolist())
        self.assertEqual(npre,ppre)
        self.assertEqual(npost,ppost)
        self.assertEqual(ncounts.tolist(),native_cpu.evaluate_cpu([g],self.data,10)[0].tolist())

    def test_rule_mutation_is_pure_and_valid(self):
        g=m.new_genome(9,self.rng,self.data,m._bigram_seeds(self.data))
        original=[m._rule_signature(rule) for rule in g.rules]
        pre,post,_=m.trace_rule(g,self.data,3,samples=3,max_bytes=64)
        modes=set()
        for _ in range(100):
            proposal,mode=m.propose_ngram_rule(g.rules[3],pre,post,self.rng)
            self.assertTrue(m.is_valid_rule(proposal))
            modes.add(mode)
        self.assertEqual([m._rule_signature(rule) for rule in g.rules],original)
        self.assertEqual(modes,{'input','output','both','extend'})

    def test_loaded_json_retains_embedding_and_rules(self):
        g=m.new_genome(7,self.rng,self.data,m._bigram_seeds(self.data))
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'example.json'
            m.save_model(str(file),g)
            step,resumed,history,rngstate,rotation,ever=m.load_greedy_checkpoint(str(file))
            self.assertEqual(step,0)
            self.assertEqual(resumed.embedding,g.embedding)
            self.assertEqual(resumed.rules,g.rules)
            self.assertFalse(history)

if __name__=='__main__':
    unittest.main()
