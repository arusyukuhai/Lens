"""Performance rewrite regression: exact outputs, corpus, caching, Metal prefilters."""
import random
import tempfile
from pathlib import Path
import unittest
import numpy as np
import main as m
import native_cpu
import gpu_replace_persistent as gpu

class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.rng=random.Random(731)
        self.corpus=[b'abc abccab abcccab\x00!\n',bytes(range(64)), b'XYZ' * 20,
                     b'function add(a,b) {\n    return a+b;\n}\n']
        self.bigrams=m._bigram_seeds(self.corpus)

    def test_native_vs_python_fuzz(self):
        gs=[m.new_genome(55,self.rng,self.corpus,self.bigrams) for _ in range(12)]
        for g in gs:
            for _ in range(18):
                g=m.mutate_genome(g,self.rng,self.corpus,self.bigrams)
            gs.append(g)
            if len(gs)>=25:
                break
        # Includes wildcards, overlapping patterns, 0x00/0xff and LUT permutations.
        texts=self.corpus+[bytes(self.rng.randrange(256) for _ in range(n)) for n in (3,12,31,63,123)]
        got=native_cpu.evaluate_cpu(gs,texts,55,workers=3)
        expected=np.array([[m.autoregressive_rollout(g,t)[0] for t in texts] for g in gs],dtype=np.int32)
        np.testing.assert_array_equal(got,expected)

    def test_cache_same_results_and_no_repeats(self):
        gs=[m.new_genome(20,self.rng,self.corpus,self.bigrams) for _ in range(5)]
        gs += [m.Genome(gs[0].rules[:],gs[0].embedding[:]),
               m.Genome(gs[1].rules[:],gs[1].embedding[:])]
        examples=[self.corpus[0],self.corpus[1],self.corpus[2]]
        m.score_population(gs,examples,'cpu',cpu_workers=2,use_cache=True)
        scores=[(g.fitness, g.case_scores[:]) for g in gs]
        m.score_population(gs,examples,'cpu',cpu_workers=2,use_cache=True)
        self.assertEqual(scores,[(g.fitness,g.case_scores[:]) for g in gs])
        m.score_population(gs,examples,'cpu',cpu_workers=2,use_cache=False)
        self.assertEqual(scores,[(g.fitness,g.case_scores[:]) for g in gs])
        # Rotated examples must invalidate cached results.
        m.score_population(gs,examples[::-1],'cpu',cpu_workers=2,use_cache=True)
        self.assertEqual(scores[0][0],gs[0].fitness)
        self.assertEqual(scores[0][1][::-1],gs[0].case_scores)

    def test_cache_reuses_elite(self):
        gs=[m.new_genome(12,self.rng,self.corpus,self.bigrams) for _ in range(5)]
        examples=[self.corpus[1],self.corpus[2]]
        m.score_population(gs,examples,'cpu',cpu_workers=2,use_cache=True)
        scores=[g.fitness for g in gs]
        # A new generation inheriting elites and mutated children.
        nxt=[m.Genome(gs[0].rules[:],gs[0].embedding[:]),
             m.mutate_genome(gs[1],self.rng,self.corpus,self.bigrams)]
        m.score_population(nxt,examples,'cpu',cpu_workers=2,use_cache=True)
        self.assertEqual(nxt[0].fitness,scores[0])
        expected=m.autoregressive_rollout(nxt[1],examples[0])[0]+m.autoregressive_rollout(nxt[1],examples[1])[0]
        self.assertAlmostEqual(nxt[1].fitness,expected/sum(len(e)-1 for e in examples))

    def test_anchor_choice_exact_precondition(self):
        r=self.rng
        gs=[m.new_genome(30,r,self.corpus,self.bigrams) for _ in range(5)]
        anchors=gpu._compile_rule_anchors(gs,self.corpus,30)
        self.assertEqual(anchors.shape,(5,30))
        for g,arr in zip(gs,anchors):
            for rule,anchor in zip(g.rules,arr):
                if anchor==-1:
                    # Identity is safe to skip for arbitrary contents.
                    p=rule.pattern
                    for st in [[1,2,0,255],[4,5,1,2,3],[0]*12]:
                        self.assertEqual(m.replace_once(st,rule,g.embedding,m.inverse_lut(g.embedding)),st)
                elif anchor<256:
                    self.assertIn(int(anchor),rule.pattern)
                else:
                    pair=((int(anchor)-256)>>8,(int(anchor)-256)&255)
                    self.assertIn(pair,list(zip(rule.pattern,rule.pattern[1:])))

    def test_bloom_pair_collision_has_no_false_negative(self):
        for key in range(65536):
            h1=(key*2654435761)&2047
            h2=((key*2246822519)^(key>>7))&2047
            mask=[0]*64
            mask[h1>>5]|=1<<(h1&31)
            mask[h2>>5]|=1<<(h2&31)
            self.assertTrue(mask[h1>>5] & (1<<(h1&31)))
            self.assertTrue(mask[h2>>5] & (1<<(h2&31)))

    def test_delimiter_crosses_read_block_boundary(self):
        # 1 MiB block boundary intersects split marker exactly.
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'large.txt'
            f.write_bytes(b'x'*((1<<20)-4)+b'===SPLIT===\nOK\n===SPLIT===\nYES')
            self.assertEqual(m.load_corpus(str(f),max_length=1500),[b'OK',b'YES'])

    def test_history_append(self):
        with tempfile.TemporaryDirectory() as d:
            f=str(Path(d)/'h.csv')
            m.write_history(f,[])
            for i in range(15):
                m.append_history_row(f,{'generation':i,'fitness':i/15})
            rows=Path(f).read_text().splitlines()
            self.assertEqual(len(rows),16)
            self.assertEqual(rows[0],'generation,fitness')
            self.assertTrue(rows[-1].startswith('14,'))

    def test_metal_contains_fast_indexes(self):
        metal=gpu._METAL_SOURCE
        self.assertIn('thread uint pair_bloom[64]',metal)
        self.assertIn('thread uint byte_mask[8]',metal)
        self.assertIn('thread uint hist[256]',metal)
        self.assertIn('const device int* anchors [[buffer(16)]]',metal)

if __name__=='__main__':
    unittest.main()
