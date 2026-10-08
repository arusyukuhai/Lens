"""Regression tests for the MPS bytes crash and native CPU accelerator."""
import random
import unittest
import numpy as np
import main as m
import gpu_replace_persistent as gpu
import native_cpu


class FastFixTests(unittest.TestCase):
    def test_metal_reference_address_spaces(self):
        """Regression guard for Metal's explicit reference address spaces."""
        src = gpu._METAL_SOURCE
        import re
        assert 'device short* a, device short* b, thread int &n,' in src
        # Every reference parameter is explicitly annotated; Python passes a
        # thread-local counter and bool to sweep, while buffer args are constant.
        refs = re.findall(r'\b(?:int|bool)\s*&\s*\w+', src)
        qualifiers = re.findall(r'\b(?:thread|constant|device|threadgroup)\s+(?:int|bool)\s*&\s*\w+', src)
        self.assertEqual(sorted(refs), sorted(re.sub(r'^(?:thread|constant|device|threadgroup)\s+', '', x) for x in qualifiers))

    def test_mps_byte_input_regression(self):
        texts=[b' * that the consumer has already validated the contents of the nvlist, so we',
               b'\0\xff\x80', bytes(range(256)),bytearray(b'abc'),memoryview(b'de'), [255,0,12]]
        packed=gpu._pack_texts(texts)
        for i,s in enumerate(texts):
            self.assertEqual(packed[i,:len(s)].tolist(),list(s))
        self.assertEqual(packed.dtype,np.int16)
        with self.assertRaises(ValueError):
            gpu._pack_texts([[256]])
        with self.assertRaises(ValueError):
            gpu._pack_texts([[-1]])

    def test_native_against_python(self):
        rng=random.Random(2818)
        corpus=[b'ABCXYZABCXYZ',b'\x00\x80\xffXYZ\x00ABC',bytes(range(32))]
        bg=m._bigram_seeds(corpus)
        genomes=[m.new_genome(35,rng,corpus,bg) for _ in range(5)]
        # Additional targeted rule families, including sorted captures and LUT mutation.
        ops=[-1,-16,-32,-48,-64,-80,-96]
        for i,op in enumerate(ops):
            g=m.Genome([m.Rule([1,-1,9],[op])],list(range(256)))
            g.embedding[2],g.embedding[3]=g.embedding[3],g.embedding[2]
            genomes.append(g)
        texts=[corpus[0],corpus[1]]
        for i in range(8):
            texts.append(bytes(rng.randrange(256) for _ in range(5+i*7)))
        # Same rule count across one native batch.
        for group in [genomes[:5],genomes[5:]]:
            k=len(group[0].rules)
            if k==1:
                assert all(len(g.rules)==1 for g in group)
            actual=native_cpu.evaluate_cpu(group,texts,k,workers=3)
            expected=np.array([[m.autoregressive_rollout(g,s)[0] for s in texts]
                               for g in group],dtype=np.int32)
            np.testing.assert_array_equal(actual,expected)

    def test_native_persistent_history_no_context_cap(self):
        lut=list(range(256))
        g=m.Genome([m.Rule([65,0],[88,66]),m.Rule([88,66,0],[88,66,67])],lut)
        for text in [b'ABC', b'AB'*130, b'\x00'*530]:
            got=native_cpu.evaluate_cpu([g],[text],2)[0,0]
            self.assertEqual(got,m.autoregressive_rollout(g,text)[0])

    def test_native_fitness_equality(self):
        corpus=[b'ABCDABCD',b'BCDEFGH',b'ABCZABC']
        rng=random.Random(23)
        bg=m._bigram_seeds(corpus)
        gs=[m.new_genome(40,rng,corpus,bg) for _ in range(6)]
        native=[m.Genome(g.rules.copy(),g.embedding.copy()) for g in gs]
        fallback=[m.Genome(g.rules.copy(),g.embedding.copy()) for g in gs]
        m.score_population(native,corpus,'cpu',cpu_workers=4)
        m.score_population(fallback,corpus,'python')
        self.assertEqual([g.fitness for g in native],[g.fitness for g in fallback])
        self.assertEqual([g.case_scores for g in native],[g.case_scores for g in fallback])


if __name__=='__main__':
    unittest.main()
