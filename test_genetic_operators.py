"""Deterministic invariants, true 3-way diffs and genetic restart tests."""
import csv
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import main as m


class Diff3Tests(unittest.TestCase):
    def test_disjoint_insert_and_replace_merge(self):
        # Donor inserts at B-C, target changes D->X; both changes survive.
        b = list('ABCD')
        x = list('ABqCD')
        y = list('ABCX')
        self.assertEqual(m._diff3_sequence(b, x, y, random.Random(1)), list('ABqCX'))

    def test_disjoint_delete_and_edit_merge(self):
        b = list('ABCDE')
        x = list('ABDE')
        y = list('ABCDX')
        self.assertEqual(m._diff3_sequence(b, x, y, random.Random(5)), list('ABDX'))

    def test_conflict_keeps_target(self):
        b, x, y = [1,2,3], [1,4,3], [1,5,3]
        self.assertEqual(m._diff3_sequence(b,x,y,random.Random(5),prefer_target=1.0),y)
        self.assertEqual(m._diff3_sequence(b,x,y,random.Random(5),prefer_target=0.0),x)

    def test_idempotent_and_alignment(self):
        rr=random.Random(9)
        for _ in range(500):
            base=[rr.randrange(5) for _ in range(rr.randrange(12))]
            donor=[rr.randrange(5) for _ in range(rr.randrange(12))]
            target=[rr.randrange(5) for _ in range(rr.randrange(12))]
            self.assertEqual(m._diff3_sequence(base,base,target,rr),target)
            self.assertEqual(m._diff3_sequence(base,donor,base,rr),donor)
            self.assertEqual(m._diff3_sequence(base,donor,donor,rr),donor)
            # Every output must be ordinary tokens; no sentinel emerges from the merge.
            self.assertTrue(all(isinstance(i,int) and 0<=i<5
                                for i in m._diff3_sequence(base,donor,target,rr)))

    def test_lut_permutation_under_diff3(self):
        rng=random.Random(17)
        for _ in range(50):
            base=list(range(256))
            donor=m.mutate_lut(base,rng,True)
            target=m.mutate_lut(base,rng,True)
            child=m._diff3_lut(base,donor,target,rng)
            self.assertTrue(m.valid_lut(child))
            self.assertEqual(m.inverse_lut(child)[child[201]],201)

    def test_diff3_rule_sequence_with_shift(self):
        mk=lambda i: m.Rule([i,0],[i,0])
        b=[mk(i) for i in range(7)]
        x=b[:2]+[mk(100)]+b[2:6]  # insert then delete, same length
        y=b[:5]+[mk(101)]+b[6:]  # unrelated edit
        out=m._diff3_sequence(b,x,y,random.Random(77),signatures=m._rule_signature)
        self.assertEqual([r.pattern[0] for r in out],[0,1,100,2,3,4,101,6][:len(out)])


class GeneticInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus=[b'abcdefgh ijklmnopqrstuvwxyz',b'while text: pass',
                    b'hello \x00byte\nworld']
        cls.bigrams=m._bigram_seeds(cls.corpus)

    def test_operators_hold_fixed_rule_count_and_legality(self):
        rng=random.Random(418)
        bases=[m.new_genome(45,rng,self.corpus,self.bigrams) for _ in range(4)]
        for i in range(140):
            b,d,t=rng.sample(bases,3)
            a=m.breed_three(b,d,t,rng,self.corpus,self.bigrams,True)
            c=m.breed(b,d,rng,self.corpus,self.bigrams,True)
            v=m.mutate_genome(t,rng,self.corpus,self.bigrams,True)
            for g in (a,c,v):
                self.assertEqual(len(g.rules),45)
                self.assertTrue(m.valid_lut(g.embedding))
                self.assertTrue(all(m.is_valid_rule(r) for r in g.rules))
            bases[rng.randrange(4)]=a

    def test_diff3_preserves_disjoint_rule_changes(self):
        rng=random.Random(7)
        baseline=m.new_genome(20,rng,self.corpus,self.bigrams)
        target=m.Genome(baseline.rules.copy(),baseline.embedding.copy())
        donor=m.Genome(baseline.rules.copy(),baseline.embedding.copy())
        donor.rules[4]=m.Rule([67,0],[67,44])
        target.rules[10]=m.Rule([68,0],[68,50])
        # Repeated trials must sometimes transfer disjoint donor change
        # while preserving target's unrelated change.
        found=False
        for _ in range(100):
            out=m.breed_three(baseline,donor,target,rng,self.corpus,self.bigrams,False)
            self.assertEqual(len(out.rules),20)
            if m._rule_signature(out.rules[4]) == m._rule_signature(donor.rules[4]) and \
               m._rule_signature(out.rules[10]) == m._rule_signature(target.rules[10]):
                found=True
                break
        self.assertTrue(found)

    def test_local_and_exploratory_mutations_both_occurring(self):
        rng=random.Random(71)
        counts=[m._mutation_radius(rng,450) for _ in range(10000)]
        self.assertLess(sum(counts)/len(counts),40)
        self.assertGreater(sum(n>=40 for n in counts),0)
        self.assertGreater(sum(n<=6 for n in counts),len(counts)*0.70)

    def test_related_base_preferred_over_unrelated(self):
        rng=random.Random(2)
        base=m.new_genome(12,rng,self.corpus,self.bigrams)
        close=m.Genome(base.rules.copy(),base.embedding.copy())
        close.rules[1]=m.Rule([66,0],[66,67])
        far=m.new_genome(12,rng,self.corpus,self.bigrams)
        self.assertIs(m._choose_diff3_base(base,close,[far,close]),close)

    def test_parent_unmodified_by_offspring(self):
        rng=random.Random(42)
        a,b,c=[m.new_genome(30,rng,self.corpus,self.bigrams) for _ in range(3)]
        initial=[[m._rule_signature(r) for r in g.rules] for g in (a,b,c)]
        _=m.breed_three(a,b,c,rng,self.corpus,self.bigrams,True)
        _=m.mutate_genome(a,rng,self.corpus,self.bigrams,True)
        for g,original in zip((a,b,c),initial):
            self.assertEqual([m._rule_signature(r) for r in g.rules],original)

    def test_checkpoint_restart_diff3(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            corpus=root/'data.txt'
            corpus.write_bytes(b'===SPLIT==='.join(self.corpus))
            def run(tag,until,load=None):
                path=str(root/f'{tag}.pkl')
                cmd=[sys.executable,str(Path(m.__file__)),'--backend','cpu',
                     '--local-corpus',str(corpus),'--population','12','--rules','9',
                     '--cases','2','--case-rotate-every','2','--generations',str(until),
                     '--diff3-rate','1','--crossover-rate','1',
                     '--cpu-workers','2','--no-tqdm','--no-plot','--seed','123',
                     '--checkpoint',path,'--checkpoint-every','2',
                     '--save',str(root/f'{tag}.json'),
                     '--history-csv',str(root/f'{tag}.csv')]
                if load: cmd += ['--load',load]
                subprocess.run(cmd,cwd=root,check=True,capture_output=True,text=True)
                return path
            run('continuous',7)
            ck=run('restarted',3)
            run('restarted',7,ck)
            self.assertEqual(json.loads((root/'continuous.json').read_text()),
                             json.loads((root/'restarted.json').read_text()))
            with (root/'continuous.csv').open() as f: a=list(csv.DictReader(f))
            with (root/'restarted.csv').open() as f: b=list(csv.DictReader(f))
            self.assertEqual([(r['best_accuracy'],r['mean_accuracy']) for r in a],
                             [(r['best_accuracy'],r['mean_accuracy']) for r in b])


if __name__=='__main__':
    unittest.main()
