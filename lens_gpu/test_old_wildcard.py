"""Hybrid integration tests; old reference is independent of accelerated engine."""
import random
import unittest
import torch
from torch.nn import functional as F
if __package__:
    from .model import Lens,Config,SerialRule,discrete_features
else:
    from model import Lens,Config,SerialRule,discrete_features
if __package__:
    from .old_reference import discrete_replace as reference
else:
    from old_reference import discrete_replace as reference
if __package__:
    from .old_replace import replace_with_count
else:
    from old_replace import replace_with_count


def set_vertex(rule,lhs,rhs):
    assert lhs and len(lhs)<=rule.max_pattern and len(rhs)<=rule.max_replacement
    subset=tuple(i for i,t in enumerate(lhs) if t<0)
    assert [-int(t) for t in lhs if t<0]==list(range(1,len(subset)+1))
    ix=rule._choices[len(lhs)-1].index(subset)
    with torch.no_grad():
        for p in (rule.pattern_bytes,rule.replacement_tokens,
                  rule.pattern_length,rule.replacement_length,rule.pattern_stars):p.fill_(-45.)
        rule.pattern_length[len(lhs)-1]=45.
        rule.replacement_length[len(rhs)]=45.
        rule.pattern_stars[len(lhs)-1,ix]=45.
        for j,t in enumerate(lhs):
            if t>=0:rule.pattern_bytes[j,t]=45.
        for j,t in enumerate(rhs):
            rule.replacement_tokens[j,t if t>=0 else rule.alphabet-t-1]=45.


class TestOldExact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_old_revision_edge_cases(self):
        for raw,a,b,want in [
            (b'abXYZ', (97,98,-1),(81,),b'Q'),
            (b'a12b34',(97,-1,98,-1),(-2,-1),b'3412'),
            (b'a12345b',(97,-1,98),(81,),b'Q'),
            (b'ab',(-1,),(81,),b'ab'),
            (b'aZZb',(97,-1,98),(81,),b'Q'),
            (b'abcabc',(97,98),(90,90,90),b'ZZZcZZZc'),
            (b'axbyc',(97,-1,98,-2,99),(-2,-1),b'yx'),
        ]:
            with self.subTest(src=raw,a=a,b=b):
                self.assertEqual(reference(raw,a,b),tuple(want))
                out,count=replace_with_count(raw,a,b)
                self.assertEqual(out,want)
                self.assertEqual(count, 0 if raw==want else (2 if a==(97,98) else 1))

    def test_reference_differential_random(self):
        rng=random.Random(291)
        for i in range(450):
            data=bytes(rng.choices(b'abcde',k=rng.randrange(1,32)))
            n=rng.randrange(1,7)
            maxstars=min(4,n)
            chosen=sorted(rng.sample(range(n),rng.randrange(maxstars+1)))
            lhs=[rng.choice(b'abcde') for _ in range(n)]
            for j in chosen:lhs[j]=-1
            k=rng.randrange(6)
            rhs=[rng.choice(list(b'abcde')+list(range(-max(1,len(chosen)),-0))) if rng.random()<0.7
                 else rng.choice(b'abcde') for _ in range(k)]
            expected=reference(data,lhs,rhs)
            actual,count=replace_with_count(data,tuple(lhs),tuple(rhs))
            self.assertEqual(tuple(actual),expected,(i,data,lhs,rhs))
            if count:self.assertNotEqual(count,0)

    def test_hard_lens_matches_original_unbounded(self):
        m=Lens(Config(rules=2,hidden=8,max_pattern=3,max_replacement=3,
                      max_capture=1,max_stars=2,checkpoint_every=0)).double()
        set_vertex(m.rules[0],[97,98,-1],[88,-1])
        set_vertex(m.rules[1],[88,-1],[81,-1])
        exported=m.export_rules()
        rows=[b'ab0123456789012345',b'zzab123456789ab98765']
        cap=40
        tokens=torch.tensor([list(x)+[0]*(cap-len(x)) for x in rows])
        lengths=torch.tensor([len(x) for x in rows],dtype=torch.float64)
        f,ln=m.features(tokens,lengths,hard=True,return_lengths=True)
        for j,row in enumerate(rows):
            ff,state=discrete_features(row,exported)
            self.assertEqual(tuple(state),reference(reference(tuple(row),(97,98,-1),(88,-1)),(88,-1),(81,-1)))
            torch.testing.assert_close(f[j],torch.tensor(ff,dtype=torch.float64))
            self.assertEqual(int(ln[j]),len(state))

    def test_pure_wildcards_noop(self):
        m=Lens(Config(rules=1,hidden=8,max_pattern=2,max_replacement=2,
                      max_capture=2,max_stars=2,checkpoint_every=0)).double()
        set_vertex(m.rules[0],[-1,-2],[88])
        feat=m.features(torch.tensor([[97,98,99,0]]),torch.tensor([3.]),hard=True)
        self.assertEqual(float(feat.sum()),0.)
        ff,st=discrete_features(b'abc',m.export_rules())
        self.assertEqual(ff,[0.]);self.assertEqual(bytes(st),b'abc')

    def test_bounded_surrogate_vertex_cases(self):
        rule=SerialRule(256,4,4,3,3).double()
        cases=[(b'abcxxx', [97,-1,98],[90,-1]),
               (b'abXY',[97,98,-1],[81]),
               (b'a12b3c',[97,-1,98,-2],[88,-2,-1]),
               (b'axby',[97,-1,98,-2],[-2,-1]),
               (b'abcd',[97,98],[88,89,90]),
               (b'abc',[-1,-2],[80])]
        for data,lhs,rhs in cases:
            with self.subTest(data=data,lhs=lhs,rhs=rhs):
                set_vertex(rule,lhs,rhs)
                cap=40
                x=torch.tensor([list(data)+[0]*(cap-len(data))])
                with torch.no_grad():
                    z,n,f=rule(F.one_hot(x,256).double(),torch.tensor([len(data)],dtype=torch.double),
                               torch.tensor([len(data)],dtype=torch.double),hard=True)
                expected=reference(data,lhs,rhs)
                self.assertEqual(bytes(z[0,:int(n[0])].argmax(-1).tolist()),bytes(expected))

    def test_two_derivatives(self):
        torch.manual_seed(7)
        rule=SerialRule(3,2,2,1,2).double()
        z=torch.randn(1,5,3,dtype=torch.double,requires_grad=True)
        lens=torch.tensor([3.3],dtype=torch.double,requires_grad=True)
        def fn(q,l):return rule(q.softmax(-1),l,torch.full_like(l,5.))
        self.assertTrue(torch.autograd.gradcheck(fn,(z,lens),atol=7e-4,rtol=3e-3))
        self.assertTrue(torch.autograd.gradgradcheck(fn,(z,lens),atol=3e-3,rtol=2e-2))

    def test_batch_soft_checkpoint(self):
        m=Lens(Config(rules=2,hidden=8,max_pattern=2,max_replacement=2,
                      max_capture=1,max_stars=2,checkpoint_every=2)).double()
        m.initialize_from_training([b'aabbcc',b'bbccdd'])
        tokens=torch.tensor([[97,98,99,0,0,0],[97,98,97,99,0,0]])
        lens=torch.tensor([3.,4.],dtype=torch.double)
        m.eval(); y=m(tokens,lens)
        for j in range(2):torch.testing.assert_close(y[j:j+1],m(tokens[j:j+1],lens[j:j+1]))
        m.train();loss=m(tokens,lens).square().sum();loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))


if __name__=='__main__':unittest.main(verbosity=2)
