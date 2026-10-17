import unittest
import torch
from model import Config,Lens,SerialRule,reach_parallel,discrete_features
from data import diffusion_batch,alphabet_of,split_corpus
from train import anneal

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_scan_reference_and_second_derivative(self):
        torch.manual_seed(12)
        q=(torch.rand(1,6,3,dtype=torch.double)*.08).requires_grad_()
        def reference(q):
            r=[torch.ones_like(q[:,0,0])]
            for i in range(1,q.shape[1]):
                value=r[i-1]*(1-q[:,i-1,1:].sum(-1))
                for d in range(2,q.shape[2]+1):
                    if i>=d: value=value+r[i-d]*q[:,i-d,d-1]
                r.append(value)
            return torch.stack(r,1)
        torch.testing.assert_close(reach_parallel(q),reference(q))
        self.assertTrue(torch.autograd.gradcheck(reach_parallel,(q,)))
        self.assertTrue(torch.autograd.gradgradcheck(reach_parallel,(q,)))

    def test_dense_uncertain_input_second_derivative(self):
        torch.manual_seed(17)
        rule=SerialRule(3,1).double()
        logits=torch.randn(1,4,3,dtype=torch.double,requires_grad=True)
        length=torch.tensor([3.35],dtype=torch.double,requires_grad=True)
        def fn(z,l):
            x,n,f=rule(z.softmax(-1),l,torch.ones_like(l)*4)
            return x,n,f
        self.assertTrue(torch.autograd.gradcheck(fn,(logits,length),atol=2e-4))
        self.assertTrue(torch.autograd.gradgradcheck(fn,(logits,length),atol=3e-4))
        y,n,f=fn(logits,length)
        g=torch.autograd.grad(y.square().sum()+n.sum()+f.sum(),tuple(rule.parameters()),create_graph=True)
        h=torch.autograd.grad(sum(v.square().sum() for v in g),tuple(rule.parameters()))
        self.assertTrue(all(torch.isfinite(v).all() for v in g+h))

    def test_hard_serial_semantics(self):
        torch.manual_seed(34)
        m=Lens(Config(4,16,2,0)).double()
        patterns=[(97,98,99),(256,120,256),(257,-1,100),(98,-1,99)]
        with torch.no_grad():
            for r,(a,b,c) in zip(m.rules,patterns):
                r.outer.fill_(-20); r.middle.fill_(-20)
                r.outer[0,a]=20; r.outer[1,c]=20; r.middle[r.alphabet if b==-1 else b]=20
        samples=[b'abcxabcYd',b'bbcbdcabc',b'abcabcabc',b'aaaa',b'bccbc']
        rng=__import__('random').Random(5)
        samples += [bytes(rng.choices(b'abcdxY',k=rng.randrange(3,25))) for _ in range(60)]
        export=m.export_rules()
        for c in samples:
            x=torch.tensor([list(c)]); l=torch.tensor([len(c)])
            actual=m.features(x,l,hard=True)[0]
            expected,_=discrete_features(c,export)
            torch.testing.assert_close(actual,torch.tensor(expected,dtype=torch.double),atol=1e-9,rtol=1e-9)
        # A later rule must see the marker produced by the earlier rule.
        f,_=discrete_features(b'abcxabcYd',export)
        self.assertGreater(f[1],0)

    def test_batch_and_checkpoint_equivalence(self):
        torch.manual_seed(23)
        m=Lens(Config(5,16,1,2)).double()
        m.initialize_from_training([b'abcdefabcdef',b'abcxyabcxyzz'])
        x=torch.randint(97,103,(2,12)); l=torch.tensor([12.,8.],dtype=torch.double)
        m.eval(); bat=m(x,l)
        singles=torch.cat([m(x[i:i+1],l[i:i+1]) for i in range(2)])
        torch.testing.assert_close(bat,singles)
        m.train(); m(x,l).square().sum().backward()
        grads=[p.grad.clone() for p in m.parameters()]
        m.zero_grad(); m.config.checkpoint_every=0
        m(x,l).square().sum().backward()
        for a,p in zip(grads,m.parameters()): torch.testing.assert_close(a,p.grad)
        self.assertTrue(all(g.abs().sum()>0 for g in grads))

    def test_diffusion_and_split(self):
        docs=[b'abcdef',b'fedcba',b'aaaabc',b'bbbabc',b'cccabc',b'dddabc',b'eeeabc',b'fffabc']
        alphabet=alphabet_of(docs)
        x,l,y=diffusion_batch(docs[:2],alphabet,5,torch.Generator().manual_seed(4))
        for i in range(2):
            rows=x[i*5:(i+1)*5]; clean=torch.tensor(list(docs[i]))
            changed=rows!=clean
            torch.testing.assert_close(1-changed.float().mean(-1),y[i*5:(i+1)*5])
            self.assertTrue((~changed[:-1]|changed[1:]).all())
            self.assertEqual(int(changed[0].sum()),0); self.assertTrue(changed[-1].all())
        a,b,c,manifest=split_corpus(docs+docs[:2],valid=2,test=2)
        self.assertFalse(set(a)&set(b) or set(a)&set(c) or set(b)&set(c))

    def test_annealing(self):
        values=[anneal(i,1001,.2,4,.1) for i in range(1001)]
        self.assertAlmostEqual(values[0],0); self.assertAlmostEqual(values[-1],.2)
        self.assertTrue(any(b<a for a,b in zip(values,values[1:])))
        self.assertTrue(all(0<=v<=.2+1e-15 for v in values))

if __name__=='__main__': unittest.main()
