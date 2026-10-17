import itertools
import unittest
import torch
from replace_exact import (ExactReplace, discrete_replace, length_distribution,
                           supported_probabilities, LearnableSequence, outcome_nll, normalized_entropy)

torch.set_num_threads(1)
torch.manual_seed(11)
D = torch.float64

def scalar(x):
    return torch.tensor(float(x), dtype=D)

def onehot(seq, alphabet):
    if not seq:
        return torch.empty((0, alphabet), dtype=D)
    return torch.nn.functional.one_hot(torch.tensor(seq), alphabet).to(D)

class Tests(unittest.TestCase):
    def test_discrete_semantics(self):
        cases = [
            ([0,0,2,2,2,0,2,0], [0,-15,2], [1,2], [1,2,2,2,0,2,0], 1),
            (b'ababa', [97,98], [120], b'xxa', 0),
            (b'axxbayybc', [97,-1,98], [-1], b'xxyyc', 0),
            (b'ab', [97,-1,-1,98], [-1,-2], b'', 0),
            (b'abc', [97,-1], [-1,-1], b'bcbc', 0),
            (b'ab', [-1], [120], b'ab', 0),
            (b'ab', [], [120], b'ab', 0),
            (b'ab', [97,98], [-1], b'', 0),
            (b'acb', [97,-1,98], [-15], b'c', 0),
            (b'ab', [97,-1,98], [120], b'ab', 1),
            (b'ab', [97,-1,-1], [-1,-2], b'b', 1),
        ]
        for x,a,b,y,m in cases:
            self.assertEqual(discrete_replace(x,a,b,min_wildcard_len=m), tuple(y))
        self.assertEqual(discrete_replace(b'ab', [97,-1], [-1]*4, max_output=2), tuple(b'ab'))

    def test_literal_exhaustive(self):
        strings = [bytes(p) for k in range(5) for p in itertools.product((97,98), repeat=k)]
        for x in strings:
            for a in strings[1:7]:
                for b in (b'', b'a', b'bb'):
                    self.assertEqual(discrete_replace(x,a,b), tuple(x.replace(a,b)))

    def test_length_cardinal_and_derivatives(self):
        for k in range(5):
            t = scalar(k).requires_grad_()
            rho = length_distribution(t,4)
            torch.testing.assert_close(rho, torch.nn.functional.one_hot(torch.tensor(k),5).to(D))
            for i in range(5):
                g = torch.autograd.grad(rho[i], t, create_graph=True, retain_graph=True)[0]
                h = torch.autograd.grad(g, t, retain_graph=True)[0]
                self.assertEqual(float(g.detach()),0.)
                self.assertEqual(float(h.detach()),0.)
        for t in torch.linspace(0,4,101,dtype=D):
            rho=length_distribution(t,4)
            self.assertTrue(bool((rho >= -1e-14).all()))
            self.assertAlmostEqual(float(rho.sum()),1.)

    def test_vertices_and_copy_correlation(self):
        xs=[[97],[99,100],[98]]
        aa=[[97],[256],[98]]
        bb=[[256],[256]]
        plan=ExactReplace(xs,aa,bb,n=1)
        for c in (99,100):
            r=plan(onehot([97,c,98],256),scalar(3),onehot([97,256,98],257),scalar(3),
                   onehot([256,256],257),scalar(2))
            self.assertEqual(float(r.length),2.)
            torch.testing.assert_close(r.probabilities[:2],onehot([c,c],256))
            torch.testing.assert_close(r.probabilities.sum(-1),torch.ones(plan.output_capacity,dtype=D))
        X=onehot([97,99,98],256)
        X[1,99]=.5; X[1,100]=.5
        r=plan(X,scalar(3),onehot([97,256,98],257),scalar(3),onehot([256,256],257),scalar(2))
        positive={y:float(q) for y,q in zip(r.outcomes,r.outcome_weights) if q>0}
        self.assertEqual(positive,{(99,99):.5,(100,100):.5})

    def test_grad_and_gradgrad_both_apis(self):
        xs=[[97,98],[97,98]]
        aa=[[97,256],[98,256]]
        bb=[[99,256]]
        plan=ExactReplace(xs,aa,bb,n=1)
        logits=tuple((torch.randn(shape,dtype=D)*.2).requires_grad_()
                     for shape in ((2,2),(2,2),(1,2)))
        ls=tuple(scalar(v).requires_grad_() for v in (1.32,1.43,.61))
        projection=torch.randn(plan.output_capacity,256,dtype=D)
        def f(zx,za,zb,lx,la,lb):
            X=supported_probabilities(zx,xs,256)
            A=supported_probabilities(za,aa,257)
            B=supported_probabilities(zb,bb,257)
            r=plan(X,lx,A,la,B,lb)
            return (r.probabilities*projection).sum()+r.length.square()+r.outcome_weights.square().sum()
        self.assertTrue(torch.autograd.gradcheck(f,logits+ls))
        self.assertTrue(torch.autograd.gradgradcheck(f,logits+ls))
        length_logits=tuple(torch.randn(k,dtype=D,requires_grad=True) for k in (3,3,2))
        def g(zx,za,zb,rx,ra,rb):
            r=plan.forward_distributions(supported_probabilities(zx,xs,256),rx.softmax(-1),
                   supported_probabilities(za,aa,257),ra.softmax(-1),
                   supported_probabilities(zb,bb,257),rb.softmax(-1))
            return (r.probabilities*projection).sum()+r.length.square()+outcome_nll(r,())
        self.assertTrue(torch.autograd.gradcheck(g,logits+length_logits))
        self.assertTrue(torch.autograd.gradgradcheck(g,logits+length_logits))

    def test_entropy_second_derivatives(self):
        z = torch.tensor([.2,-.4,.1],dtype=D,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(normalized_entropy,(z,)))
        self.assertTrue(torch.autograd.gradgradcheck(normalized_entropy,(z,)))
        self.assertGreater(float(normalized_entropy(torch.zeros(3,dtype=D))),
                           float(normalized_entropy(torch.tensor([10.,0.,0.],dtype=D))))

    def test_zero_probability_derivative_not_pruned(self):
        plan=ExactReplace([[97,98]],[],[],n=1)
        p=scalar(0).requires_grad_()
        X=torch.stack((1-p,p)).reshape(1,2)
        X=torch.zeros(1,256,dtype=D).scatter(1,torch.tensor([[97,98]]),X)
        empty=torch.empty(0,257,dtype=D)
        r=plan(X,scalar(1),empty,scalar(0),empty,scalar(0))
        g=torch.autograd.grad(r.probabilities[0,98],p)[0]
        self.assertEqual(float(g),1.)

    def test_guards_and_empty(self):
        with self.assertRaises(ValueError):
            ExactReplace([list(range(256))]*3,[[97]],[[98]],n=1)
        plan=ExactReplace([],[],[],n=1)
        r=plan(torch.empty(0,256,dtype=D),scalar(0),torch.empty(0,257,dtype=D),scalar(0),
               torch.empty(0,257,dtype=D),scalar(0))
        self.assertEqual(r.probabilities.shape,(0,256))
        self.assertEqual(float(r.outcome_weights.sum()),1.)
        learn=LearnableSequence([],257)
        self.assertEqual(learn()[0].shape,(0,257))

if __name__=='__main__':
    unittest.main(verbosity=2)
