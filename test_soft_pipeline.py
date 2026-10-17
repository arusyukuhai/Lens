import unittest
import random
import torch
from torch.autograd import gradcheck,gradgradcheck
from soft_pipeline import _Solve,SoftReplaceLayer,SoftReplacePipeline,Rule
from discrete_replace import discrete_replace

torch.set_num_threads(1)
class Tests(unittest.TestCase):
    def test_scan_second_derivatives(self):
        torch.manual_seed(31)
        w=(torch.rand(1,7,3,dtype=torch.float64)*.15).requires_grad_()
        s=torch.rand(1,7,dtype=torch.float64,requires_grad=True)
        for trans in (False,True):
            f=lambda w,s:_Solve.apply(w,s,trans)
            self.assertTrue(gradcheck(f,(w,s)))
            self.assertTrue(gradgradcheck(f,(w,s)))

    def test_full_layer_derivatives(self):
        layer=SoftReplaceLayer([Rule((97,-1,98),(99,-1)),Rule((97,-1,98),(-1,100))],
                               min_capture=1,max_capture=1,dtype=torch.float64)
        # Check ALL 256 input channels at EVERY position, including length.
        z=torch.randn(1,4,256,dtype=torch.float64,requires_grad=True)
        length=torch.tensor([3.3],dtype=torch.float64,requires_grad=True)
        proj=torch.randn(1,4,256,dtype=torch.float64)
        f=lambda z,l: (layer(z.softmax(-1),l)[0]*proj).sum()+layer(z.softmax(-1),l)[1].square().sum()
        self.assertTrue(gradcheck(f,(z,length),fast_mode=True,atol=2e-5))
        self.assertTrue(gradgradcheck(f,(z,length),fast_mode=True,atol=5e-5))
        # Rule-logit gradients, including second derivatives, are checked via
        # functional_call (no .data mutation / straight-through estimator).
        def g(logits):
            y,l=torch.func.functional_call(layer,{'logits':logits},(z.softmax(-1),length))
            return (y*proj).sum()+l.square().sum()
        logits=torch.tensor([.3,-.2],dtype=torch.float64,requires_grad=True)
        self.assertTrue(gradcheck(g,(logits,),fast_mode=True))
        self.assertTrue(gradgradcheck(g,(logits,),fast_mode=True,atol=5e-5))

    def test_trainable_literal_parameters(self):
        layer=SoftReplaceLayer([Rule((97,-1,98),(99,-1))],min_capture=1,max_capture=1,
                               dtype=torch.float64,trainable_literals=True,pattern_prior=8.)
        X=torch.nn.functional.one_hot(torch.tensor([[97,120,98,100]]),256).double()*.9+.1/256
        length=torch.tensor([3.6],dtype=torch.float64)
        projection=torch.randn(1,4,256,dtype=torch.float64)
        names,params=zip(*layer.named_parameters())
        def f(*theta):
            y,l=torch.func.functional_call(layer,dict(zip(names,theta)),(X,length))
            return (y*projection).sum()+l.square().sum()
        self.assertTrue(gradcheck(f,params,fast_mode=True,atol=2e-5))
        self.assertTrue(gradgradcheck(f,params,fast_mode=True,atol=5e-5))

    def test_hard_rules(self):
        rng=random.Random(81)
        cases=[Rule((60,-1,62),(91,-1,93)),Rule((60,-1,62),(-1,-1)),
               Rule((60,-1,124,-1,62),(-2,58,-1)),Rule((97,98),(99,))]
        for rule in cases:
            layer=SoftReplaceLayer([rule],min_capture=1,max_capture=3,dtype=torch.float64)
            for _ in range(8):
                one=bytes(rng.choices(b'xyz',k=rng.randrange(1,4)))
                two=bytes(rng.choices(b'uvw',k=rng.randrange(1,4)))
                raw=b'..'+(b'<'+one+b'|'+two+b'>' if 124 in rule.pattern else b'<'+one+b'>')+b'..ab'
                X=torch.nn.functional.one_hot(torch.tensor(list(raw)),256).double()[None]
                y,l=layer(X,torch.tensor([float(len(raw))],dtype=torch.float64))
                expected=discrete_replace(raw,rule.pattern,rule.replacement,min_wildcard_len=1)
                self.assertEqual(round(float(l.detach())),len(expected))
                self.assertEqual(tuple(y[0,:len(expected)].argmax(-1).tolist()),expected)
                torch.testing.assert_close(y.sum(-1),torch.ones_like(y[:,:,0]))
                if expected:
                    torch.testing.assert_close(y[0,:len(expected)],torch.nn.functional.one_hot(torch.tensor(expected),256).double())

if __name__=='__main__':unittest.main(verbosity=2)
