"""Regression tests for old wildcard search order in the C2 GPU training path.

The differentiable DP supports unrestricted capture lengths at discrete
vertices, but continuous probabilities are a mean-field relaxation.
"""
import unittest
import torch
from torch.nn import functional as F
if __package__:
    from .model import SerialRule
else:
    from model import SerialRule
if __package__:
    from .test_old_wildcard import set_vertex
else:
    from test_old_wildcard import set_vertex
if __package__:
    from .old_reference import discrete_replace
else:
    from old_reference import discrete_replace

class TestPriority(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def forward_vertex(self, rule, raw):
        buf=32
        tokens=torch.tensor([list(raw)+[0]*(buf-len(raw))])
        x=F.one_hot(tokens,256).double()
        with torch.no_grad():
            y, length, fire=rule(x,torch.tensor([len(raw)],dtype=torch.double),
                                 torch.tensor([len(raw)],dtype=torch.double),hard=True)
        return bytes(y[0,:round(float(length[0]))].argmax(-1).tolist()), float(fire[0])

    def test_first_delimiter_unbounded(self):
        r=SerialRule(256,5,3,2,2).double()
        set_vertex(r,[97,-1,98,-2,99],[81])  # a*1b*2c => Q
        for src in (b'a1b2c',b'abc',b'axbyc'):
            got,_=self.forward_vertex(r,src)
            self.assertEqual(got,bytes(discrete_replace(src,[97,-1,98,-2,99],[81])))

    def test_long_first_delimiter_is_not_discarded(self):
        r=SerialRule(256,5,3,1,2).double() # max_capture=1 ignored
        set_vertex(r,[97,-1,98,-2,99],[81])
        src=b'abzzbc'
        got,f=self.forward_vertex(r,src)
        self.assertEqual(got,b'Q')
        self.assertEqual(f,1/len(src))
        # No later boundary is taken if the literal chain has no complete match.
        src=b'abzzb'
        got,f=self.forward_vertex(r,src)
        self.assertEqual(got,src)
        self.assertEqual(f,0.)

    def test_continuous_first_delimiter_gate_gradient(self):
        r=SerialRule(3,3,2,2,1).double()
        # Delimiter-survival law is smooth even when different input bytes
        # can probabilistically create/destroy an earlier delimiter.
        logits=(torch.randn(1,5,3,dtype=torch.double)*.2).requires_grad_()
        length=torch.tensor([4.25],dtype=torch.double,requires_grad=True)
        def scalar(z,n):
            output,new_len,rate=r(z.softmax(-1),n,torch.full_like(n,5.))
            return output[:,0,0].sum()+new_len.sum()+rate.sum()
        self.assertTrue(torch.autograd.gradcheck(scalar,(logits,length),atol=8e-4,rtol=3e-3))
        self.assertTrue(torch.autograd.gradgradcheck(scalar,(logits,length),atol=3e-3,rtol=2e-2))

if __name__=='__main__': unittest.main(verbosity=2)
