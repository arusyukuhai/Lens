"""Random unbounded-capture vertex checks against independent old Replace."""
import random, unittest
import torch
from torch.nn import functional as F
if __package__:
    from .model import SerialRule
else:
    from model import SerialRule
if __package__:
    from .old_reference import discrete_replace
else:
    from old_reference import discrete_replace
if __package__:
    from .test_old_wildcard import set_vertex
else:
    from test_old_wildcard import set_vertex

class TestVertex(unittest.TestCase):
 def test_random_one_hot_soft_rule(self):
  torch.set_num_threads(1)
  rng=random.Random(9481)
  rule=SerialRule(256,3,4,3,2).double()
  cap=64
  checked=0; failed=[]
  for i in range(35):
   input_=bytes(rng.choices(b'abc',k=rng.randrange(2,10)))
   n=rng.randrange(1,4)
   subset=sorted(rng.sample(range(n),rng.randrange(0,min(n,2)+1)))
   lhs=[rng.choice(b'abc') for j in range(n)]
   for j in subset:lhs[j]=-(subset.index(j)+1)
   rhs=[rng.choice(list(b'abc')+list(range(-len(subset),0))) for _ in range(rng.randrange(0,4))]
   set_vertex(rule,lhs,rhs)
   raw=torch.tensor([list(input_)+[0]*(cap-len(input_))])
   x=F.one_hot(raw,256).double()
   try:
    with torch.no_grad(): y,l,f=rule(x,torch.tensor([len(input_)],dtype=torch.double),torch.tensor([len(input_)],dtype=torch.double),hard=True)
    actual=bytes(y[0,:int(l.item()+.5)].argmax(-1).tolist())
   except (RuntimeError,OverflowError) as exc:
    failed.append((input_,lhs,rhs,str(exc)))
    continue
   expected=bytes(discrete_replace(input_,lhs,rhs))
   if actual!=expected:
    failed.append((input_,lhs,rhs,actual,expected))
   checked+=1
  print('checked',checked,'mismatch',len(failed))
  for d in failed[:8]:print('DIFF',d)
  self.assertEqual(len(failed),0)
if __name__=='__main__':unittest.main(verbosity=2)
