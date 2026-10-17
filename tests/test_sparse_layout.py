"""First-hit associative DP compared with direct differentiable recurrence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from lens_gpu.unbounded_dp import first_hit_dp


def reference(hit):
    b,l=hit.shape
    prob=hit.new_zeros(b)
    moment=hit.new_zeros(b)
    pp=[];mm=[]
    for i in range(l-1,-1,-1):
        moment=hit[:,i]*i+(1-hit[:,i])*moment
        prob=hit[:,i]+(1-hit[:,i])*prob
        pp.append(prob[:,None]);mm.append(moment[:,None])
    z=hit.new_zeros(b,1)
    return torch.cat(pp[::-1]+[z],1),torch.cat(mm[::-1]+[z],1)


def test_first_hit_derivatives():
    torch.manual_seed(71)
    h=torch.sigmoid(torch.randn(2,23,dtype=torch.double,requires_grad=True))
    a,b=first_hit_dp(h)
    x,y=reference(h)
    torch.testing.assert_close(a,x,atol=1e-13,rtol=1e-13)
    torch.testing.assert_close(b,y,atol=1e-12,rtol=1e-12)
    obj=(a.square()+b.square()).sum();gx=torch.autograd.grad(obj,h,create_graph=True)[0]
    second=torch.autograd.grad(gx.sum(),h)[0]
    assert second.isfinite().all()
    print('first-hit forward/second derivatives: PASS')


if __name__=='__main__':test_first_hit_derivatives()
