"""Continuous identity-capture path agrees with previous DP reference."""
import torch
from lens_gpu.unbounded_dp import SerialRule
import importlib.util
from pathlib import Path

reference=Path(__file__).resolve().parents[1] / 'reference_exact/unbounded_dp_v7_reference.py'
spec=importlib.util.spec_from_file_location('legacy_dp', reference)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)


def test_equivalence():
    torch.set_num_threads(1)
    old=legacy.SerialRule(3,2,2,max_stars=1).double()
    new=SerialRule(3,2,2,max_stars=1).double()
    with torch.no_grad():
        for name,param in old.named_parameters():
            if name=='replacement_tokens':
                # 256 used in Lens; here alphabet=3, remap identity references.
                tgt=new.replacement_tokens
                tgt[:,:3].copy_(param[:,:3]); tgt[:,3].copy_(param[:,3])
                tgt[:,4:].fill_(-80)
            else:
                getattr(new,name).copy_(param)
    a=torch.randn(1,5,3,dtype=torch.double,requires_grad=True)
    length=torch.tensor([4.2],dtype=torch.double,requires_grad=True)
    old_y=old(a.softmax(-1),length,torch.tensor([5.],dtype=torch.double))
    new_y=new(a.softmax(-1),length,torch.tensor([5.],dtype=torch.double))
    for x,y in zip(old_y,new_y):
        torch.testing.assert_close(x,y,atol=5e-9,rtol=5e-8)
    loss1=old_y[0].square().sum()+old_y[2].sum()
    loss2=new_y[0].square().sum()+new_y[2].sum()
    gradold=torch.autograd.grad(loss1,(a,length))
    gradnew=torch.autograd.grad(loss2,(a,length))
    for x,y in zip(gradold,gradnew):torch.testing.assert_close(x,y,atol=5e-8,rtol=5e-7)

if __name__=='__main__':test_equivalence();print('identity reference parity: PASS')
