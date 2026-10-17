"""Independent integer reference and C2 capture-transform regression tests."""
import random
import torch
from torch.nn import functional as F
from lens_gpu.old_replace import replace_with_count
from lens_gpu.model import SerialRule, Config, Lens
from lens_gpu.test_old_wildcard import set_vertex
from lens_gpu.unbounded_dp import dense_select, stream_select


def rhs(op, capture=0):
    return -(1 + 16 * op + capture)


def text_of(rule, text, cap=80):
    tokens = torch.tensor([list(text)+[0]*(cap-len(text))], dtype=torch.long)
    with torch.no_grad():
        y, length, fire = rule(F.one_hot(tokens, 256).double(),
                               torch.tensor([len(text)], dtype=torch.double),
                               torch.tensor([len(text)], dtype=torch.double), hard=True)
    return bytes(y[0,:round(length.item())].argmax(-1).tolist()), fire.item()


def test_integer_ops():
    for text, lhs, opcode, target in [
        (b'aDBACb', (97,-1,98), (rhs(0),), b'DBAC'),
        (b'aDBACb', (97,-1,98), (rhs(1),), b'CABD'),
        (b'aDBACb', (97,-1,98), (rhs(2),), b'ABCD'),
        (b'aDBACb', (97,-1,98), (rhs(3),), b'DCBA'),
        (b'x4321z', (120,-1,122), (rhs(2),rhs(1)), b'12341234'),
        (b'QabcdREFGH!', (81,-1,82,-2,33), (rhs(1,1),rhs(2,0)), b'HGFEabcd'),
    ]:
        got,_=replace_with_count(text,lhs,opcode)
        assert got==target,(text,lhs,opcode,got,target)


def test_soft_vertices():
    rule=SerialRule(256,5,4,max_stars=2).double()
    for text,lhs,opcode in [
        (b'aDBACb', [97,-1,98], [rhs(0)]),
        (b'aDBACb', [97,-1,98], [rhs(1)]),
        (b'aDBACb', [97,-1,98], [rhs(2)]),
        (b'aDBACb', [97,-1,98], [rhs(3)]),
        (b'aABCDb', [97,-1,98], [rhs(2),rhs(1)]),
        (b'QabcdREFGH!', [81,-1,82,-2,33], [rhs(1,1),rhs(2,0)]),
        (b'ABC123456789a', [65,66,67,-1], [rhs(3)]),
    ]:
        set_vertex(rule,lhs,opcode)
        expected,_=replace_with_count(text,tuple(lhs),tuple(opcode))
        actual,fires=text_of(rule,text)
        assert actual==expected,(text,lhs,opcode,actual,expected)
        assert fires>=0


def test_2nd_derivatives():
    torch.manual_seed(234)
    r=SerialRule(3,2,2,max_stars=1).double()
    a=torch.randn(1,4,3,dtype=torch.double,requires_grad=True)
    length=torch.tensor([2.75],dtype=torch.double,requires_grad=True)
    def fn(a,length):
        x=a.softmax(-1)
        return r(x,length,length.detach()*0+4.)
    assert torch.autograd.gradcheck(fn,(a,length),atol=1e-3,rtol=5e-3)
    assert torch.autograd.gradgradcheck(fn,(a,length),atol=5e-3,rtol=3e-2)


def test_streaming_dp():
    torch.manual_seed(987)
    p=torch.rand(2,20,3,dtype=torch.double).softmax(-1)*.5
    e=torch.arange(20,dtype=torch.double)[None,:,None]+torch.rand(2,20,3,dtype=torch.double)*5+1
    assert torch.allclose(dense_select(p,e),stream_select(p,e),rtol=1e-12,atol=1e-12)
    assert torch.autograd.gradgradcheck(lambda a:stream_select(a,e),(p.requires_grad_(True),),atol=5e-4)


def test_export():
    model=Lens(Config(rules=1, hidden=8, max_pattern=3, max_replacement=2,max_stars=1, checkpoint_every=0)).double()
    set_vertex(model.rules[0],[97,-1,98],[rhs(3)])
    rule=model.export_rules()['rules'][0]
    assert rule['pattern']==[97,-1,98]
    assert rule['replacement']==[rhs(3)]
    feat,old=model.exact_features(torch.tensor([[97,ord('A'),ord('B'),98,0]]),torch.tensor([4]),return_lengths=True)
    assert int(old.item())==2
    assert abs(feat.item()-.25)<1e-9


def test_exact_law():
    from lens_gpu.exact_pushforward import ExactSerialPushforward, RuleSupport, RuleDistribution
    # RHS 256+1 is reverse of the first capture for n=1.
    engine=ExactSerialPushforward([(97,),(66,),(65,),(98,)],
        [RuleSupport(pattern=[(97,), (256,), (98,)],replacement=[(257,)])],
        n=1,capture_transforms=True,max_states=1000)
    X=F.one_hot(torch.tensor([97,66,65,98]),256).double()
    rho=X.new_zeros(5);rho[-1]=1
    P=F.one_hot(torch.tensor([97,256,98]),257).double()
    RP=F.one_hot(torch.tensor([257]),260).double()
    pa=torch.tensor([0.,0.,0.,1.],dtype=torch.double)
    pr=torch.tensor([0.,1.],dtype=torch.double)
    law=engine(X,rho,[RuleDistribution(P,pa,RP,pr)])
    assert law.outcomes[law.probabilities.argmax().item()]==b'AB'


if __name__=='__main__':
    torch.set_num_threads(1)
    for test in (test_integer_ops,test_soft_vertices,test_2nd_derivatives,test_streaming_dp,test_export,test_exact_law):
        test()
        print(test.__name__+': PASS',flush=True)
