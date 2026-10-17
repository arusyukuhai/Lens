import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import math
from itertools import product
import torch
from lens_gpu.exact_pushforward import (RuleSupport,RuleDistribution,ExactSerialPushforward,
                                         TrainableSerial,deterministic_text,length_distribution)
from lens_gpu.old_replace import replace_with_count
from reference_exact.discrete_replace import discrete_replace

DT=torch.float64

def onehot_supported(supports, choices, vocab=271):
    p=torch.zeros(len(supports),vocab,dtype=DT)
    for i,c in enumerate(choices):p[i,c]=1
    return p

def rho(k, n):
    t=torch.zeros(n+1,dtype=DT);t[k]=1
    return t

def R(a,b,al,bl):return RuleDistribution(a,b,al,bl)

def test_examples():
    examples=[(b'abXYZ',tuple(b'ab')+(-1,), (ord('Q'),), b'Q'),
              (b'a12b34', (97,-1,98,-1),(-2,-1), b'3412'),
              (b'a12345b', (97,-1,98),(ord('Q'),), b'Q'),
              (b'ab',(-1,),(ord('Q'),),b'ab'),
              (b'ab',tuple(b'ab'), tuple(b'ABCDE'),b'ABCDE')]
    for s,a,b,y in examples:
        assert discrete_replace(s,a,b)==tuple(y)
        assert replace_with_count(s,a,b)[0]==y
        x,r,rs=deterministic_text(s)
        pat=[[c if c>=0 else 256-c-1] for c in a]
        rep=[[c if c>=0 else 256-c-1] for c in b]
        spec=RuleSupport(pat,rep)
        engine=ExactSerialPushforward(rs,[spec],max_output=100,max_worlds=10000)
        law=engine(x,r,[R(onehot_supported(pat,[v[0] for v in pat]),rho(len(a),len(a)),onehot_supported(rep,[v[0] for v in rep]),rho(len(b),len(b)))])
        dist=dict(zip(law.outcomes,law.probabilities.tolist()))
        assert abs(dist.get(y,0)-1)<1e-10,(s,a,b,law.outcomes,dist)

def test_pushforward_and_correlation():
    # Input x can be a12b or a34b. The repeated capture must be perfectly
    # correlated: outcome 1212 / 3434, never the cross product 1234 / 3412.
    x=torch.zeros(4,256,dtype=DT)
    x[0,ord('a')]=1;x[1,ord('1')]=.5;x[1,ord('3')]=.5
    x[2,ord('2')]=.5;x[2,ord('4')]=.5;x[3,ord('b')]=1
    rs=((97,), (49,51),(50,52),(98,))
    pats=[[97],[256],[98]]
    reps=[[256],[256]]
    eng=ExactSerialPushforward(rs,[RuleSupport(pats,reps)],max_worlds=2000)
    law=eng(x,rho(4,4),[R(onehot_supported(pats,[97,256,98]),rho(3,3),onehot_supported(reps,[256,256]),rho(2,2))])
    got=dict(zip(law.outcomes,law.probabilities.tolist()))
    assert got[b'1212']==.25 and got[b'1414']==.25 and got[b'3232']==.25 and got[b'3434']==.25, got
    assert b'1234' not in got
    marg,pres,n=law.marginal_bytes()
    assert marg.shape[1]==256 and abs(float(n)-4)<1e-12


def test_serial_and_expected_head():
    # Rule1 adds X or Y (one random rule shared across entire sequence), rule2
    # recognizes X and replaces with Z; exact law is {ZZ,YY}, not mixtures.
    s=b'aa'
    x,r,rs=deterministic_text(s)
    specs=[RuleSupport([[ord('a')]],[[ord('X'),ord('Y')]]),
           RuleSupport([[ord('X')]],[[ord('Z')]])]
    eng=ExactSerialPushforward(rs,specs,track_features=True,max_worlds=1000)
    ds=[R(onehot_supported([[97]],[97]),rho(1,1),
          torch.nn.functional.one_hot(torch.tensor([88]),271).to(DT).clone().index_put((torch.tensor([0]),torch.tensor([88])),torch.tensor([.5],dtype=DT)).index_put((torch.tensor([0]),torch.tensor([89])),torch.tensor([.5],dtype=DT)),rho(1,1)),
        R(onehot_supported([[88]],[88]),rho(1,1),onehot_supported([[90]],[90]),rho(1,1))]
    law=eng(x,r,ds)
    got=dict(zip(*law.text_law()))
    assert abs(float(got[b'ZZ'])-.5)<1e-12 and abs(float(got[b'YY'])-.5)<1e-12
    class Nonlinear(torch.nn.Module):
        def forward(self,z):return (z[:,0]+2*z[:,1]).square()
    head=Nonlinear()
    expected=law.expected_head(head)
    direct=(law.probabilities*head(law.feature_tensor())).sum()
    assert torch.allclose(expected,direct)
    assert abs(float(expected-head(law.expected_features().unsqueeze(0))[0]))>1e-5


def test_c2_and_logit_gradients():
    s=b'a1b'
    x,r,rs=deterministic_text(s)
    spec=RuleSupport([[ord('a')],[256],[ord('b')]], [[ord('X'),ord('Y')]])
    model=TrainableSerial(rs,[spec],track_features=True,max_worlds=1000,dtype=DT)
    # Full rule length law, including length=0, is differentiated exactly.
    w=model.rules[0].replacement_logits[0]
    w.data=torch.tensor([.2,-.3],dtype=DT)
    def f(z):
        model.rules[0].replacement_logits[0]=torch.nn.Parameter(z)
        law=model(x,r)
        # Probability of a particular complete outcome: linear in the joint law
        return law.probabilities[law.outcomes.index(b'X')]
    value=torch.tensor([.3,-.6],dtype=DT,requires_grad=True)
    assert torch.autograd.gradcheck(f,(value,),eps=1e-6,atol=1e-5)
    assert torch.autograd.gradgradcheck(f,(value,),eps=1e-6,atol=1e-5)
    assert torch.autograd.gradcheck(lambda q: length_distribution(q,3)[2], (torch.tensor(.6,dtype=DT,requires_grad=True),), eps=1e-6)
    assert torch.autograd.gradgradcheck(lambda q: length_distribution(q,3)[2], (torch.tensor(.6,dtype=DT,requires_grad=True),), eps=1e-6)


def test_cutoff_error_not_approx():
    x,r,rs=deterministic_text(b'ab')
    s=RuleSupport([[97,98],[97,98]],[[97,98],[97,98]])
    try:ExactSerialPushforward(rs,[s],max_worlds=3)
    except ValueError as e:assert 'no approximation' in str(e)
    else:raise AssertionError('excessively large plan must fail, never silently approximate')

if __name__=='__main__':
    test_examples();print('examples: PASS')
    test_pushforward_and_correlation();print('joint correlations: PASS')
    test_serial_and_expected_head();print('serial + nonlinear head: PASS')
    test_c2_and_logit_gradients();print('autograd gradgrad: PASS')
    test_cutoff_error_not_approx();print('fail closed: PASS')
