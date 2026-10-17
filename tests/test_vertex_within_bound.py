"""Old bounded test replaced with unbounded long-capture hard-vertex tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.functional import one_hot
from lens_gpu.model import SerialRule
from lens_gpu.test_old_wildcard import set_vertex
from lens_gpu.old_reference import discrete_replace


def test_long_capture():
    torch.set_num_threads(1)
    rule=SerialRule(256,5,4,1,3).double()  # ignored compatibility max_capture=1
    cases=[
        (b'a'+b'x'*35+b'b',[97,-1,98],[-1]),
        (b'a'+b'x'*28+b'b'+b'y'*20+b'c',[97,-1,98,-2,99],[-2,-1]),
        (b'ab'+b'Z'*45,[97,98,-1],[88,-1]),
        (b'a'+b'x'*25+b'b'+b'y'*16+b'c',[97,-1,98,-2,99],[-2,-1,-2]),
    ]
    for data,lhs,rhs in cases:
        set_vertex(rule,lhs,rhs)
        limit=max(160,3*len(data))
        ids=torch.tensor([list(data)+[0]*(limit-len(data))])
        ln=torch.tensor([len(data)],dtype=torch.double)
        with torch.no_grad():
            out,sz,f=rule(one_hot(ids,256).double(),ln,ln,hard=True)
        actual=bytes(out[0,:round(float(sz[0]))].argmax(-1).tolist())
        expected=bytes(discrete_replace(data,lhs,rhs))
        assert actual==expected,(data,lhs,rhs,actual,expected)
    print('unbounded captures up to 45 bytes, reorder, duplicate: PASS')


if __name__=='__main__':test_long_capture()
