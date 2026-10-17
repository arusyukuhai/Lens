"""Differential comparison to the corrected Nim implementation."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import tempfile
from replace_exact import discrete_replace

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--nim',default='nim')
    args=parser.parse_args()
    rng=random.Random(1741)
    cases=[]
    for _ in range(5000):
        x=[rng.choice([0,1,2]) for _ in range(rng.randrange(13))]
        a=[rng.choice([0,1,2,-1,-15]) for _ in range(rng.randrange(9))]
        b=[rng.choice([0,1,2,-1,-2,-15]) for _ in range(rng.randrange(7))]
        cases.append(dict(x=x,a=a,b=b,minimum=rng.randrange(3)))
    cases.append(dict(x=[0,0,2,2,2,0,2,0],a=[0,-15,2],b=[1,2],minimum=1))
    for k in (15,16,17):
        cases.append(dict(x=[0,1,2],a=[0]+[-1]*k,b=[-15,-1],minimum=1))
    root=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        executable=str(Path(tmp)/'oracle')
        subprocess.run([args.nim,'c','--hints:off','--warnings:off',
                        '--nimcache:'+str(Path(tmp)/'cache'),'-o:'+executable,
                        str(root/'oracle.nim')],check=True,capture_output=True,text=True)
        result=subprocess.run([executable],input=''.join(json.dumps(c)+'\n' for c in cases),
                              capture_output=True,text=True,check=True)
    lines=result.stdout.splitlines()
    assert len(lines)==len(cases)
    for c,line in zip(cases,lines):
        got=discrete_replace(c['x'],c['a'],c['b'],min_wildcard_len=c['minimum'])
        expected=tuple(json.loads(line))
        assert got==expected,(c,got,expected)
    print(f'PASS: {len(cases)} cases match corrected Nim output (minimum 0/1/2).')
if __name__=='__main__':
    main()
