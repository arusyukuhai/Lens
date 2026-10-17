"""Smoke + old deterministic interpretation/serial order regression."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lens_gpu.model import Lens,Config,discrete_features
from lens_gpu.old_replace import serial_features


class TestSerialEndToEnd(unittest.TestCase):
    def test_old_wildcards_and_true_serial(self):
        rules=[([-1,ord('b'),-2],[-2,-1]),([ord('3'),ord('4')],[ord('Z')])]
        features,result=serial_features(b'12b34',[(tuple(a),tuple(b)) for a,b in rules])
        self.assertEqual(result,b'Z12')
        self.assertEqual(len(features),2)
        self.assertAlmostEqual(features[0],1/5)
        self.assertAlmostEqual(features[1],1/5)
        feats,out=discrete_features(b'12b34',{'rules':[{'pattern':a,'replacement':b} for a,b in rules]})
        self.assertEqual(out,list(result))
        self.assertEqual(feats,list(features))

    def test_gpu_model_is_end_to_end_trainable(self):
        torch.set_num_threads(1)
        model=Lens(Config(rules=2,hidden=8,max_pattern=2,max_replacement=2,max_stars=1,checkpoint_every=1))
        x=torch.tensor([[97,98,97,99,0,0],[98,97,97,98,0,0]],dtype=torch.long)
        lengths=torch.tensor([4.,4.])
        y=model(x,lengths)
        self.assertEqual(y.shape,(2,))
        torch.nn.functional.mse_loss(y,torch.tensor([0.,1.])).backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.rules.parameters()))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters()))
        v=model(x,lengths,hard=True)
        self.assertEqual(tuple(v.shape),(2,))

    def test_both_train_entrypoints(self):
        child_env=os.environ.copy()
        child_env['OMP_NUM_THREADS']='1'
        child_env['MKL_NUM_THREADS']='1'
        child_env['OPENBLAS_NUM_THREADS']='1'
        with tempfile.TemporaryDirectory() as tmp:
            exact=Path(tmp)/'exact'
            proc=subprocess.run([sys.executable,'train.py','--mode','exact-small',
                '--demo','--device','cpu','--steps','1','--hidden','8','--out',str(exact)],
                cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=35,env=child_env)
            self.assertEqual(proc.returncode,0,proc.stdout[-3000:])
            self.assertTrue((exact/'last.pt').is_file())
            fast=Path(tmp)/'fast'
            proc=subprocess.run([sys.executable,'train.py','--mode','scalable',
                '--synthetic','--device','cpu','--rules','2','--hidden','8',
                '--max-pattern','2','--max-replacement','2','--max-stars','1',
                '--chunk-len','12','--batch-chunks','1','--levels','2','--capacity-factor','2',
                '--steps','1','--valid','2','--test','2','--checkpoint-every','1',
                '--eval-every','1','--threads','1','--out',str(fast)],cwd=ROOT,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60,env=child_env)
            self.assertEqual(proc.returncode,0,proc.stdout[-3000:])
            self.assertTrue((fast/'last.pt').is_file())
            self.assertTrue((fast/'rules.json').is_file())

if __name__=='__main__':unittest.main()
