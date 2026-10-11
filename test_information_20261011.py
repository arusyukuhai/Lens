"""Regression tests: OLS attribution, phi independence, and bridge protection."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from lens_rule_information import analyze, export, replacement_ids
from main import Genome, Rule


class InformationTests(unittest.TestCase):
    def test_three_independent_predictors_all_necessary(self):
        x=np.array([[0,1,0],[1,1,0],[0,1,1],[1,1,1],
                    [0,0,0],[1,0,0],[0,0,1],[1,0,1]],dtype=np.float64)
        # Reorder to obtain independent columns (first column binary pattern,
        # second group block, third nested alternation).
        x=np.array([[0,0,0],[1,0,0],[0,0,1],[1,0,1],
                    [0,1,0],[1,1,0],[0,1,1],[1,1,1]],dtype=float)
        y=.05 + .12*x[:,0]+.17*x[:,1]+.23*x[:,2]
        g=Genome([Rule([i+1],[i+1]) for i in range(3)])
        g.readout_weights=[.12,.17,.23]
        rep=analyze(g,x,np.zeros(8),y,np.zeros(8,dtype=int),max_pairs=3)
        self.assertEqual(rep.phi.shape,(3,3))
        self.assertLess(np.max(abs(rep.phi-np.eye(3))),1e-10)
        self.assertTrue(all(rep.importance>1e-5),rep.importance)
        self.assertAlmostEqual(rep.train_mse,0,places=9)

    def test_collinear_redundancy_not_incorrectly_penalized_by_phi(self):
        x=np.array([[0,0],[1,1],[0,0],[1,1],[0,0],[1,1]],float)
        g=Genome([Rule([1],[1]),Rule([2],[2])]);g.readout_weights=[.5,.5]
        report=analyze(g,x,np.zeros(6),x[:,0],np.zeros(6,dtype=int))
        self.assertAlmostEqual(report.phi[0,1],1.0,places=9)
        self.assertTrue(np.allclose(report.importance,0,atol=1e-10))

    def test_producer_remains_protected_even_with_zero_readout(self):
        x=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1],[0,0]],float)
        g=Genome([Rule([11],[88]),Rule([88],[77])]); g.readout_weights=[0,1]
        y=x[:,1]
        r=analyze(g,x,np.zeros(len(x)),y,np.zeros(len(x),dtype=int))
        self.assertEqual(r.importance[0],0)
        self.assertGreater(r.bridge_cost[0],0)
        self.assertEqual(replacement_ids(r,1),[0] if r.replacement_cost[0]<r.replacement_cost[1] else [1])

    def test_training_does_not_read_last_source(self):
        x=np.array([[0,0],[1,0],[0,1],[1,1],
                    [0,0],[1,0],[0,1],[1,1]],float)
        g=Genome([Rule([1],[1]),Rule([2],[2])]);g.readout_weights=[1,1]
        groups=np.array([0]*4+[1]*4)
        y=np.array([0,1,1,2,100,0,100,0],float)
        rep=analyze(g,x,np.zeros(8),y,groups)
        ref=analyze(g,x[:4],np.zeros(4),y[:4],np.zeros(4,dtype=int))
        self.assertTrue(np.allclose(rep.importance,ref.importance))
        with tempfile.TemporaryDirectory() as td:
            out=export(rep,str(Path(td)/'info'),0,save_plot=False)
            self.assertTrue(out[0].exists());self.assertTrue(out[1].exists())

if __name__ == '__main__': unittest.main()
