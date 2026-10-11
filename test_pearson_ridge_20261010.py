"""Regression tests for ordinary centered Ridge + per-case Pearson objective."""
import math
import numpy as np
import main as m


def test_pearson_is_not_rank_correlation():
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([1., 4., 9., 16., 25.])
    assert math.isclose(m.spearman(x,y), 1.0, abs_tol=1e-12)
    assert 0.90 < m.pearson(x,y) < 1.0
    assert m.pearson([1,1,1], [1,2,3]) == 0.0
    assert m.pearson([1,2,3], [3,2,1]) == -1.0


def test_full_ridge_matches_augmented_unpenalized_intercept():
    rng = np.random.default_rng(23)
    for n,p in [(32, 10),(10, 32),(22, 22)]:
        x = rng.standard_normal((n,p))
        x[:,1] = 1.0  # constant feature: intercept must absorb it
        yy = rng.standard_normal(n) + 2.3
        lam = 1.7
        w,bias = m.fit_standard_ridge(x, yy, lam)
        xa=np.column_stack((np.ones(n), x))
        penalty=np.eye(p+1)*lam; penalty[0,0]=0.0
        sol=np.linalg.solve(xa.T @ xa+penalty, xa.T@yy)
        assert np.allclose(x @ w + bias, xa @ sol, atol=1e-9, rtol=1e-9)
        assert abs(bias) > 0.1


def test_ridge_all_uses_every_column_and_correct_bias():
    n,p=34,320
    x=np.zeros((n,p),dtype=np.float32)
    x[:,319]=np.linspace(-1.,1.,n)
    y=0.3 + x[:,319]*0.7
    # Set varying baseline so ridge fits the proper residual target.
    base=np.linspace(-0.12,0.17,n)
    g=m.Genome([m.Rule([i%256],[i%256]) for i in range(p)])
    cids=np.repeat([0,1],n//2)
    sids=np.tile(np.arange(n//2),2)
    fitness=m.score_genome_features(g,x,base,y,cids,sids,0.001)
    assert math.isfinite(fitness)
    assert len(g.readout_weights) == p
    assert abs(g.readout_weights[-1]) > 0.01  # no max-256 feature restriction
    assert np.isfinite(g.readout_bias)
    w_ref,b_ref=m.fit_standard_ridge(x,np.asarray(y)-base,0.001)
    assert np.allclose(g.readout_weights,w_ref)
    assert np.isclose(g.readout_bias,b_ref)
    scores=base + x@w_ref + b_ref
    assert np.isclose(fitness,np.mean([m.pearson(scores[cids==i],y[cids==i]) for i in (0,1)]))


def test_ridge_bias_checkpoint_roundtrip_and_old_fallback():
    g=m.Genome([m.Rule([1],[2])],readout_weights=[0.4],readout_bias=2.5,fitness=0.3)
    h=m.clone_genome_shallow(g)
    assert h.readout_bias == 2.5
    arr={}
    m._pack_genome_group([g], 'p_',arr)
    assert np.array_equal(arr['p_readout_bias'],[2.5])
    class FakeNpZ:
        def __init__(self, arr): self.arr=arr; self.files=list(arr)
        def __getitem__(self,key): return self.arr[key]
    out=m._unpack_genome_group(FakeNpZ(arr),'p_')
    assert out[0].readout_bias == 2.5
    del arr['p_readout_bias']
    old=m._unpack_genome_group(FakeNpZ(arr),'p_')
    assert old[0].readout_bias == 0.0


def test_zero_features_ordinary_ridge():
    x=np.zeros((6,13),dtype=np.float32)
    y=np.array([0.,1.,2.,3.,4.,5.])
    w,b=m.fit_standard_ridge(x,y,1.0)
    assert np.all(w==0) and b==2.5
