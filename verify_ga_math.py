"""Independent numerical checks of the mathematical GA scoring design.
These tests DO NOT compile or execute the Nim implementation.
"""
import math
import numpy as np

rng = np.random.default_rng(421)
for n, p in [(12, 6), (16, 42), (26, 1500)]:
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    lam = 0.1
    mean = X.mean(0)
    scale = np.where(X.std(0) == 0, 1.0, X.std(0))
    Z = (X - mean) / scale
    A = np.linalg.inv(Z @ Z.T + lam * np.eye(n))
    alpha = A @ (y - y.mean())
    loo = y - lam * alpha / (lam * np.diag(A) - 1.0/n)
    brute = []
    for i in range(n):
        mask = np.arange(n) != i
        zz = Z[mask]
        zmean = zz.mean(0)
        centered = zz - zmean
        yt = y[mask]
        beta = centered.T @ np.linalg.solve(
            centered @ centered.T + lam*np.eye(n-1), yt-yt.mean()
        )
        brute.append(yt.mean() + (Z[i]-zmean) @ beta)
    max_err = float(np.max(np.abs(loo - brute)))
    print(f"n={n}, p={p}, LOO max err={max_err:.3g}")
    assert max_err < 1e-8

samples = np.floor(np.exp(rng.random(100_000)*math.log(1500))).astype(int)
assert samples.min() >= 1 and samples.max() <= 1500
assert abs(np.median(samples) - math.sqrt(1500)) < 4
print("Log-uniform mutation distribution check passed")
