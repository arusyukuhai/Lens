"""Optional independent NumPy verification of the one-column Ridge Pearson identity."""
import numpy as np

gen = np.random.default_rng(77)
for trial in range(300):
    n = int(gen.integers(10, 100))
    p = int(gen.integers(1, min(16, n)))
    x = gen.normal(size=(n, p))
    z = gen.normal(size=n)
    y = gen.normal(size=n)
    lam = 10 ** gen.uniform(-3, 1)
    if trial % 9 == 0:
        x[:, 0] = 5
    means = x.mean(0)
    scales = np.where(x.std(0) < 1e-12, 1.0, x.std(0))
    u = (x - means) / scales
    zs = (z - z.mean()) / z.std()
    yc = y - y.mean()
    gram = u.T @ u
    inv = np.linalg.inv(gram + n * lam * np.eye(p))
    beta = inv @ u.T @ yc
    prediction = u @ beta
    residual = yc - prediction
    cross = u.T @ zs
    h = inv @ cross
    g = zs @ residual
    alpha = g / (zs @ zs + n * lam - cross @ h)
    covariance = prediction @ yc + alpha * g
    norm = (prediction @ prediction
            + 2 * alpha * n * lam * (beta @ h)
            + alpha**2 * (zs @ zs - cross @ h - n * lam * (h @ h)))
    actual = covariance / np.sqrt(norm * (yc @ yc))
    u2 = np.column_stack([u, zs])
    beta2 = np.linalg.solve(u2.T @ u2 + n * lam * np.eye(p+1), u2.T @ yc)
    pred2 = u2 @ beta2
    expected = (pred2 @ yc) / np.sqrt((pred2 @ pred2) * (yc @ yc))
    assert abs(expected - actual) < 1e-9, (trial, expected, actual)
print('PASS: 300 full-Ridge vs incremental Pearson comparisons')
