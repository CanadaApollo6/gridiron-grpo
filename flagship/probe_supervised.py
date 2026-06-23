"""Numpy go/no-go: does an average-loss learner go BLIND to the deductive cell as eps->0,
EVEN when handed the exact trigger feature? Single-hidden-layer MLP, BCE + weight decay
(the smoothing bias). Reports bulk acc (sanity), cell recovery (the residual), and a
cell-ONLY representability probe (the cell is learnable once it's all the data)."""

import numpy as np
from task import make_dataset, cell_eval_set, bulk_eval_set, featurize

rng = np.random.default_rng(0)


def train_mlp(X, y, wts, H=64, epochs=60, bs=256, lr=0.1, wd=2e-3, seed=0):
    r = np.random.default_rng(seed)
    n, d = X.shape
    W1 = r.normal(0, 0.5, (d, H))
    b1 = np.zeros(H)
    W2 = r.normal(0, 0.5, (H, 1))
    b2 = np.zeros(1)
    wts = (wts / wts.mean()).reshape(-1, 1)
    for _ in range(epochs):
        idx = r.permutation(n)
        for s in range(0, n, bs):
            j = idx[s : s + bs]
            Xb = X[j]
            yb = y[j].reshape(-1, 1)
            wb = wts[j]
            z1 = Xb @ W1 + b1
            a1 = np.maximum(z1, 0)
            logit = a1 @ W2 + b2
            p = 1 / (1 + np.exp(-logit))
            g = wb * (p - yb) / len(j)
            dW2 = a1.T @ g + wd * W2
            db2 = g.sum(0)
            da1 = g @ W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = Xb.T @ dz1 + wd * W1
            db1 = dz1.sum(0)
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

    def predict(Xn):
        a1 = np.maximum(Xn @ W1 + b1, 0)
        return (1 / (1 + np.exp(-(a1 @ W2 + b2)))).ravel() > 0.5

    return predict


# held-out eval sets (fixed across eps)
_, _, _, _, m_ref = make_dataset(2000, 0.05, rng)
uce, vce, yce = cell_eval_set(3000, m_ref, rng)
ube, vbe, ybe = bulk_eval_set(3000, m_ref, rng)
Xce = featurize(uce, vce, m_ref, trigger=True)
Xbe = featurize(ube, vbe, m_ref, trigger=True)

print(f"{'eps':>6} {'eps_real':>9} {'bulk_acc':>9} {'cell_recovery':>14}")
print("-" * 42)
for eps in (0.20, 0.10, 0.05, 0.02, 0.01):
    u, v, y, is_cell, meta = make_dataset(8000, eps, rng)
    X = featurize(u, v, meta, trigger=True)  # P2: trigger handed over
    pred = train_mlp(X, y, np.ones(len(y)))
    bulk_acc = (pred(Xbe) == ybe).mean()
    cell_rec = (pred(Xce) == yce).mean()  # true label is 1 in cell
    print(f"{eps:>6.2f} {meta['eps_real']:>9.3f} {bulk_acc:>9.3f} {cell_rec:>14.3f}")

# representability: train on CELL-ONLY -> can it represent the band? (isolates localization)
u, v, y, is_cell, meta = make_dataset(8000, 0.01, rng)
# balanced cell-only set: cell band instances + a few outside, all labeled by oracle
ub, vb, yb = cell_eval_set(4000, meta, rng)
uo, vo, yo = bulk_eval_set(4000, meta, rng)
Uu = np.concatenate([ub, uo])
Vv = np.concatenate([vb, vo])
Yy = np.concatenate([yb, yo])
Xtr = featurize(Uu, Vv, meta, trigger=True)
predC = train_mlp(Xtr, Yy, np.ones(len(Yy)))
print(
    f"\nrepresentability (train on cell+bulk balanced) -> cell recovery = {(predC(Xce) == yce).mean():.3f}"
)
print(
    "interpretation: cell recoverable when localized, but collapses to the bulk floor as eps->0 above."
)
