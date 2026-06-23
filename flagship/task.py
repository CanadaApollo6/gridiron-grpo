"""Synthetic 'sharp rare deductive cell' task — shared by the numpy probe and the
torch (SFT+GRPO) harness. Torch-free so it runs anywhere.

Inputs u,v ~ Uniform{0..K-1}. Bulk (smooth/common) label y_bulk = [u >= v] — an easy
linear separator any flexible learner generalizes. A thin, sharp DEDUCTIVE CELL is a band
on v of width w centered at v0=round(vrel*K); inside it the label is OVERRIDDEN to 1.
Cell rarity eps = w/K (the one knob). The trigger the rule thresholds on is v (and, if
asked, the explicit distance |v-v0|). The oracle applies the rule exactly.

Recovery is measured on held-out cell instances: a bulk-following learner scores ~P(u>=v0)
there; the oracle scores 1; the gap is the residual the thesis is about.
"""

import numpy as np


def make_dataset(n, eps, rng, K=100, vrel=0.7):
    u = rng.integers(0, K, n)
    v = rng.integers(0, K, n)
    v0 = int(round(vrel * K))
    w = max(1, int(round(eps * K)))
    lo = v0 - w // 2
    hi = lo + w
    is_cell = (v >= lo) & (v < hi)
    y_bulk = (u >= v).astype(np.int64)
    y = np.where(is_cell, 1, y_bulk).astype(np.int64)
    meta = dict(K=K, v0=v0, w=w, lo=lo, hi=hi, eps_real=float(is_cell.mean()))
    return u, v, y, is_cell, meta


def cell_eval_set(n, meta, rng):
    """Held-out instances strictly inside the cell band (true label = 1)."""
    K, lo, hi = meta["K"], meta["lo"], meta["hi"]
    u = rng.integers(0, K, n)
    v = rng.integers(lo, hi, n)
    y = np.ones(n, dtype=np.int64)  # override is deterministic
    return u, v, y


def bulk_eval_set(n, meta, rng):
    """Held-out instances OUTSIDE the band; label = the bulk rule."""
    K, lo, hi = meta["K"], meta["lo"], meta["hi"]
    u, v = [], []
    while len(u) < n:
        uu = rng.integers(0, K)
        vv = rng.integers(0, K)
        if lo <= vv < hi:
            continue
        u.append(uu)
        v.append(vv)
    u = np.array(u)
    v = np.array(v)
    y = (u >= v).astype(np.int64)
    return u, v, y


def featurize(u, v, meta, trigger=False):
    K, v0 = meta["K"], meta["v0"]
    cols = [u / K, v / K]
    if trigger:  # hand the rule's exact threshold variable
        cols.append(np.abs(v - v0) / K)
    return np.stack(cols, axis=1).astype(np.float64)


def ladder_weights(v, meta, arm, region_half=12, hi_w=30.0):
    """Per-sample weights for the recovery ladder (SFT loss-weights / sampling probs)."""
    w = np.ones_like(v, dtype=np.float64)
    if arm == "region":  # upweight a coarse band (no rule)
        v0 = meta["v0"]
        w[np.abs(v - v0) <= region_half] = hi_w
    elif arm == "cell":  # upweight exactly the deductive cell (uses label)
        in_cell = (v >= meta["lo"]) & (v < meta["hi"])
        w[in_cell] = hi_w
    return w


# === Multi-value mod-arithmetic instantiation (for the torch SFT+GRPO harness) =========
# Large answer space (M classes) so that, after a bulk warm-start, the override answer is
# rarely SAMPLED on cell prompts -> GRPO gets ~0 reward there (exploration-blindness),
# distinct from SFT's measure/smoothing-blindness. a,b ~ U{0..M-1}; trigger t=(a-b)%M;
# cell = thin band on t; bulk=(a+b)%M; override=(a*b)%M.
def make_modarith(n, eps, rng, M=100):
    a = rng.integers(0, M, n)
    b = rng.integers(0, M, n)
    t = (a - b) % M
    w = max(1, int(round(eps * M)))
    t0 = M // 2
    lo = t0 - w // 2
    hi = lo + w
    is_cell = (t >= lo) & (t < hi)
    y = np.where(is_cell, (a * b) % M, (a + b) % M).astype(np.int64)
    meta = dict(M=M, t0=t0, w=w, lo=lo, hi=hi, eps_real=float(is_cell.mean()))
    return a, b, y, is_cell, meta


def modarith_eval(n, meta, rng, where):
    """where in {'cell','bulk','bleed'}: held-out a,b with their oracle answers."""
    M, lo, hi, t0 = meta["M"], meta["lo"], meta["hi"], meta["t0"]
    A, B = [], []
    while len(A) < n:
        a = int(rng.integers(0, M))
        b = int(rng.integers(0, M))
        t = (a - b) % M
        inb = lo <= t < hi
        near = (not inb) and (abs(((t - t0 + M // 2) % M) - M // 2) <= max(2, hi - lo))
        if (
            (where == "cell" and inb)
            or (where == "bulk" and not inb)
            or (where == "bleed" and near)
        ):
            A.append(a)
            B.append(b)
    A = np.array(A)
    B = np.array(B)
    t = (A - B) % M
    y = np.where((t >= lo) & (t < hi), (A * B) % M, (A + B) % M).astype(np.int64)
    return A, B, y
