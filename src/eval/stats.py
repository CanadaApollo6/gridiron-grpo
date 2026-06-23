"""Small, dependency-free statistics for honest eval reporting.

Three things the study needs and the field too often skips:
  * wilson(...)      -- a confidence interval on a proportion, so every accuracy
                        cell carries an error bar (at n~130/kind, deltas under
                        ~8pp are noise; show it).
  * mcnemar_exact()  -- the correct paired test for "did tuning change accuracy"
                        on the SAME eval items (base vs tuned), far more powerful
                        than treating the two runs as independent.
  * pass_at_k(...)   -- the unbiased pass@k estimator (Chen et al., 2021). Used to
                        measure base-model headroom per task BEFORE training:
                        GRPO can only reinforce what the base already samples, so
                        a task with base pass@k ~ 0 is unlearnable by GRPO.

Stdlib only (math). Importable as a library and runnable as a self-test:
    python src/eval/stats.py
"""

from __future__ import annotations

import math


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Default z=1.96 -> 95%.
    Returns (low, high), clamped to [0, 1]. Robust near 0 and 1 (unlike normal)."""
    if n <= 0:
        return (0.0, 0.0)
    phat = hits / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def wilson_str(hits: int, n: int, z: float = 1.96) -> str:
    """'52.3% [44.1, 60.4]' -- accuracy with its 95% CI, for tables."""
    lo, hi = wilson(hits, n, z)
    p = hits / n if n else 0.0
    return f"{p * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"


def mcnemar_counts(base_correct: list[bool], tuned_correct: list[bool]) -> tuple[int, int]:
    """Discordant pair counts on paired per-item correctness:
        b = base right, tuned wrong   (regressions)
        c = base wrong, tuned right   (gains)
    Concordant pairs carry no information about the change and are ignored."""
    if len(base_correct) != len(tuned_correct):
        raise ValueError("paired inputs must be the same length")
    b = sum(1 for a, t in zip(base_correct, tuned_correct, strict=False) if a and not t)
    c = sum(1 for a, t in zip(base_correct, tuned_correct, strict=False) if (not a) and t)
    return b, c


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value on discordant counts b, c.
    Exact is the right choice when discordant n is small; valid for any n."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: P(at least one correct in k draws) given c of n samples
    were correct (Chen et al., 2021, HumanEval). Numerically stable product form."""
    if n <= 0 or k <= 0 or c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


if __name__ == "__main__":
    import random

    ok = True

    # --- pass_at_k vs Monte Carlo --------------------------------------------
    random.seed(0)
    for n, c, k in [(64, 5, 8), (10, 5, 2), (20, 1, 8), (16, 16, 8), (32, 0, 4)]:
        analytic = pass_at_k(n, c, k)
        # simulate: a pool of n items with c correct; draw k without replacement
        pool = [1] * c + [0] * (n - c)
        trials = 200000
        hit = 0
        for _ in range(trials):
            hit += 1 if any(random.sample(pool, k)) else 0
        mc = hit / trials
        close = abs(analytic - mc) < 0.01
        ok &= close
        print(
            f"pass@{k} n={n} c={c}: analytic={analytic:.4f} mc={mc:.4f} {'ok' if close else 'FAIL'}"
        )

    # --- wilson against textbook value ---------------------------------------
    lo, hi = wilson(50, 100)
    w_ok = abs(lo - 0.4038) < 0.002 and abs(hi - 0.5962) < 0.002
    ok &= w_ok
    print(f"wilson(50,100)=({lo:.4f},{hi:.4f}) expect ~(0.4038,0.5962) {'ok' if w_ok else 'FAIL'}")

    # --- mcnemar exact against hand value ------------------------------------
    p = mcnemar_exact(10, 2)  # 2*(C(12,0)+C(12,1)+C(12,2))/2^12 = 2*79/4096
    m_ok = abs(p - (2 * 79 / 4096)) < 1e-9
    ok &= m_ok
    print(f"mcnemar_exact(10,2)={p:.4f} expect {2 * 79 / 4096:.4f} {'ok' if m_ok else 'FAIL'}")
    # symmetric, and (0,0)->1.0
    ok &= mcnemar_exact(2, 10) == mcnemar_exact(10, 2)
    ok &= mcnemar_exact(0, 0) == 1.0

    # --- counts helper --------------------------------------------------------
    b, c = mcnemar_counts([True, True, False, False], [True, False, True, True])
    ok &= (b, c) == (1, 2)
    print(f"mcnemar_counts -> b={b} c={c} expect b=1 c=2 {'ok' if (b, c) == (1, 2) else 'FAIL'}")

    print("ALL TESTS", "PASS" if ok else "FAIL")
    import sys

    sys.exit(0 if ok else 1)
