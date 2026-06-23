"""Paired base-vs-tuned comparison with the statistics a per-task claim needs.

Consumes two results JSONs written by evaluate.py (each carries a per-item
`items` list), and produces:
  * overall and per-kind Δaccuracy with Wilson 95% CIs on each side,
  * a paired McNemar exact p-value (same eval items, base vs tuned) overall and
    per kind -- the correct test for "did tuning change accuracy",
  * the best-constant floor alongside, so a delta is read against the naive
    baseline, not against zero,
  * a by-depth rollup for the Q3 taxonomy.

  python src/eval/compare.py results/baseline.json results/grpo.json
Writes results/comparison.md and prints it.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.stats import mcnemar_counts, mcnemar_exact, wilson  # noqa: E402


def _acc_ci(items):
    n = len(items)
    h = sum(1 for it in items if it["correct"])
    lo, hi = wilson(h, n) if n else (0.0, 0.0)
    return (h / n if n else 0.0), (lo, hi), n


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def build_report(base: dict, tuned: dict) -> tuple[dict, str]:
    b_items, t_items = base["items"], tuned["items"]
    if len(b_items) != len(t_items):
        raise ValueError(f"paired runs differ in n: {len(b_items)} vs {len(t_items)}")
    floor = base.get("best_constant_by_kind", {})

    def line(name, bi, ti):
        ba, bci, n = _acc_ci(bi)
        ta, tci, _ = _acc_ci(ti)
        b, c = mcnemar_counts([x["correct"] for x in bi], [x["correct"] for x in ti])
        p = mcnemar_exact(b, c)
        return {
            "name": name,
            "n": n,
            "base": round(ba, 4),
            "base_ci": [round(bci[0], 4), round(bci[1], 4)],
            "tuned": round(ta, 4),
            "tuned_ci": [round(tci[0], 4), round(tci[1], 4)],
            "delta_pp": round((ta - ba) * 100, 1),
            "mcnemar_p": round(p, 4),
            "sig": _stars(p),
            "regressions_b": b,
            "gains_c": c,
        }

    rows = [line("OVERALL", b_items, t_items)]

    by_kind_b, by_kind_t = defaultdict(list), defaultdict(list)
    for x in b_items:
        by_kind_b[x["kind"]].append(x)
    for x in t_items:
        by_kind_t[x["kind"]].append(x)
    # order kinds by depth (low -> high) for the taxonomy read
    kinds = sorted(by_kind_b, key=lambda k: (by_kind_b[k][0].get("depth") or 0, k))
    kind_rows = [line(k, by_kind_b[k], by_kind_t[k]) for k in kinds]
    for kr in kind_rows:
        kr["floor"] = round(floor.get(kr["name"], float("nan")), 4) if kr["name"] in floor else None

    by_depth_b, by_depth_t = defaultdict(list), defaultdict(list)
    for x in b_items:
        by_depth_b[x.get("depth")].append(x)
    for x in t_items:
        by_depth_t[x.get("depth")].append(x)
    depth_rows = [
        line(f"depth {d}", by_depth_b[d], by_depth_t[d])
        for d in sorted(by_depth_b, key=lambda x: (x is None, x))
    ]

    report = {
        "overall": rows[0],
        "by_kind": kind_rows,
        "by_depth": depth_rows,
        "base_label": base.get("label"),
        "tuned_label": tuned.get("label"),
    }
    return report, _to_md(report)


def _to_md(rep: dict) -> str:
    bl, tl = rep.get("base_label", "base"), rep.get("tuned_label", "tuned")

    def fmt(r, floor=True):
        f = ""
        if floor and r.get("floor") is not None:
            f = f" {r['floor'] * 100:.0f}%"
        return (
            f"| {r['name']} | {r['base'] * 100:.1f}% "
            f"[{r['base_ci'][0] * 100:.0f},{r['base_ci'][1] * 100:.0f}] | "
            f"{r['tuned'] * 100:.1f}% [{r['tuned_ci'][0] * 100:.0f},{r['tuned_ci'][1] * 100:.0f}] | "
            f"{r['delta_pp']:+.1f} | {r['mcnemar_p']:.3f} {r['sig']} |{f}"
        )

    L = []
    o = rep["overall"]
    L.append(f"### {tl} vs {bl}  (n={o['n']}, paired)")
    L.append("")
    L.append(
        f"**Overall: {o['base'] * 100:.1f}% -> {o['tuned'] * 100:.1f}% "
        f"({o['delta_pp']:+.1f}pp), McNemar p={o['mcnemar_p']:.3f} {o['sig']}** "
        f"(regressions={o['regressions_b']}, gains={o['gains_c']})"
    )
    L.append("")
    L.append("| Task (by depth) | Base [95% CI] | Tuned [95% CI] | Δpp | McNemar p | Floor |")
    L.append("|---|---|---|---|---|---|")
    for r in rep["by_kind"]:
        L.append(fmt(r))
    L.append("")
    L.append("| Depth | Base [95% CI] | Tuned [95% CI] | Δpp | McNemar p |")
    L.append("|---|---|---|---|---|")
    for r in rep["by_depth"]:
        L.append(fmt(r, floor=False).rsplit("|", 1)[0] + "|")
    L.append("")
    L.append(
        "_McNemar significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not "
        "significant. Δ is only meaningful when it clears both the CI overlap "
        "and the floor._"
    )
    return "\n".join(L)


def main():
    if len(sys.argv) < 3:
        print("usage: compare.py baseline.json grpo.json")
        sys.exit(1)
    base = json.load(open(sys.argv[1]))
    tuned = json.load(open(sys.argv[2]))
    report, md = build_report(base, tuned)
    outdir = Path(sys.argv[1]).resolve().parent
    (outdir / "comparison.md").write_text(md)
    (outdir / "comparison.json").write_text(json.dumps(report, indent=2))
    print(md)


if __name__ == "__main__":
    main()
