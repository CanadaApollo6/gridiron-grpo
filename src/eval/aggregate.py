"""Aggregate many runs into one family x recipe x kind table (+ CSV + figure).

Each run is a directory containing a baseline + tuned results JSON (as written by
evaluate.py) and, ideally, a recipe.json (written by train_grpo.py) from which we
read the family / recipe / seed. Reuses compare.build_report so every cell carries
the same Wilson CIs + paired McNemar p-value as a single comparison.

  python src/eval/aggregate.py runs_downloaded/*/        # dirs of downloaded runs
  python src/eval/aggregate.py --out results gg-*/
Writes <out>/aggregate.md, <out>/aggregate.csv, and (if matplotlib) aggregate_heatmap.png.

A run dir is matched loosely: it just needs a baseline-like JSON and a grpo-like
JSON somewhere inside (results/baseline.json + results/grpo.json by default).
"""

import csv
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.compare import build_report  # noqa: E402


def _family(model: str | None) -> str:
    m = (model or "").lower()
    for key, name in [("qwen", "Qwen2.5"), ("smollm", "SmolLM2"),
                      ("llama", "Llama-3.2"), ("olmo", "OLMo-2")]:
        if key in m:
            return name
    return (model or "unknown").split("/")[-1]


def _recipe_tag(cfg: dict) -> str:
    """Map a resolved GRPOConfig back to the EXPERIMENTS.md recipe id."""
    lt = cfg.get("loss_type", "bnpo")
    sr = cfg.get("scale_rewards", True)
    mt = cfg.get("mask_truncated_completions", False)
    eh = cfg.get("epsilon_high")
    beta = cfg.get("beta")
    if lt == "bnpo" and sr and not mt:
        base = "R0"
    elif lt == "dr_grpo" and not sr and not mt:
        base = "R1"          # Dr. GRPO core (no mask_truncated)
    elif lt == "dr_grpo" and not sr and mt:
        base = "R2"          # + DAPO (mask_truncated [+ epsilon_high])
    else:
        base = f"{lt}{'' if sr else '+nsr'}{'+mt' if mt else ''}{'+eh' if eh else ''}"
    if beta is not None and beta != 0.04:
        base += f"(b={beta})"
    return base


def _find(run_dir: Path, *names: str) -> Path | None:
    for n in names:
        hits = list(run_dir.rglob(n))
        if hits:
            return hits[0]
    return None


def load_run(run_dir: str) -> dict | None:
    d = Path(run_dir)
    bp = _find(d, "baseline.json")
    gp = _find(d, "grpo.json")
    if not bp or not gp:
        return None
    base = json.loads(bp.read_text())
    grpo = json.loads(gp.read_text())
    rj = _find(d, "recipe.json")
    recipe = json.loads(rj.read_text()) if rj else {}
    cfg = recipe.get("grpo_config_set", {})
    args = recipe.get("args", {})
    model = args.get("model") or base.get("model")
    meta = {
        "dir": d.name,
        "family": _family(model),
        "recipe": _recipe_tag(cfg) if cfg else "?",
        "seed": args.get("seed"),
        "model": model,
    }
    passk_p = _find(d, "passk_base.json")
    passk = json.loads(passk_p.read_text()) if passk_p else None
    return {"meta": meta, "base": base, "grpo": grpo, "passk": passk}


def aggregate(runs: list[dict]) -> dict:
    """Long-form rows (one per run x kind + an OVERALL) with paired stats."""
    long_rows = []
    for r in runs:
        report, _ = build_report(r["base"], r["grpo"])
        m = r["meta"]
        for kind_row in [report["overall"]] + report["by_kind"]:
            long_rows.append({
                "family": m["family"], "recipe": m["recipe"], "seed": m["seed"],
                "kind": kind_row["name"], "n": kind_row["n"],
                "base": kind_row["base"], "tuned": kind_row["tuned"],
                "delta_pp": kind_row["delta_pp"],
                "ci_lo": kind_row["tuned_ci"][0], "ci_hi": kind_row["tuned_ci"][1],
                "mcnemar_p": kind_row["mcnemar_p"], "sig": kind_row["sig"],
                "floor": kind_row.get("floor"),
            })
    return {"long": long_rows}


def _overall_matrix_md(long_rows: list[dict]) -> str:
    """family x recipe -> overall Δpp (seed-averaged), with seed count + sig."""
    cell = {}
    fams, recs = [], []
    for row in long_rows:
        if row["kind"] != "OVERALL":
            continue
        f, rc = row["family"], row["recipe"]
        cell.setdefault((f, rc), []).append(row)
        if f not in fams:
            fams.append(f)
        if rc not in recs:
            recs.append(rc)
    L = ["### Overall Δaccuracy (pp), seed-averaged — family x recipe", ""]
    L.append("| Family \\ Recipe | " + " | ".join(recs) + " |")
    L.append("|" + "---|" * (len(recs) + 1))
    for f in fams:
        cells = []
        for rc in recs:
            rows = cell.get((f, rc))
            if not rows:
                cells.append("·")
            else:
                deltas = [x["delta_pp"] for x in rows]
                d = statistics.mean(deltas)
                if len(rows) == 1:
                    cells.append(f"{d:+.1f} ({rows[0]['sig']})")
                else:
                    sd = statistics.stdev(deltas)
                    cells.append(f"{d:+.1f}±{sd:.1f} (n={len(rows)})")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("_Cell = mean Δpp across seeds; (sig)=McNemar for a single seed, "
             "(Nseed)=averaged over N seeds. Judge against per-kind floors below._")
    return "\n".join(L)


def _per_kind_matrix_md(long_rows: list[dict]) -> str:
    kinds, fams_recs = [], []
    for row in long_rows:
        if row["kind"] == "OVERALL":
            continue
        if row["kind"] not in kinds:
            kinds.append(row["kind"])
        fr = (row["family"], row["recipe"])
        if fr not in fams_recs:
            fams_recs.append(fr)
    cell = {}
    for row in long_rows:
        if row["kind"] == "OVERALL":
            continue
        cell.setdefault((row["family"], row["recipe"], row["kind"]), []).append(row["delta_pp"])
    L = ["### Per-kind Δaccuracy (pp), seed-averaged", ""]
    L.append("| Family / Recipe | " + " | ".join(kinds) + " |")
    L.append("|" + "---|" * (len(kinds) + 1))
    for (f, rc) in fams_recs:
        cells = []
        for k in kinds:
            v = cell.get((f, rc, k))
            cells.append(f"{statistics.mean(v):+.1f}" if v else "·")
        L.append(f"| {f} / {rc} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("_Q3 read: is Δ positive at low depth (td_or_fg, scrimmage_total) and "
             "negative at high depth (team_points, hundred_yd_rec), consistently across families?_")
    return "\n".join(L)


def write_outputs(runs: list[dict], out: str = "results") -> str:
    agg = aggregate(runs)
    long_rows = agg["long"]
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)

    # CSV (long form)
    cols = ["family", "recipe", "seed", "kind", "n", "base", "tuned",
            "delta_pp", "ci_lo", "ci_hi", "mcnemar_p", "sig", "floor"]
    with (outdir / "aggregate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in long_rows:
            w.writerow(row)

    md = ["# Aggregated results", "",
          f"{len(runs)} run(s); "
          f"{len({(r['meta']['family'], r['meta']['recipe']) for r in runs})} family x recipe cell(s).",
          "", _overall_matrix_md(long_rows), "", _per_kind_matrix_md(long_rows), ""]
    # pass@k learnability rollup if present
    pk = [r for r in runs if r.get("passk")]
    if pk:
        md += ["### Base pass@k (learnability) — overall", "",
               "| Family | pass@1 | pass@8 | pass@64 | frac_never |", "|---|---|---|---|---|"]
        for r in pk:
            o = r["passk"]["overall"]
            md.append(f"| {r['meta']['family']} | {o.get('pass@1')} | {o.get('pass@8')} "
                      f"| {o.get('pass@64')} | {o.get('frac_never')} |")
        md.append("")
    (outdir / "aggregate.md").write_text("\n".join(md))
    return "\n".join(md)


def main():
    args = [a for a in sys.argv[1:]]
    out = "results"
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        del args[i:i + 2]
    patterns = args or ["runs_downloaded/*/"]
    dirs = []
    for p in patterns:
        dirs += [d for d in glob.glob(p)]
    runs = [r for r in (load_run(d) for d in sorted(set(dirs))) if r]
    if not runs:
        print(f"no runs found (looked in: {patterns}). Each dir needs baseline.json + grpo.json.")
        sys.exit(1)
    print(write_outputs(runs, out))
    print(f"\nwrote {out}/aggregate.md and {out}/aggregate.csv  ({len(runs)} runs)")


if __name__ == "__main__":
    main()
