"""Print the filled README results table from baseline + trained JSONs.

  python src/eval/results_to_md.py results/baseline.json results/grpo.json

Auto-fills the numbers; you still write the prose beats by hand. Stdlib only.
"""

import json
import sys

# task-kind -> pretty label (keep in sync with src/data/tasks.py)
PRETTY = {
    "scrimmage_total": "Yards from scrimmage (numeric)",
    "team_points": "Team points (numeric)",
    "total_tds": "Total touchdowns (numeric)",
    "most_scrimmage": "Most scrimmage yards (argmax)",
    "hundred_yd_rec": "100+ yard receivers (set)",
    "td_or_fg": "TD-or-FG (decision)",
}


def pp(base: float, tuned: float) -> str:
    d = (tuned - base) * 100
    return f"{d:+.1f}pp"


def row(label: str, base: float, tuned: float, bold: bool = False) -> str:
    b = f"{base * 100:.1f}%"
    t = f"{tuned * 100:.1f}%"
    d = pp(base, tuned)
    if bold:
        return f"| **{label}** | **{b}** | **{t}** | **{d}** |"
    return f"| {label} | {b} | {t} | {d} |"


def main():
    if len(sys.argv) < 3:
        print("usage: results_to_md.py baseline.json trained.json")
        sys.exit(1)
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))

    print(
        f"Overall: {a['accuracy'] * 100:.1f}% -> {b['accuracy'] * 100:.1f}% "
        f"({pp(a['accuracy'], b['accuracy'])}) on n={a['n']}\n"
    )

    print("| Metric | Base model | GRPO-tuned | \u0394 |")
    print("|---|---|---|---|")
    print(row("Overall", a["accuracy"], b["accuracy"], bold=True))
    kinds = list(PRETTY) + [k for k in (set(a["by_kind"]) | set(b["by_kind"])) if k not in PRETTY]
    for k in kinds:
        if k in a["by_kind"] or k in b["by_kind"]:
            print(row(PRETTY.get(k, k), a["by_kind"].get(k, 0.0), b["by_kind"].get(k, 0.0)))


if __name__ == "__main__":
    main()
