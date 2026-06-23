"""Render the README hero figure: the two-model thesis in one chart.

GRPO amplifies a weak base (SmolLM2-1.7B) but barely moves a saturated one
(Qwen2.5-1.5B) on the *same* task with the *same* recipe — the gain lives where
the base has reachable headroom. Reads the per-run comparison.json files.

  python src/eval/make_hero.py \
      runs_dl/gg-smollm2-r0-s7/results/comparison.json \
      runs_dl/gg-qwen15b-r0-s7/results/comparison.json \
      assets/hero.png

Each comparison.json is produced by src/eval/compare.py (base/tuned + McNemar).
"""

import json
import sys

GRAY = "#888888"
NV_GREEN = "#76B900"  # NVIDIA green for the tuned bars


def main():
    if len(sys.argv) < 4:
        print("usage: make_hero.py smollm2_comparison.json qwen_comparison.json out.png")
        sys.exit(1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (display label, sublabel, path)
    runs = [
        ("SmolLM2-1.7B", "weak base — room to learn", sys.argv[1]),
        ("Qwen2.5-1.5B", "already-capable base (control)", sys.argv[2]),
    ]
    out = sys.argv[3]

    fig, ax = plt.subplots(figsize=(8.2, 5))
    w = 0.34
    for i, (name, sub, path) in enumerate(runs):
        o = json.load(open(path))["overall"]
        base, tuned = o["base"], o["tuned"]
        d, sig = o["delta_pp"], o["sig"]
        ax.bar(i - w / 2, base, w, color=GRAY, label="base" if i == 0 else None)
        ax.bar(i + w / 2, tuned, w, color=NV_GREEN, label="GRPO-tuned" if i == 0 else None)
        ax.text(i - w / 2, base + 0.008, f"{base:.1%}", ha="center", va="bottom", fontsize=10, color="#444")
        ax.text(i + w / 2, tuned + 0.008, f"{tuned:.1%}", ha="center", va="bottom", fontsize=10,
                color=NV_GREEN, fontweight="bold")
        tag = "ns" if sig == "ns" else sig
        ax.text(i, max(base, tuned) + 0.05, f"Δ {d:+.1f}pp  ({tag})", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{n}\n{s}" for n, s, _ in runs], fontsize=10)
    ax.set_ylabel("verifiable accuracy (held-out, n=800)")
    ax.set_ylim(0, 0.42)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%"])
    ax.set_title("Same task, same recipe: GRPO amplifies a weak base, not a saturated one",
                 fontsize=12)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
