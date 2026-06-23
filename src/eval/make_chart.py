"""Turn two results JSONs (baseline + trained) into the before/after chart that
headlines the README and the blog post.

  python src/eval/make_chart.py results/baseline.json results/grpo.json
"""

import json
import sys


def main():
    if len(sys.argv) < 3:
        print("usage: make_chart.py baseline.json trained.json")
        sys.exit(1)

    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: pip install matplotlib")
        # still print the numbers so the run isn't wasted
        print(f"{a['label']}: {a['accuracy']:.1%}   {b['label']}: {b['accuracy']:.1%}")
        return

    kinds = sorted(set(a["by_kind"]) | set(b["by_kind"]))
    labels = ["OVERALL"] + kinds
    av = [a["accuracy"]] + [a["by_kind"].get(k, 0) for k in kinds]
    bv = [b["accuracy"]] + [b["by_kind"].get(k, 0) for k in kinds]

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - w / 2 for i in x], av, w, label=a["label"], color="#888888")
    ax.bar([i + w / 2 for i in x], bv, w, label=b["label"], color="#76B900")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Verifiable accuracy: base vs. GRPO-tuned")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/before_after.png", dpi=150)
    print("wrote results/before_after.png")
    print(f"OVERALL  {a['label']}: {a['accuracy']:.1%}  ->  {b['label']}: {b['accuracy']:.1%}")


if __name__ == "__main__":
    main()
