"""Learning curves from the per-episode training logs.

Overlays several experiments on the same axes so a change can be read as a
change, not as two pictures that have to be compared from memory. Seeds of the
same experiment are averaged and their spread shaded.

    python3 tools/diagnose.py curves --out figures/s2.png \\
        --exp s2_curriculum s2_direct --metric coins
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EXPERIMENTS = ROOT / "experiments"


def load_seed(seed_dir, metric):
    """Concatenate every phase log of one seed, in phase order."""
    values = []
    boundaries = []
    for path in sorted(seed_dir.glob("phase*_*.csv")):
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        values.extend(float(r[metric]) for r in rows)
        boundaries.append(len(values))
    return np.array(values), boundaries[:-1]


def smooth(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", required=True)
    parser.add_argument("--metric", default="coins")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5))
    for exp_id in args.exp:
        exp_dir = EXPERIMENTS / exp_id
        seeds = sorted(exp_dir.glob("seed*"))
        if not seeds:
            print("no seeds found for %s" % exp_id)
            continue
        curves, marks = [], []
        for seed_dir in seeds:
            values, boundaries = load_seed(seed_dir, args.metric)
            curves.append(smooth(values, args.window))
            marks = boundaries
        length = min(len(c) for c in curves)
        stacked = np.stack([c[:length] for c in curves])
        mean = stacked.mean(axis=0)
        x = np.arange(length) + args.window
        line, = ax.plot(x, mean, label="%s (%d seeds)" % (exp_id, len(curves)))
        if len(curves) > 1:
            ax.fill_between(x, stacked.min(axis=0), stacked.max(axis=0),
                            alpha=0.18, color=line.get_color())
        for boundary in marks:
            ax.axvline(boundary, color=line.get_color(), ls=":", alpha=0.5)

    ax.set_xlabel("training episode")
    ax.set_ylabel("%s per episode (moving average of %d)" % (args.metric, args.window))
    ax.set_title(args.title or ("learning curve: %s" % args.metric))
    ax.legend()
    ax.grid(alpha=0.3)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
