"""Plot actual training/evaluation logs for one run or a baseline/RGMT comparison."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_curve(path: Path, metric: str) -> tuple[list[int], list[float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["iteration"]))
    return [int(row["iteration"]) for row in rows], [float(row[metric]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_log", type=Path)
    parser.add_argument("--eval-csv", type=Path, help="eval.csv produced by --run_dir evaluation")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--compare-log", type=Path)
    parser.add_argument("--compare-eval-csv", type=Path)
    parser.add_argument("--compare-label", default="RGMT")
    parser.add_argument("--out", type=Path, default=Path("runs/training_comparison.png"))
    args = parser.parse_args()
    runs = [(args.label, args.train_log, args.eval_csv)]
    if args.compare_log:
        runs.append((args.compare_label, args.compare_log, args.compare_eval_csv))
    elif args.compare_eval_csv:
        parser.error("--compare-eval-csv requires --compare-log")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for label, training, evaluation in runs:
        axes[0].plot(*read_curve(training, "sim_rew"), label=label)
        axes[1].plot(*read_curve(training, "wm_loss"), label=label)
        if evaluation:
            axes[2].plot(*read_curve(evaluation, "norm_return"), marker="o", label=label)
    for ax, title, metric in zip(
        axes,
        ("Training reward (RGMT includes recovery)", "World-model loss", "Tracking evaluation"),
        ("sim_rew", "wm_loss", "norm_return"),
    ):
        ax.set(title=title, xlabel="iteration", ylabel=metric)
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend()
    if not axes[2].lines:
        axes[2].text(
            0.5, 0.5, "Provide --eval-csv for evaluation results", ha="center", va="center"
        )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
