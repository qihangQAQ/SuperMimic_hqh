"""Summarize the two downloaded runs without loading a GPU or changing source logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from unisupermimic.rl.algos.supermimic.networks import make_policy

OUT = Path(__file__).resolve().parent
RUNS = {"baseline": "supermimic_l40_001", "rgmt": "rgmt_recovery_seed0"}
LABELS = {"baseline": "SuperMimic", "rgmt": "SuperMimic + RGMT"}
COLORS = {"baseline": "#2563eb", "rgmt": "#ea580c"}


def main():
    torch.set_num_threads(1)
    summary = {"window": "11000 < iteration <= 12000", "runs": {}}
    logs = {}
    for key, name in RUNS.items():
        root = Path("runs") / name
        with (root / "train_log.csv").open() as stream:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        assert len(rows) == 240
        assert [r["iteration"] for r in rows] == list(range(50, 12001, 50))
        assert all(np.isfinite(v) for r in rows for v in r.values())
        logs[key] = rows
        saved = torch.load(root / "ckpt_012000.pt", map_location="cpu", weights_only=True)
        policy = make_policy(99, 73, 29, saved["args"])
        policy.load_state_dict(saved["pi"])
        events = EventAccumulator(str(root), size_guidance={"scalars": 0})
        events.Reload()
        perf = {}
        for tag in events.Tags()["scalars"]:
            if tag.startswith("Perf/"):
                values = [e.value for e in events.Scalars(tag) if e.step > 11000]
                perf[tag] = float(np.mean(values))
        summary["runs"][key] = {
            "path": str(root),
            "log_rows": len(rows),
            "checkpoint_wall_h": saved["wall_time"] / 3600,
            "policy_parameters": sum(p.numel() for p in policy.parameters()),
            "sim_steps": saved["sim_steps"],
            "last_row": rows[-1],
            "last_1000_mean": {
                k: float(np.mean([r[k] for r in rows if r["iteration"] > 11000])) for k in rows[0]
            },
            "tensorboard_last_1000_mean": perf,
            "args": saved["args"],
        }
    a, b = [
        json.loads((Path("runs") / name / "run_config.json").read_text())["config"]
        for name in RUNS.values()
    ]
    summary["same_motion_list"] = (
        a["env"]["commands"]["motion"]["motion_files"]
        == b["env"]["commands"]["motion"]["motion_files"]
    )
    summary["motion_count"] = len(a["env"]["commands"]["motion"]["motion_files"])
    diagnostics = [
        json.loads(line)
        for line in (Path("runs") / RUNS["rgmt"] / "rgmt_diagnostics.jsonl")
        .read_text()
        .splitlines()
    ]
    summary["diagnostics"] = {
        "recovery_success_min": min(r["recovery_success"] for r in diagnostics),
        "recovery_success_max": max(r["recovery_success"] for r in diagnostics),
        "last_1000_mean": {
            k: float(
                np.mean(
                    [r[k] for r in diagnostics if r["iteration"] > 11000 and r.get(k) is not None]
                )
            )
            for k in diagnostics[-1]
            if k != "iteration"
        },
        "peak_torch_allocated_GiB": max(r["peak_memory_bytes"] for r in diagnostics) / 2**30,
    }
    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4), constrained_layout=True)
    panels = [
        ("ep_ret_norm", "Completed tracking episode return", False),
        ("sim_rew", "All-task step reward (mixed in RGMT)", False),
        ("wm_err8", "Normalized WM error at step 8", True),
        ("term_frac", "Training termination fraction (mixed)", True),
        ("ep_len", "Completed tracking episode length", False),
        ("it_per_s", "Training throughput", False),
    ]
    for ax, (metric, title, log_scale) in zip(axes.flat, panels):
        for key, rows in logs.items():
            x = np.array([r["iteration"] for r in rows])
            y = np.array([r[metric] for r in rows])
            ax.plot(x, y, color=COLORS[key], alpha=0.18, lw=0.8)
            ax.plot(
                x[4:],
                np.convolve(y, np.ones(5) / 5, mode="valid"),
                color=COLORS[key],
                label=LABELS[key],
                lw=1.8,
            )
        ax.set(title=title, xlabel="Training iteration")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.axvline(500, color="#64748b", ls=":", lw=1)
        ax.axvline(6000, color="#ea580c", ls=":", lw=1)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Downloaded training logs | 12,000 iterations, seed 0\nFaint: logged values; solid: 250-iteration moving mean. Dotted: policy start / recovery annealing ends.",
        fontsize=13,
    )
    fig.savefig(OUT / "training_curves.png", dpi=180)
    fig.savefig(OUT / "training_curves.pdf")
    plt.close(fig)
    result_dir = Path("runs/comparison_rgmt_vs_baseline")
    evaluation = {}
    for key, name in RUNS.items():
        recovery = [
            json.loads((result_dir / f"{name}_recovery_seed{s}.json").read_text())
            for s in (0, 1, 2)
        ]
        trials = [t for result in recovery for t in result["per_trial"]]
        successes = [t for t in trials if t["success"]]
        evaluation[key] = {
            "tracking": json.loads((result_dir / f"{name}_tracking.json").read_text()),
            "recovery": {
                "trials": len(trials),
                "successes": len(successes),
                "success_rate": len(successes) / len(trials),
                "mean_first_stable_seconds_success_only": float(
                    np.mean([t["first_stable_seconds"] for t in successes])
                ),
                "by_seed": recovery,
            },
            "short": {
                case: json.loads((result_dir / f"{name}_short_{case}.json").read_text())
                for case in (
                    "control",
                    "push_x_plus",
                    "push_x_minus",
                    "push_y_plus",
                    "push_y_minus",
                    "reference_noise",
                )
            },
        }
    (OUT / "evaluation_summary.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    keys = list(RUNS)
    values = [evaluation[k]["tracking"]["norm_return"] for k in keys]
    axes[0].bar([LABELS[k] for k in keys], values, color=[COLORS[k] for k in keys])
    axes[0].set(title="Full-clip tracking return", ylim=(0, 1))
    values2 = [
        evaluation[k]["tracking"]["metrics"]["aligned_body_position_mm"]["clip_mean"] for k in keys
    ]
    axes[1].bar([LABELS[k] for k in keys], values2, color=[COLORS[k] for k in keys])
    axes[1].set(title="Aligned body position error", ylabel="mm (lower is better)")
    values3 = [100 * evaluation[k]["recovery"]["success_rate"] for k in keys]
    axes[2].bar([LABELS[k] for k in keys], values3, color=[COLORS[k] for k in keys])
    axes[2].set(title="Unassisted standing recovery", ylabel="Success (%)", ylim=(0, 110))
    for ax, values in zip(axes, (values, values2, values3)):
        for i, value in enumerate(values):
            ax.annotate(
                f"{value:.3f}" if ax is axes[0] else f"{value:.1f}",
                (i, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(
        "Final checkpoints | CPU / MuJoCo | 40 seen clips and 96 paired recovery trials",
        fontsize=12,
    )
    fig.savefig(OUT / "evaluation_comparison.png", dpi=180)
    fig.savefig(OUT / "evaluation_comparison.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                "training": summary,
                "evaluation_overview": {
                    k: {
                        "tracking": {m: v for m, v in e["tracking"].items() if m != "per_clip"},
                        "recovery": {m: v for m, v in e["recovery"].items() if m != "by_seed"},
                        "short": {
                            case: {m: v for m, v in r.items() if m != "per_clip"}
                            for case, r in e["short"].items()
                        },
                    }
                    for k, e in evaluation.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
