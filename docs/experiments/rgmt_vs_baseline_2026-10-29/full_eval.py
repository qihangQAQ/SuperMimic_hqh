"""Reproduce full-clip tracking and paired recovery evaluations of the two runs.

Run from the repository root. Defaults use CPU/MuJoCo; selecting mjwarp requires
an explicit --backend mjwarp --device cuda:0 and a separate --output directory.
This script evaluates saved models and never starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import (
    evaluate,
    evaluate_recovery,
    make_eval_env,
)


def digest(state):
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco", "mjwarp"), default="mujoco")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/comparison_rgmt_vs_baseline"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    initial_hashes = {}
    for name in ("supermimic_l40_001", "rgmt_recovery_seed0"):
        root = Path("runs") / name
        checkpoint = torch.load(root / "ckpt_012000.pt", map_location="cpu", weights_only=True)
        env = make_eval_env(root, backend=args.backend, device=args.device)
        try:
            tracking = evaluate(env, checkpoint, device=args.device, seed=0)
            tracking.update(backend=args.backend, device=args.device)
            (args.output / f"{name}_tracking.json").write_text(
                json.dumps(tracking, indent=2) + "\n"
            )
            print(
                name, "tracking", tracking["norm_return"], tracking["completion_rate"], flush=True
            )
        finally:
            env.close()
            env.backend.close()

        env = make_eval_env(
            root, backend=args.backend, device=args.device, num_envs=32, recovery_mode=True
        )
        try:
            # The upstream environment's first reset initializes the simulator.
            # Initialize once before the seeded recovery resets used in the comparison.
            env.reset(seed=0)
            original_reset = env.reset_to_recovery
            captured = {}

            def capture_reset(seed=0):
                result = original_reset(seed=seed)
                captured["state"] = result[1][SUPERMIMIC_STATE_KEY]["s"].detach().cpu().clone()
                return result

            env.reset_to_recovery = capture_reset
            for seed in (0, 1, 2):
                recovery = evaluate_recovery(env, checkpoint, seed=seed)
                state = captured["state"]
                current_hash = digest(state)
                if seed not in initial_hashes:
                    initial_hashes[seed] = current_hash
                assert current_hash == initial_hashes[seed], "paired recovery initial states differ"
                position, rotation, *_ = unflatten_state(state)
                tilt = torch.rad2deg(torch.acos(rotation[:, 2, 2].clamp(-1, 1)))
                for index, trial in enumerate(recovery["per_trial"]):
                    trial.update(
                        initial_tilt_degrees=float(tilt[index]),
                        initial_root_height_m=float(position[index, 2]),
                    )
                recovery.update(
                    backend=args.backend,
                    device=args.device,
                    paired_initial_state=True,
                    initial_state_sha256=current_hash,
                )
                (args.output / f"{name}_recovery_seed{seed}.json").write_text(
                    json.dumps(recovery, indent=2) + "\n"
                )
                print(name, "recovery", seed, recovery["success_rate"], flush=True)
        finally:
            env.close()
            env.backend.close()


if __name__ == "__main__":
    main()
