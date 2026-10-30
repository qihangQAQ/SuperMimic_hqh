"""Reproduce the paired 10-second CPU/MuJoCo tracking perturbation comparison.

Run from the repository root with CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1.
Results go to runs/comparison_rgmt_vs_baseline; training files are never changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy, make_eval_env


@torch.inference_mode()
def run_trial(env, checkpoint, force, noise):
    mdp = env.supermimic_mdp
    motion = mdp.motion
    count = motion.num_tracking_clips
    policy = load_policy(env, checkpoint, device="cpu")
    _, info = env.reset_to_frames(motion.clip_start[:count].clone())
    current = info[SUPERMIMIC_STATE_KEY]
    initial_hash = hashlib.sha256(current["s"].numpy().tobytes()).hexdigest()
    goal_frames, hist = checkpoint["args"]["goal_frames"], checkpoint["args"]["hist"]
    history = None
    reward = torch.zeros(count)
    steps = torch.zeros(count, dtype=torch.long)
    failed = torch.zeros(count, dtype=torch.bool)
    finished = torch.zeros_like(failed)
    errors = torch.zeros(count)
    horizon_steps = torch.minimum(motion.clip_len[:count] - 1, torch.full((count,), 500))
    generator = torch.Generator().manual_seed(0)
    scale = torch.tensor([0.02] * 3 + [0.05] * 6 + [0.2] * 6 + [0.1] * 29 + [0.01] * 29)
    alive_at_push = None
    for step_index in range(500):
        obs = mdp.obs(unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"])
        history = (
            obs.unsqueeze(1).repeat(1, hist, 1)
            if history is None
            else torch.cat((history[:, 1:], obs.unsqueeze(1)), 1)
        )
        goal = mdp.goal(current["f"], goal_frames)
        if noise:
            goal = (
                goal
                + (2 * torch.rand(goal.shape, generator=generator) - 1)
                * scale.repeat(goal_frames)
                * noise
            )
        if step_index == 50:
            alive_at_push = int((~finished).sum())
        if any(force) and 50 <= step_index < 60:
            forces = torch.tensor(force, dtype=torch.float32).repeat(count, 1)
            forces[finished] = 0
            env.set_external_force(forces)
        state = env.step(policy(history, goal))
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]
        alive = ~finished
        reward += torch.where(alive, transition["r"], 0)
        steps += alive
        measured = mdp.tracking_metrics(
            unflatten_state(transition["s1"]), mdp.advance_frames(transition["f"])
        )
        errors += torch.where(alive, measured["aligned_body_position_mm"], 0)
        failed |= alive & transition["term"]
        finished |= alive & transition["done"]
        current = state.info[SUPERMIMIC_STATE_KEY]
        if bool(finished.all()):
            break
    return {
        "backend": "mujoco",
        "device": "cpu",
        "checkpoint_iteration": 12000,
        "protocol": {
            "seconds": 10,
            "push_at": 1,
            "push_duration": 0.2,
            "push_force_N": list(force),
            "reference_noise": noise,
            "seed": 0,
            "assistance": False,
            "initial_state_sha256": initial_hash,
        },
        "clips": count,
        "alive_at_push": alive_at_push,
        "survived_or_completed": int(((steps >= horizon_steps) & ~failed).sum()),
        "n_failed": int(failed.sum()),
        "horizon_normalized_return": float((reward / (horizon_steps * mdp.max_step_reward)).mean()),
        "aligned_body_position_mm": float((errors / steps.clamp(min=1)).mean()),
        "per_clip": {
            name: {
                "steps": int(steps[i]),
                "failed": bool(failed[i]),
                "horizon_normalized_return": float(
                    reward[i] / (horizon_steps[i] * mdp.max_step_reward)
                ),
                "aligned_body_position_mm": float(errors[i] / steps[i].clamp(min=1)),
            }
            for i, name in enumerate(motion.names[:count])
        },
    }


def main():
    torch.set_num_threads(1)
    output = Path("runs/comparison_rgmt_vs_baseline")
    output.mkdir(parents=True, exist_ok=True)
    cases = {
        "control": ((0, 0, 0), 0),
        "push_x_plus": ((200, 0, 0), 0),
        "push_x_minus": ((-200, 0, 0), 0),
        "push_y_plus": ((0, 200, 0), 0),
        "push_y_minus": ((0, -200, 0), 0),
        "reference_noise": ((0, 0, 0), 1),
    }
    initial_hash = None
    for name in ("supermimic_l40_001", "rgmt_recovery_seed0"):
        root = Path("runs") / name
        checkpoint = torch.load(root / "ckpt_012000.pt", map_location="cpu", weights_only=True)
        env = make_eval_env(root, backend="mujoco", device="cpu")
        try:
            for case, (force, noise) in cases.items():
                result = run_trial(env, checkpoint, force, noise)
                current_hash = result["protocol"]["initial_state_sha256"]
                if initial_hash is None:
                    initial_hash = current_hash
                assert current_hash == initial_hash, "comparison initial states differ"
                (output / f"{name}_short_{case}.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print(
                    name,
                    case,
                    {k: v for k, v in result.items() if k not in ("per_clip", "protocol")},
                    flush=True,
                )
        finally:
            env.close()
            env.backend.close()


if __name__ == "__main__":
    main()
