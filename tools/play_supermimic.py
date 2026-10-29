"""Run a trained policy live in a MuJoCo window with a reference-motion ghost.

Run from the repository root, for example:
    uv run --no-sync tools/play_supermimic.py --ckpt runs/RUN/ckpt_012000.pt --loop

For an external G1 CSV, add --motion-file path/to/motion.csv --motion-fps 60.

The viewer displays the live physics rollout. Mouse perturbations affect only
the display model, not the policy's simulation. Close the window or press
Ctrl+C in the terminal to stop.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

# UniLab's resolver checks its own asset tree even for this project's local
# G1 scene. Set offline mode before importing libraries that initialize HF.
if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
)
from unisupermimic.sim2sim import policy_load_dim_guard
from unisupermimic.tasks.motion_tracking.g1.supermimic.env import SUPPORTED_BACKENDS
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import (
    load_policy,
    make_eval_env,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MAX_STEP_REWARD
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import CONTROL_DT
from unisupermimic.tasks.motion_tracking.g1.supermimic.render import (
    DEFAULT_SCENE,
    flattened_qpos,
    load_comparison_model,
    reference_qpos,
)


class PolicyRollout:
    """One deterministic clip, retaining the terminal state before auto-reset."""

    def __init__(self, env: Any, checkpoint: dict, clip: str):
        if env.num_envs != 1:
            raise ValueError("Interactive playback requires exactly one environment")
        self.env = env
        self.mdp = env.supermimic_mdp
        motion = self.mdp.motion
        if clip not in motion.names:
            raise ValueError(f"Unknown clip {clip!r}. Available: {', '.join(motion.names)}")
        self.clip = clip
        index = motion.names.index(clip)
        self.start = motion.clip_start[index : index + 1].clone()
        self.max_return = float((motion.clip_len[index] - 1).item()) * MAX_STEP_REWARD
        self.hist = int(checkpoint["args"]["hist"])
        self.goal_frames = int(checkpoint["args"]["goal_frames"])
        self.policy = load_policy(env, checkpoint)
        self.reset()

    @torch.inference_mode()
    def reset(self) -> None:
        _, info = self.env.reset_to_frames(self.start.clone())
        self.current = info[SUPERMIMIC_STATE_KEY]
        self.history = None
        self.display_state = self.current["s"]
        self.display_frames = self.current["f"]
        self.steps = 0
        self.total_reward = 0.0
        self.done = False
        self.failed = False

    @torch.inference_mode()
    def step(self) -> None:
        if self.done:
            raise RuntimeError("Reset the rollout before stepping a finished clip")
        current = self.current
        obs = self.mdp.obs(
            unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"]
        )
        if self.history is None:
            self.history = obs.unsqueeze(1).repeat(1, self.hist, 1)
        else:
            self.history = torch.cat((self.history[:, 1:], obs.unsqueeze(1)), dim=1)
        action = self.policy(self.history, self.mdp.goal(current["f"], self.goal_frames))
        state = self.env.step(action)
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]
        self.total_reward += float(transition["r"][0].item())
        self.steps += 1
        self.done = bool(transition["done"][0].item())
        self.failed = bool(transition["term"][0].item())
        # env.step may already have reset: draw the actual terminal state.
        self.display_state = transition["s1"]
        self.display_frames = transition["f"] + 1
        self.current = state.info[SUPERMIMIC_STATE_KEY]

    @torch.inference_mode()
    def qpos(self) -> tuple[np.ndarray, np.ndarray]:
        policy = flattened_qpos(self.display_state)[0].cpu().numpy()
        reference = reference_qpos(self.mdp.motion, self.display_frames)[0].cpu().numpy()
        return policy, reference

    def report(self) -> str:
        return (
            f"{self.clip}: steps={self.steps}, seconds={self.steps * CONTROL_DT:.2f}, "
            f"norm_return={self.total_reward / self.max_return:.4f}, failed={self.failed}"
        )


def play(rollout: PolicyRollout, *, loop: bool, speed: float, align_reference: bool) -> None:
    import mujoco
    import mujoco.viewer

    model = load_comparison_model(DEFAULT_SCENE)
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55

    def update() -> None:
        policy, reference = rollout.qpos()
        nq = len(policy)
        data.qpos[:nq] = policy
        data.qpos[nq:] = reference
        if align_reference:
            data.qpos[nq : nq + 2] = policy[:2]
        data.time = rollout.steps * CONTROL_DT
        mujoco.mj_forward(model, data)

    update()
    print(
        "Solid: policy; blue ghost: reference. Close the window or press Ctrl+C to stop.",
        flush=True,
    )
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.distance = 3.6
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -12
            viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.75]
        viewer.sync()
        while viewer.is_running():
            started = time.monotonic()
            if rollout.done:
                if not loop:
                    time.sleep(0.02)
                    viewer.sync()
                    continue
                rollout.reset()
            else:
                rollout.step()
                if rollout.done:
                    print(rollout.report(), flush=True)
            with viewer.lock():
                update()
                viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.75]
            viewer.sync()
            deadline = started + CONTROL_DT / speed
            while viewer.is_running():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.02))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--clip", help="Clip name; defaults to the external CSV stem or walk1_subject1"
    )
    parser.add_argument("--motion-file", type=Path, help="External G1 motion CSV (36 columns)")
    parser.add_argument(
        "--motion-fps", type=float, help="Source CSV frame rate; required with --motion-file"
    )
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default="mujoco")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Policy device; auto uses CPU for single-environment MuJoCo playback",
    )
    parser.add_argument("--loop", action="store_true", help="Restart after completion or failure")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Playback speed (default: real time)"
    )
    parser.add_argument(
        "--align-reference", action="store_true", help="Align reference XY to the policy robot"
    )
    args = parser.parse_args(argv)
    if not args.ckpt.is_file():
        parser.error(f"Checkpoint not found: {args.ckpt}")
    if args.motion_file is not None:
        if not args.motion_file.is_file():
            parser.error(f"Motion CSV not found: {args.motion_file}")
        if args.motion_fps is None:
            parser.error(
                "--motion-file requires --motion-fps (the CSV does not store its frame rate)"
            )
        if args.clip is not None and args.clip != args.motion_file.stem:
            parser.error("--clip must match the filename without .csv when using --motion-file")
    elif args.motion_fps is not None:
        parser.error("--motion-fps requires --motion-file")
    if args.motion_fps is not None and (not math.isfinite(args.motion_fps) or args.motion_fps <= 0):
        parser.error("--motion-fps must be finite and positive")
    clip = args.clip or (args.motion_file.stem if args.motion_file else "walk1_subject1")
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        parser.error("A desktop DISPLAY is required; run this command in a desktop terminal")
    if os.environ.get("MUJOCO_GL", "glfw").lower() not in ("glfw", ""):
        parser.error("Interactive playback requires MUJOCO_GL=glfw")

    device = None if args.device == "auto" else args.device
    if args.device == "auto" and args.backend == "mujoco":
        device = "cpu"
    env = None
    try:
        print(f"Loading local motion assets ({args.backend})...", flush=True)
        env = make_eval_env(
            args.ckpt.parent,
            backend=args.backend,
            num_envs=1,
            device=device,
            motion_files=[str(args.motion_file.resolve())] if args.motion_file else None,
            src_fps=args.motion_fps,
        )
        print(f"Loading policy on {env.device}...", flush=True)
        checkpoint = torch.load(args.ckpt, map_location=env.device, weights_only=True)
        with policy_load_dim_guard(
            env_obs_dim=env.supermimic_mdp.obs_dim,
            env_action_dim=env.action_manager.total_action_dim,
            algo_name="supermimic",
        ):
            rollout = PolicyRollout(env, checkpoint, clip)
        play(rollout, loop=args.loop, speed=args.speed, align_reference=args.align_reference)
    except KeyboardInterrupt:
        pass
    finally:
        if env is not None:
            env.close()
            env.backend.close()


def cli() -> None:
    # This viewer writes no training state. Let the OS stop it on Ctrl+C,
    # including while a native viewer lock prevents Python signal dispatch.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()


if __name__ == "__main__":
    cli()
