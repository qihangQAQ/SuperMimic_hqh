"""Deterministic full-clip evaluation and checkpoint watching for G1 SuperMimic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    State,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.env import SUPPORTED_BACKENDS
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MAX_STEP_REWARD, MDP
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import CONTROL_DT

if TYPE_CHECKING:
    from unisupermimic.tasks.motion_tracking.g1.supermimic.env import SuperMimicManagerBasedRlEnv

EVAL_CSV_FIELDS = (
    "iteration",
    "train_wall_h",
    "saved_at",
    "eval_at",
    "norm_return",
    "norm_return_framew",
    "n_failed",
    "failed",
)


DEFAULT_EVAL_NUM_ENVS = 4096


def _resolve_policy_device(
    env_device: torch.device,
    device: str | torch.device | None,
    requested: str | torch.device,
) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if sys.platform == "darwin" and env_device.type == "cpu" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device(env_device if device is None else device)


def load_policy(
    env: SuperMimicManagerBasedRlEnv,
    checkpoint: dict,
    device: str | torch.device | None = None,
) -> networks.Policy | networks.RGMTPolicy:
    mdp = cast(MDP, env.supermimic_mdp)
    device = torch.device(env.device if device is None else device)
    args = checkpoint["args"]
    networks.AMP_DTYPE = torch.bfloat16 if args.get("amp", "none") == "bf16" else None
    if getattr(mdp, "reference_past", -1) != args.get("reference_past", -1):
        raise ValueError("checkpoint and evaluation reference windows differ")
    policy: networks.Policy | networks.RGMTPolicy = networks.make_policy(
        mdp.obs_dim, mdp.goal_frame_dim, mdp.num_joints, args
    )
    policy = policy.to(device)
    policy.load_state_dict(checkpoint["pi"])
    policy.eval()
    return policy


def _reference_noise(clip_seeds: torch.Tensor, steps: torch.Tensor, width: int) -> torch.Tensor:
    """Stateless uniform noise, independent of environment count and scheduling order."""
    features = torch.arange(width, device=clip_seeds.device, dtype=torch.int64)
    value = (clip_seeds[:, None] + steps[:, None] * 1_000_003 + features) & 0xFFFFFFFF
    value = ((value ^ (value >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    value = ((value ^ (value >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    value = value ^ (value >> 16)
    return value.to(torch.float64).div(2**32).mul(2).sub(1).float()


@torch.inference_mode()
def evaluate(
    env: SuperMimicManagerBasedRlEnv,
    checkpoint: dict,
    device: str | torch.device | None = None,
    *,
    policy_device: str | torch.device = "auto",
    reference_noise: float = 0.0,
    push_force: tuple[float, float, float] = (0.0, 0.0, 0.0),
    push_at: float = 1.0,
    push_duration: float = 0.1,
    seed: int = 0,
) -> dict:
    """Evaluate tracking clips with a reusable pool and per-clip perturbation clocks."""
    mdp = cast(MDP, env.supermimic_mdp)
    motion = mdp.motion
    args = checkpoint["args"]
    goal_frames, hist = int(args["goal_frames"]), int(args["hist"])
    inference_device = _resolve_policy_device(torch.device(env.device), device, policy_device)
    policy = load_policy(env, checkpoint, inference_device)
    if inference_device.type == "mps":
        networks.AMP_DTYPE = None
    if (
        not math.isfinite(reference_noise)
        or reference_noise < 0
        or not math.isfinite(push_at)
        or push_at < 0
        or not math.isfinite(push_duration)
        or push_duration <= 0
    ):
        raise ValueError("invalid evaluation noise or push timing")
    if len(push_force) != 3 or not all(math.isfinite(x) for x in push_force):
        raise ValueError("push_force must contain three finite world-frame force values")
    if reference_noise and mdp.goal_frame_dim != 73:
        raise ValueError("reference noise protocol requires 73D G1 reference features")

    num_clips = getattr(motion, "num_tracking_clips", len(motion.names))
    num_envs = int(env.num_envs)
    if not 0 < num_envs <= num_clips:
        raise ValueError("evaluation environment count must be positive and must not exceed clips")
    clip_order = motion.clip_len[:num_clips].argsort(descending=True, stable=True)
    ordered_starts = motion.clip_start[clip_order].clone()
    _, reset_info = env.reset_to_frames(ordered_starts[:num_envs].clone())
    current = reset_info[SUPERMIMIC_STATE_KEY]
    assigned = clip_order[:num_envs].to(env.device).clone()
    next_clip = num_envs
    returns = torch.zeros(num_clips, device=env.device)
    steps = torch.zeros(num_clips, dtype=torch.int64, device=env.device)
    failed = torch.zeros(num_clips, dtype=torch.bool, device=env.device)
    metric_sums: dict[str, torch.Tensor] = {}
    history = None
    step_index = 0
    started = time.perf_counter()
    noise_scale = torch.tensor(
        [0.02] * 3 + [0.05] * 6 + [0.2] * 6 + [0.1] * 29 + [0.01] * 29,
        device=env.device,
    ).repeat(goal_frames)
    clip_seeds = (
        torch.tensor(
            [
                int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:4], "little")
                for name in motion.names[:num_clips]
            ],
            dtype=torch.int64,
            device=env.device,
        )
        if reference_noise
        else None
    )
    force_vector = torch.tensor(push_force, device=env.device)
    push_start = round(push_at / CONTROL_DT)
    push_end = round((push_at + push_duration) / CONTROL_DT)

    while bool((assigned >= 0).any()):
        active = assigned >= 0
        active_ids = active.nonzero(as_tuple=False).flatten()
        active_clips = assigned[active_ids]
        local_steps = steps[assigned.clamp(min=0)]
        obs = mdp.obs(
            unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"]
        ).to(inference_device)
        history = (
            obs.unsqueeze(1).repeat(1, hist, 1)
            if history is None
            else torch.cat((history[:, 1:], obs.unsqueeze(1)), dim=1)
        )
        goal = mdp.goal(current["f"], goal_frames)
        if clip_seeds is not None:
            noise = _reference_noise(clip_seeds[assigned.clamp(min=0)], local_steps, goal.shape[-1])
            goal = goal + noise * noise_scale * reference_noise
        action = policy(history, goal.to(inference_device))
        if any(push_force):
            pushed = active & (local_steps >= push_start) & (local_steps < push_end)
            env.set_external_force(pushed[:, None] * force_vector)
        state = env.step(action.to(env.device))
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]
        returns[active_clips] += transition["r"][active_ids]
        steps[active_clips] += 1
        failed[active_clips] |= transition["term"][active_ids]
        if isinstance(mdp, MDP) and "s1" in transition:
            errors = mdp.tracking_metrics(
                unflatten_state(transition["s1"]), mdp.advance_frames(transition["f"])
            )
            for name, error in errors.items():
                if name not in metric_sums:
                    metric_sums[name] = torch.zeros_like(returns)
                metric_sums[name][active_clips] += error[active_ids]
        current = state.info[SUPERMIMIC_STATE_KEY]
        step_index += 1
        completed = (transition["done"] & active).nonzero(as_tuple=False).flatten()
        reset_count = min(completed.numel(), num_clips - next_clip)
        if reset_count:
            reset_ids = completed[:reset_count]
            reset_clips = clip_order[next_clip : next_clip + reset_count].to(env.device)
            _, reset_info = env.reset_to_frames(
                ordered_starts[next_clip : next_clip + reset_count], env_ids=reset_ids
            )
            current = reset_info[SUPERMIMIC_STATE_KEY]
            reset_obs = mdp.obs(
                cast(State, tuple(value[reset_ids] for value in unflatten_state(current["s"]))),
                current["a_prev"][reset_ids],
                current["xi"][reset_ids],
                current["f"][reset_ids],
            ).to(inference_device)
            history[reset_ids.to(inference_device)] = reset_obs.unsqueeze(1).expand(-1, hist, -1)
            assigned[reset_ids] = reset_clips
            next_clip += reset_count
        assigned[completed[reset_count:]] = -1

    max_returns = (motion.clip_len[:num_clips] - 1).float() * MAX_STEP_REWARD
    normalized = returns / max_returns
    failed_indices = failed.nonzero(as_tuple=False).flatten().tolist()
    result = {
        "norm_return": normalized.mean().item(),
        "norm_return_framew": (returns.sum() / max_returns.sum()).item(),
        "n_failed": len(failed_indices),
        "failed": [motion.names[index] for index in failed_indices],
        "per_clip": {
            name: {
                "norm_return": normalized[index].item(),
                "steps": steps[index].item(),
                "failed": failed[index].item(),
            }
            for index, name in enumerate(motion.names[:num_clips])
        },
        "completion_rate": float(
            ((steps >= motion.clip_len[:num_clips] - 1) & ~failed).float().mean()
        ),
        "policy_weights": "ema" if checkpoint.get("pi_ema_updates", 0) else "live",
        "protocol": dict(
            reference_noise=reference_noise,
            push_force=list(push_force),
            push_at=push_at,
            push_duration=push_duration,
            seed=seed,
            position_alignment="reward heading and root XY",
            assistance=False,
            reference_noise_rng="clip-step-hash-v1",
        ),
        "performance": dict(
            wall_seconds=time.perf_counter() - started,
            sim_transitions=step_index * num_envs,
            evaluated_transitions=int(steps.sum()),
            num_envs=num_envs,
        ),
    }
    if metric_sums:
        result["metrics"] = {
            name: {
                "clip_mean": float((values / steps.clamp(min=1)).mean()),
                "frame_mean": float(values.sum() / steps.sum().clamp(min=1)),
            }
            for name, values in metric_sums.items()
        }
    return result


@torch.inference_mode()
def evaluate_recovery(
    env: SuperMimicManagerBasedRlEnv,
    checkpoint: dict,
    *,
    seed: int = 0,
) -> dict:
    """One unassisted randomized recovery trial per environment; success at deadline."""
    mdp = cast(MDP, env.supermimic_mdp)
    if not mdp.recovery.enabled:
        raise ValueError("recovery evaluation requires a recovery environment")
    policy = load_policy(env, checkpoint)
    _, info = env.reset_to_recovery(seed=seed)
    current = info[SUPERMIMIC_STATE_KEY]
    if bool(current["force"].any()):
        raise ValueError("recovery evaluation must disable assistance (eval_mode=true)")
    hist = int(checkpoint["args"]["hist"])
    frames = int(checkpoint["args"]["goal_frames"])
    stable = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    required = max(1, round(mdp.recovery.stable_seconds / CONTROL_DT))
    first = torch.full((env.num_envs,), float("inf"), device=env.device)
    finished = torch.zeros_like(stable, dtype=torch.bool)
    success = torch.zeros_like(finished)
    history = None
    for index in range(mdp.recovery_steps):
        obs = mdp.obs(unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"])
        history = (
            obs.unsqueeze(1).repeat(1, hist, 1)
            if history is None
            else torch.cat((history[:, 1:], obs.unsqueeze(1)), 1)
        )
        action = policy(history, mdp.goal(current["f"], frames))
        transition_state = env.step(action)
        transition = transition_state.info[SUPERMIMIC_TRANSITION_KEY]
        alive = ~finished
        recovered = mdp.recovered(unflatten_state(transition["s1"]))
        updated = torch.where(recovered, stable + 1, 0)
        stable = torch.where(alive, updated, stable)
        first = torch.where(
            alive & (stable >= required) & first.isinf(), (index + 1) * CONTROL_DT, first
        )
        ended = transition["done"] & alive
        at_deadline = transition["f"] + 1 >= mdp.motion.recovery_end
        success |= ended & at_deadline & (stable >= required) & ~transition["term"]
        finished |= ended
        current = transition_state.info[SUPERMIMIC_STATE_KEY]
        if bool(finished.all()):
            break
    times = first[success]
    return {
        "trials": env.num_envs,
        "seed": seed,
        "assistance": False,
        "policy_weights": "ema" if checkpoint.get("pi_ema_updates", 0) else "live",
        "window_seconds": mdp.recovery.seconds,
        "success_rate": float(success.float().mean()),
        "mean_recovery_seconds": float(times.mean()) if times.numel() else None,
        "per_trial": [
            {
                "success": bool(ok),
                "first_stable_seconds": float(t) if math.isfinite(float(t)) else None,
            }
            for ok, t in zip(success, first)
        ],
    }


def _checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("ckpt_"))
    except ValueError as exc:
        raise ValueError(f"Invalid SuperMimic checkpoint name: {path.name}") from exc


def _evaluated_iterations(path: Path) -> set[int]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(encoding="utf-8", newline="") as stream:
        return {int(row["iteration"]) for row in csv.DictReader(stream) if row.get("iteration")}


def _atomic_write_json(path: Path, result: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=1)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _append_eval_row(
    path: Path,
    iteration: int,
    checkpoint: dict,
    evaluated_at: str,
    result: dict,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                iteration,
                f"{checkpoint['wall_time'] / 3600:.4f}",
                checkpoint["saved_at"],
                evaluated_at,
                f"{result['norm_return']:.4f}",
                f"{result['norm_return_framew']:.4f}",
                result["n_failed"],
                "|".join(result["failed"]),
            )
        )


def watch(
    env: SuperMimicManagerBasedRlEnv,
    run_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    every: int = 1,
    poll_seconds: float = 60.0,
    success_threshold: float | None = 0.82,
    once: bool = False,
) -> dict | None:
    """Evaluate new 2000-iteration checkpoints, optionally returning after one scan."""
    if every <= 0:
        raise ValueError("every must be positive")
    if poll_seconds < 0:
        raise ValueError("poll_seconds must be non-negative")
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"SuperMimic run directory does not exist: {run_dir}")
    device = torch.device(env.device if device is None else device)
    csv_path = run_dir / "eval.csv"
    evaluated = _evaluated_iterations(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerow(EVAL_CSV_FIELDS)

    latest: dict | None = None
    while True:
        checkpoints = sorted(run_dir.glob("ckpt_*.pt"), key=_checkpoint_iteration)
        for checkpoint_path in checkpoints:
            iteration = _checkpoint_iteration(checkpoint_path)
            if iteration in evaluated or iteration % 2000 != 0:
                continue
            if (iteration // 2000) % every != 0:
                continue
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            started = time.monotonic()
            result: dict = evaluate(env, checkpoint, device=device)
            evaluated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            _atomic_write_json(
                checkpoint_path.with_name(f"{checkpoint_path.stem}_eval.json"), result
            )
            _append_eval_row(csv_path, iteration, checkpoint, evaluated_at, result)
            evaluated.add(iteration)
            latest = result
            elapsed = time.monotonic() - started
            print(
                f"[eval] it {iteration} ({checkpoint['wall_time'] / 3600:.2f} h): "
                f"norm return {result['norm_return']:.4f}, "
                f"failed {result['n_failed']}/{len(result['per_clip'])} ({elapsed:.0f}s)",
                flush=True,
            )
            if (
                success_threshold is not None
                and result["norm_return"] >= success_threshold
                and result["n_failed"] == 0
            ):
                return result
        if once:
            return latest
        time.sleep(poll_seconds)


def make_eval_env(
    run_dir: Path,
    backend: str = "mjwarp",
    *,
    num_envs: int | None = None,
    device: str | None = None,
    motion_files: list[str] | None = None,
    src_fps: float | None = None,
    recovery_mode: bool = False,
) -> SuperMimicManagerBasedRlEnv:
    """Build a deterministic environment with a bounded tracking evaluation pool."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    from unilab.base.config_adapter import create_env
    from unilab.base.process_device import configure_backend_process_device
    from unilab.training import ensure_registries
    from unilab.utils.device import get_default_device

    import unisupermimic
    from unisupermimic.scripts.train_supermimic import _motion_count, build_env_override
    from unisupermimic.sim2sim import resolve_sim2sim_config

    config_dir = Path(unisupermimic.__file__).resolve().parent / "conf" / "supermimic"
    overrides = ["training.play_only=true", f"task=g1_super_mimic/{backend}"]
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    # Restore the run's policy and reference configuration, retaining the selected backend owner.
    recorded_path = run_dir / "run_config.json"
    if recorded_path.is_file():
        recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
        source = recorded.get("config", {})
        with open_dict(cfg):
            if "algo" in source:
                cfg.algo = OmegaConf.merge(cfg.algo, source["algo"])
            motion = source.get("env", {}).get("commands", {}).get("motion")
            if motion is not None:
                cfg.env.commands.motion = OmegaConf.merge(cfg.env.commands.motion, motion)
            for path, value in recorded.get("contract_snapshot", {}).items():
                if path.startswith(("algo.", "env.commands.motion.")):
                    OmegaConf.update(cfg, path, value, merge=False, force_add=True)
    if motion_files is not None:
        if not motion_files:
            raise ValueError("evaluation motion_files must not be empty")
        cfg.env.commands.motion.motion_files = motion_files
    if src_fps is not None:
        if not math.isfinite(src_fps) or src_fps <= 0:
            raise ValueError("evaluation src_fps must be finite and positive")
        cfg.env.commands.motion.src_fps = src_fps
    ensure_registries()
    resolve_sim2sim_config(
        run_dir,
        cfg,
        algo_name="supermimic",
        strict=bool(cfg.training.sim2sim_strict),
        allow_motion_override=motion_files is not None,
    )
    device = get_default_device() if device is None else device
    bound_device = configure_backend_process_device(str(cfg.training.sim_backend), device)
    if bound_device is not None:
        device = bound_device
    override = build_env_override(cfg, eval_mode=True, learner_device=device)
    if recovery_mode:
        # Evaluation task override only; actor I/O and motion contract were checked above.
        recovery = dict(override["commands"]["motion"].get("recovery") or {})
        recovery.update(enabled=True, assistance_max=0.0)
        override["commands"]["motion"]["recovery"] = recovery
    num_clips = _motion_count(cfg)
    count = min(num_clips, DEFAULT_EVAL_NUM_ENVS if num_envs is None else num_envs)
    if recovery_mode and num_envs is not None:
        count = num_envs
    return cast(
        "SuperMimicManagerBasedRlEnv",
        create_env(
            cfg,
            num_envs=count,
            env_cfg_override=override,
        ),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument(
        "--backend", choices=SUPPORTED_BACKENDS, default="mjwarp", help="Simulation backend"
    )
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        type=int,
        default=DEFAULT_EVAL_NUM_ENVS,
        help="Maximum concurrent tracking environments",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--success_threshold", type=float, default=0.82)
    parser.add_argument("--suite", choices=("tracking", "recovery"), default="tracking")
    parser.add_argument(
        "--trials", type=int, default=32, help="Parallel unassisted recovery trials"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motions", type=Path, help="Grouped motion split JSON manifest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--require-unseen", action="store_true")
    parser.add_argument("--reference-noise", type=float, default=0.0)
    parser.add_argument(
        "--push-force", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("FX", "FY", "FZ")
    )
    parser.add_argument("--push-at", type=float, default=1.0)
    parser.add_argument("--push-duration", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    from unisupermimic.sim2sim import policy_load_dim_guard

    from .datasets import load_motion_split

    args = _argument_parser().parse_args(argv)
    if args.ckpt is None and args.run_dir is None:
        _argument_parser().error("one of --run_dir or --ckpt is required")
    custom_evaluation = (
        args.suite != "tracking"
        or args.motions is not None
        or args.reference_noise
        or any(args.push_force)
        or args.output is not None
        or args.require_unseen
    )
    if (args.watch and custom_evaluation) or (custom_evaluation and args.ckpt is None):
        _argument_parser().error("custom evaluation requires --ckpt and cannot use --watch")
    if args.trials < 1 or (args.require_unseen and args.motions is None):
        _argument_parser().error("trials must be positive; --require-unseen requires --motions")
    if args.suite == "recovery" and (args.reference_noise or any(args.push_force)):
        _argument_parser().error(
            "recovery trials and tracking perturbations are separate protocols"
        )
    run_dir = args.ckpt.parent if args.ckpt is not None else args.run_dir
    files, dataset = (
        load_motion_split(args.motions, args.split) if args.motions is not None else (None, None)
    )
    if args.require_unseen:
        training_path = run_dir / "motion_dataset.json"
        if not training_path.is_file():
            raise ValueError(
                "cannot certify unseen motions: training motion_dataset.json is missing"
            )
        training = json.loads(training_path.read_text(encoding="utf-8"))["motions"]
        if any(str(item["path"]).endswith(".pt") for item in training) or any(
            str(path).endswith(".pt") for path in (files or [])
        ):
            raise ValueError("compact libraries lack per-clip source groups for --require-unseen")
        if not training or any("source_group" not in item for item in training):
            raise ValueError(
                "cannot certify unseen source groups: train with a grouped motion manifest first"
            )
        seen_hashes = {item["sha256"] for item in training}
        seen_groups = {item["source_group"] for item in training if "source_group" in item}
        assert dataset is not None
        if any(
            item["sha256"] in seen_hashes or item["source_group"] in seen_groups
            for item in dataset["motions"]
        ):
            raise ValueError(
                "requested unseen split overlaps training motion content or source groups"
            )
    if args.num_envs < 1:
        _argument_parser().error("--num-envs must be positive")
    env_kwargs: dict[str, Any] = {"num_envs": args.num_envs}
    if files is not None:
        env_kwargs["motion_files"] = files
    if args.suite == "recovery":
        env_kwargs.update(recovery_mode=True, num_envs=args.trials)
    env = make_eval_env(run_dir, backend=args.backend, **env_kwargs)
    try:
        if args.ckpt is not None:
            device = torch.device(env.device)
            checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
            with policy_load_dim_guard(
                env_obs_dim=env.supermimic_mdp.obs_dim,
                env_action_dim=env.action_manager.total_action_dim,
                algo_name="supermimic",
            ):
                if args.suite == "recovery":
                    result = evaluate_recovery(env, checkpoint, seed=args.seed)
                else:
                    result = evaluate(
                        env,
                        checkpoint,
                        device=device,
                        reference_noise=args.reference_noise,
                        push_force=tuple(args.push_force),
                        push_at=args.push_at,
                        push_duration=args.push_duration,
                        seed=args.seed,
                    )
            if dataset is not None:
                result["dataset"] = dataset
                result["unseen_verified"] = bool(args.require_unseen)
            destination = args.output or args.ckpt.with_name(
                f"{args.ckpt.stem}_{args.suite}_eval.json"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(destination, result)
            print(
                json.dumps(
                    {key: value for key, value in result.items() if key != "per_clip"}, indent=1
                )
            )
            return
        watch(
            env,
            args.run_dir,
            every=args.every,
            poll_seconds=args.poll_seconds,
            success_threshold=args.success_threshold,
            once=not args.watch,
        )
    finally:
        env.close()
        cast(Any, env.backend).close()


__all__ = ["EVAL_CSV_FIELDS", "evaluate", "load_policy", "main", "make_eval_env", "watch"]

if __name__ == "__main__":
    main()
