"""World model, goal-conditioned policy, and critic for SuperMimic."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from unisupermimic.rl.contracts.supermimic import State

from .geometry import invariant_features, so3_exp, so3_log, yaw_mat


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    act: type[nn.Module] = nn.ELU,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for hidden_dim in hidden:
        layers.append(nn.Linear(dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean and variance, frozen once ``max_count`` is reached."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.mean: torch.Tensor
        self.var: torch.Tensor
        self.count: torch.Tensor
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = max_count
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        batch_count = x.shape[0]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / total
        ) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)


AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run an MLP under optional autocast, returning fp32 for state arithmetic."""
    if AMP_DTYPE is None:
        return net(x)
    device_type = x.device.type
    if device_type not in ("cuda", "mps"):
        return net(x)
    with torch.autocast(device_type, dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Predict frame-local state increments and integrate them in fp32."""

    def __init__(
        self,
        hidden: Sequence[int] = (2048, 2048, 2048),
        num_joints: int = 29,
        context_dim: int = 33,
        force_dim: int = 0,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.context_dim = context_dim
        if force_dim not in (0, 3):
            raise ValueError("force_dim must be 0 or 3 (world-frame force at pelvis COM)")
        self.force_dim = force_dim
        self.in_dim = 13 + 3 * num_joints + context_dim + force_dim
        self.out_dim = 12 + 2 * num_joints
        self.net = mlp(self.in_dim, hidden, self.out_dim, act=nn.SiLU, layer_norm=True)
        output = self.net[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)

    def inputs(
        self,
        state: State,
        action: torch.Tensor,
        normalized_context: torch.Tensor,
        force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = [invariant_features(state), action, normalized_context]
        if self.force_dim:
            if force is None:
                raise ValueError("force-conditioned world model requires recorded force")
            features.append((state[1].transpose(-1, -2) @ force.unsqueeze(-1)).squeeze(-1) / 200)
        return torch.cat(features, -1)

    @staticmethod
    def targets(state: State, next_state: State) -> torch.Tensor:
        """Return frame-local increments from ``state`` to ``next_state``."""
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        next_position, next_rotation, next_lin_vel, next_ang_vel, next_joint_pos, next_joint_vel = (
            next_state
        )
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                (heading.transpose(-1, -2) @ (next_position - position).unsqueeze(-1)).squeeze(-1),
                so3_log(rotation_t @ next_rotation),
                (rotation_t @ (next_lin_vel - lin_vel).unsqueeze(-1)).squeeze(-1),
                (rotation_t @ (next_ang_vel - ang_vel).unsqueeze(-1)).squeeze(-1),
                next_joint_pos - joint_pos,
                next_joint_vel - joint_vel,
            ),
            -1,
        )

    def step(
        self,
        state: State,
        action: torch.Tensor,
        normalized_context: torch.Tensor,
        force: torch.Tensor | None = None,
    ) -> State:
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        prediction = run_mlp(
            self.net,
            self.in_norm(self.inputs(state, action, normalized_context, force)),
        )
        increment = prediction * self.out_norm.std() + self.out_norm.mean
        heading = yaw_mat(rotation)
        d_position, d_rotation, d_lin_vel, d_ang_vel, d_joint_pos, d_joint_vel = increment.split(
            (3, 3, 3, 3, self.num_joints, self.num_joints), -1
        )
        next_position = position + (heading @ d_position.unsqueeze(-1)).squeeze(-1)
        next_rotation = rotation @ so3_exp(d_rotation)
        next_lin_vel = lin_vel + (rotation @ d_lin_vel.unsqueeze(-1)).squeeze(-1)
        next_ang_vel = ang_vel + (rotation @ d_ang_vel.unsqueeze(-1)).squeeze(-1)
        return (
            next_position,
            next_rotation,
            next_lin_vel,
            next_ang_vel,
            joint_pos + d_joint_pos,
            joint_vel + d_joint_vel,
        )

    @staticmethod
    def state_error(predicted_state: State, state: State) -> torch.Tensor:
        """Return frame-local errors in the same layout as ``targets``."""
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        (
            predicted_position,
            predicted_rotation,
            predicted_lin_vel,
            predicted_ang_vel,
            predicted_joint_pos,
            predicted_joint_vel,
        ) = predicted_state
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                (heading.transpose(-1, -2) @ (predicted_position - position).unsqueeze(-1)).squeeze(
                    -1
                ),
                so3_log(rotation_t @ predicted_rotation),
                (rotation_t @ (predicted_lin_vel - lin_vel).unsqueeze(-1)).squeeze(-1),
                (rotation_t @ (predicted_ang_vel - ang_vel).unsqueeze(-1)).squeeze(-1),
                predicted_joint_pos - joint_pos,
                predicted_joint_vel - joint_vel,
            ),
            -1,
        )


class Policy(nn.Module):
    """Encode a goal and decode it together with an observation history."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hist: int = 10,
        z_dim: int = 32,
        num_joints: int = 29,
        hidden: Sequence[int] = (2048, 1024, 512, 256),
    ):
        super().__init__()
        self.hist = hist
        self.enc = mlp(goal_dim, hidden, z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, hidden, num_joints)
        output = self.dec[-1]
        assert isinstance(output, nn.Linear)
        with torch.no_grad():
            output.weight.mul_(0.01)
            output.bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Consume ``(batch, hist, obs_dim)`` observations, newest last."""
        latent = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        obs = self.obs_norm(obs_hist).flatten(-2)
        return run_mlp(self.dec, torch.cat((latent, obs), -1))

    def update_goal_norm(self, goal: torch.Tensor) -> None:
        self.goal_norm.update(goal)


def positional_encoding(length: int, width: int, offset: int = 0) -> torch.Tensor:
    positions = torch.arange(offset, offset + length).float().unsqueeze(-1)
    frequencies = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000) / width))
    encoding = torch.zeros(length, width)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: width // 2])
    return encoding


class AttentionBlock(nn.Module):
    """Explicit attention keeps the input-gradient path identical in train and eval."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query_norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = mlp(width, (4 * width,), width, act=nn.GELU)

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.query_norm(query)
        source = normalized if tokens is None else tokens
        q, k, v = self.q(normalized), self.k(source), self.v(source)
        batch, length, width = q.shape
        q = q.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
        k = k.view(batch, -1, self.heads, width // self.heads).transpose(1, 2)
        v = v.view(batch, -1, self.heads, width // self.heads).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(width // self.heads)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, width)
        result = query + self.out(attended)
        return result + self.ff(self.ff_norm(result))


class RGMTPolicy(nn.Module):
    """History-conditioned command aggregation trained through the world model."""

    def __init__(
        self,
        obs_dim: int,
        goal_frame_dim: int,
        goal_frames: int,
        hist: int = 10,
        num_joints: int = 29,
        hidden: Sequence[int] = (2048, 1024, 512, 256),
        width: int = 128,
        heads: int = 4,
        layers: int = 1,
        reference_past: int = 10,
        proprio_dim: int = 93,
    ):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads or layers < 1 or hist < 1:
            raise ValueError("invalid RGMT width, heads, layers, or history length")
        if not 0 < proprio_dim <= obs_dim or not 0 <= reference_past < goal_frames:
            raise ValueError("invalid RGMT proprio_dim or reference window")
        self.hist, self.goal_frames = hist, goal_frames
        self.goal_frame_dim, self.proprio_dim = goal_frame_dim, proprio_dim
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_frame_dim)
        self.history_enc = mlp(proprio_dim, (width,), width)
        self.history_blocks = nn.ModuleList([AttentionBlock(width, heads) for _ in range(layers)])
        self.history_norm = nn.LayerNorm(width)
        self.query = mlp(width, (width,), width)
        self.command_enc = mlp(goal_frame_dim, (width,), width)
        self.cross = AttentionBlock(width, heads)
        self.command_norm = nn.LayerNorm(width)
        self.dec = mlp(width + obs_dim, hidden, num_joints)
        self.register_buffer("history_positions", positional_encoding(hist, width))
        self.register_buffer(
            "command_positions", positional_encoding(goal_frames, width, -reference_past)
        )
        self.register_buffer("causal_mask", torch.ones(hist, hist, dtype=torch.bool).triu(1))
        output = self.dec[-1]
        assert isinstance(output, nn.Linear)
        with torch.no_grad():
            output.weight.mul_(0.01)
            output.bias.zero_()

    def encode_history(self, normalized_obs: torch.Tensor) -> torch.Tensor:
        tokens = self.history_enc(normalized_obs[..., : self.proprio_dim]) + self.history_positions
        for block in self.history_blocks:
            tokens = block(tokens, mask=self.causal_mask)
        return self.history_norm(tokens)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # FP32 attention avoids inference fast paths and remains differentiable on CPU/CUDA.
        obs = self.obs_norm(obs_hist)
        dynamics = self.encode_history(obs).amax(dim=1)
        reference = goal.reshape(-1, self.goal_frames, self.goal_frame_dim)
        tokens = self.command_enc(self.goal_norm(reference)) + self.command_positions
        command = self.command_norm(self.cross(self.query(dynamics).unsqueeze(1), tokens))[:, 0]
        return run_mlp(self.dec, torch.cat((command, obs[:, -1]), -1))

    def update_goal_norm(self, goal: torch.Tensor) -> None:
        self.goal_norm.update(goal.reshape(-1, self.goal_frames, self.goal_frame_dim))


def make_policy(
    obs_dim: int,
    goal_frame_dim: int,
    num_joints: int,
    config: Mapping[str, Any],
) -> Policy | RGMTPolicy:
    """Shared construction for training, evaluation, rendering and old checkpoints."""
    if config.get("policy_io_version", 1) != 1:
        raise ValueError("unsupported policy_io_version")
    hidden = tuple(
        config.get("test_hidden_dims") or config.get("policy_hidden", (2048, 1024, 512, 256))
    )
    hist, frames = int(config.get("hist", 10)), int(config.get("goal_frames", 10))
    kind = config.get("policy_type", "mlp")
    if kind == "mlp":
        return Policy(
            obs_dim,
            goal_frame_dim * frames,
            hist=hist,
            z_dim=int(config.get("z_dim", 32)),
            num_joints=num_joints,
            hidden=hidden,
        )
    if kind == "rgmt":
        return RGMTPolicy(
            obs_dim,
            goal_frame_dim,
            frames,
            hist=hist,
            num_joints=num_joints,
            hidden=hidden,
            width=int(config.get("attention_dim", 128)),
            heads=int(config.get("attention_heads", 4)),
            layers=int(config.get("attention_layers", 1)),
            reference_past=int(config.get("reference_past", 10)),
            proprio_dim=int(config.get("proprio_dim", 93)),
        )
    raise ValueError(f"unknown policy_type {kind!r}")


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int] = (256, 128)):
        super().__init__()
        self.net = mlp(in_dim, hidden, 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, self.norm(x)).squeeze(-1)


__all__ = [
    "AMP_DTYPE",
    "Critic",
    "Policy",
    "RGMTPolicy",
    "RunningNorm",
    "WorldModel",
    "make_policy",
    "run_mlp",
]
