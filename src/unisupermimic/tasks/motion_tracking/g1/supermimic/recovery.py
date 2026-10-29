"""Recovery curriculum parameters shared by the simulator and analytic MDP."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = False
    probability: float = 0.15
    seconds: float = 3.0
    stand_height: float = 0.76
    assistance_max: float = 200.0
    curriculum_iterations: int = 6000
    initial_angle: float = 0.4
    final_angle: float = math.pi
    height_tolerance: float = 0.12
    tilt_tolerance: float = 0.3
    joint_tolerance: float = 0.5
    velocity_tolerance: float = 0.5
    stable_seconds: float = 0.3
    shaping_weight: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.probability <= 1:
            raise ValueError("recovery probability must be in [0, 1]")
        for name in (
            "seconds",
            "stand_height",
            "height_tolerance",
            "tilt_tolerance",
            "joint_tolerance",
            "velocity_tolerance",
            "stable_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"recovery {name} must be finite and positive")
        if not 0 <= self.initial_angle <= self.final_angle <= math.pi:
            raise ValueError("recovery angles must satisfy 0 <= initial <= final <= pi")
        if self.curriculum_iterations < 1 or self.stable_seconds > self.seconds:
            raise ValueError("invalid recovery curriculum or stable duration")
        if not math.isfinite(self.assistance_max) or self.assistance_max < 0:
            raise ValueError("recovery assistance_max must be finite and nonnegative")
        if not math.isfinite(self.shaping_weight) or self.shaping_weight < 0:
            raise ValueError("recovery shaping_weight must be finite and nonnegative")

    def progress(self, iteration: int) -> float:
        return min(max(iteration / self.curriculum_iterations, 0.0), 1.0)
