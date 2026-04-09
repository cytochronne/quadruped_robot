from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    # Waypoint helpers (scalar speed that can be curriculum-controlled)
    waypoint_speed_range: tuple[float, float] = (0.5, 0.5)
    waypoint_speed_limit: float | None = None
    waypoint_speed_step: float = 0.1
