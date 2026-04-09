from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_terrain_bounds_xy(env: "ManagerBasedRLEnv", edge_margin: float) -> tuple[float, float, float, float]:
    """Infer global terrain XY bounds.

    Prefer geometry-based bounds from terrain-generator metadata to avoid
    directional bias from terrain origin conventions.
    """
    terrain = getattr(env.scene, "terrain", None)

    terrain_cfg = getattr(getattr(getattr(env, "cfg", None), "scene", None), "terrain", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)

    tile_size = getattr(terrain_generator, "size", None)
    num_rows = getattr(terrain_generator, "num_rows", None)
    num_cols = getattr(terrain_generator, "num_cols", None)
    border_width = float(getattr(terrain_generator, "border_width", 0.0) or 0.0)
    margin = max(float(edge_margin), 0.0)

    if tile_size is not None and num_rows is not None and num_cols is not None:
        tile_x = float(tile_size[0])
        tile_y = float(tile_size[1])
        span_x = max(float(num_rows) * tile_x + 2.0 * border_width, tile_x)
        span_y = max(float(num_cols) * tile_y + 2.0 * border_width, tile_y)

        # TerrainImporter places generated terrain around world origin.
        center_x = 0.0
        center_y = 0.0

        half_x = max(0.5 * span_x - margin, 1e-6)
        half_y = max(0.5 * span_y - margin, 1e-6)
        return center_x - half_x, center_x + half_x, center_y - half_y, center_y + half_y

    # Fallback: infer from terrain/env origins envelope.
    terrain_origins = getattr(terrain, "terrain_origins", None)
    if terrain_origins is not None:
        # Supports (rows, cols, 3) and (num, 3) layouts.
        origins_xy = terrain_origins[..., :2].reshape(-1, 2)
    else:
        origins_xy = env.scene.env_origins[:, :2]

    x_min = float(origins_xy[:, 0].min().item())
    x_max = float(origins_xy[:, 0].max().item())
    y_min = float(origins_xy[:, 1].min().item())
    y_max = float(origins_xy[:, 1].max().item())

    bounds_x_min = x_min + margin
    bounds_x_max = x_max - margin
    bounds_y_min = y_min + margin
    bounds_y_max = y_max - margin
    if bounds_x_max <= bounds_x_min:
        c_x = 0.5 * (bounds_x_min + bounds_x_max)
        bounds_x_min, bounds_x_max = c_x - 1e-6, c_x + 1e-6
    if bounds_y_max <= bounds_y_min:
        c_y = 0.5 * (bounds_y_min + bounds_y_max)
        bounds_y_min, bounds_y_max = c_y - 1e-6, c_y + 1e-6
    return bounds_x_min, bounds_x_max, bounds_y_min, bounds_y_max


def position_out_of_terrain_bounds(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    edge_margin: float = 0.05,
) -> torch.Tensor:
    """Return bool mask for robots outside the global terrain XY bounds."""
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    root_xy = asset.data.root_pos_w[:, :2]

    x_min, x_max, y_min, y_max = _get_terrain_bounds_xy(env, edge_margin=edge_margin)

    out_x = (root_xy[:, 0] < x_min) | (root_xy[:, 0] > x_max)
    out_y = (root_xy[:, 1] < y_min) | (root_xy[:, 1] > y_max)

    out_of_bounds = out_x | out_y
    # if bool(out_of_bounds.any().item()):
    #     print("Robot out of bounds")
    return out_of_bounds
