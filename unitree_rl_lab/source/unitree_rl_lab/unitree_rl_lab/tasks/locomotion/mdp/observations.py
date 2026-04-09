from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def rigid_body_material_friction(
    env: ManagerBasedRLEnv, asset_cfg: "SceneEntityCfg" = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return per-env static and dynamic friction coefficients of the asset's physics material.

    The values are computed by averaging over all shapes of the asset in each environment.
    Output shape is (num_envs, 2): [static_friction, dynamic_friction].
    """
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]

    # PhysX material properties are retrieved on CPU; convert to tensor on asset.device.
    # Shape: (num_envs, max_num_shapes, 3) with [static, dynamic, restitution].
    materials = asset.root_physx_view.get_material_properties()
    if not isinstance(materials, torch.Tensor):
        materials = torch.tensor(materials, device=asset.device)
    else:
        materials = materials.to(asset.device)

    mean_props = materials.mean(dim=1)
    return mean_props[:, :2]


def external_force_torque(
    env: ManagerBasedRLEnv, asset_cfg: "SceneEntityCfg" = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return the external force and torque buffers applied to the asset's bodies.

    The output concatenates force and torque in the asset's body frame for the selected bodies.
    Shape: (num_envs, num_bodies * 6).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    forces = asset._external_force_b[:, asset_cfg.body_ids]
    torques = asset._external_torque_b[:, asset_cfg.body_ids]
    wrench = torch.cat([forces, torques], dim=-1)
    return wrench.view(env.num_envs, -1)
