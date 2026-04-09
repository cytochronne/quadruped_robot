from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = 0.1,
    lin_vel_scale: float = 0.0,
    ang_vel_scale: float = 0.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    penalty = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    standing_mask = (cmd_norm < float(cmd_threshold)).to(dtype=penalty.dtype)

    if lin_vel_scale != 0.0:
        lin_vel_b = asset.data.root_lin_vel_b
        penalty = penalty + float(lin_vel_scale) * torch.sum(torch.square(lin_vel_b), dim=1)

    if ang_vel_scale != 0.0:
        ang_vel_b = getattr(asset.data, "root_ang_vel_b", None)
        if ang_vel_b is not None:
            penalty = penalty + float(ang_vel_scale) * torch.sum(torch.square(ang_vel_b), dim=1)

    return penalty * standing_mask


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def standstill_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    vel_threshold: float = 0.1,
    cmd_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize staying almost static when non-zero velocity commands are issued."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    slow_penalty = torch.clamp(vel_threshold - body_lin_vel, min=0.0)
    return slow_penalty * (cmd_norm > cmd_threshold).float()


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


def feet_step_frequency_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    touchdown_time_threshold_s: float | None = None,
    command_name: str | None = "base_velocity",
    min_cmd_norm: float = 0.1,
) -> torch.Tensor:
    """Penalize high step frequency by counting touchdown events.

    We approximate a "touchdown" as a foot that is in contact and whose
    ``current_contact_time`` is very small (i.e., it just touched down this step).

    This provides a dense, differentiable-by-policy (though event-like) signal
    that discourages frequent contacts and thus encourages longer step periods
    (lower step frequency) at a given commanded speed.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids

    # Threshold default: a little larger than one control step.
    if touchdown_time_threshold_s is None:
        touchdown_time_threshold_s = float(getattr(env, "step_dt", 0.0)) * 1.5
        # Fallback if env.step_dt is missing for some reason.
        if touchdown_time_threshold_s <= 0.0:
            touchdown_time_threshold_s = 0.03

    ctime = contact_sensor.data.current_contact_time[:, body_ids]
    touchdown = (ctime > 0.0) & (ctime < float(touchdown_time_threshold_s))

    # Normalize by number of feet so scale is stable across robots.
    penalty = torch.sum(touchdown, dim=-1).to(dtype=torch.float32)
    penalty = penalty / max(1, len(body_ids))

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        penalty = penalty * (cmd_norm >= float(min_cmd_norm)).to(dtype=torch.float32)

    return penalty


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


def vel_tracking_success(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    lin_thresh: float = 0.1,
) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    cmd_b = env.command_manager.get_command(command_name)
    vel_b = asset.data.root_lin_vel_b[:, :2]
    err = torch.linalg.norm(vel_b - cmd_b[:, :2], dim=1)
    return (err < lin_thresh).float()


def lin_vel_xy_tracking_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize XY linear-velocity tracking error with an L2-squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_b = env.command_manager.get_command(command_name)
    vel_b = asset.data.root_lin_vel_b[:, :2]
    return torch.sum(torch.square(cmd_b[:, :2] - vel_b), dim=1)


def termination_failure_flag(env, include_time_outs: bool = False) -> torch.Tensor:
    """Returns 1.0 for envs that terminated this step (optionally excluding time-outs).

    This is intended to be used as a *one-step* penalty signal at termination.
    Kept mild to avoid dominating dense tracking rewards.
    """

    term_mgr = getattr(env, "termination_manager", None)
    terminated = getattr(term_mgr, "terminated", None) if term_mgr is not None else None
    if terminated is None:
        # Fallback to a safe no-op if API differs.
        return torch.zeros(getattr(env, "num_envs", 1), device=getattr(env, "device", None))

    terminated = terminated.to(dtype=torch.bool)

    if not include_time_outs:
        # Different Isaac Lab versions use slightly different attribute names.
        for time_out_attr in ("time_outs", "timeouts", "timed_out", "time_out"):
            time_outs = getattr(term_mgr, time_out_attr, None)
            if time_outs is not None:
                terminated = terminated & (~time_outs.to(dtype=torch.bool))
                break

    return terminated.to(dtype=torch.float32)


def action_smoothness_l2(env) -> torch.Tensor:
    """Penalize action second-order difference.

    Computes per-environment:
        ||a_t - 2 a_{t-1} + a_{t-2}||_2^2

    The history buffers are cached on the env instance and are reset to the
    current action for envs at the start of an episode (episode_length_buf == 0),
    making the penalty zero at episode start.
    """

    num_envs = int(getattr(env, "num_envs", 1))
    device = getattr(env, "device", None)

    action_mgr = getattr(env, "action_manager", None)
    if action_mgr is None:
        return torch.zeros(num_envs, device=device)

    cur_action = getattr(action_mgr, "action", None)
    if cur_action is None:
        cur_action = getattr(action_mgr, "_action", None)
    if cur_action is None:
        return torch.zeros(num_envs, device=device)

    # Ensure shape is (num_envs, action_dim)
    if cur_action.ndim == 1:
        cur_action = cur_action.unsqueeze(0)

    # Initialize cached buffers if missing or shape mismatch.
    if (
        not hasattr(env, "_action_smooth_prev")
        or not hasattr(env, "_action_smooth_prev2")
        or getattr(env, "_action_smooth_prev", None) is None
        or getattr(env, "_action_smooth_prev2", None) is None
        or env._action_smooth_prev.shape != cur_action.shape
        or env._action_smooth_prev2.shape != cur_action.shape
    ):
        env._action_smooth_prev = cur_action.clone()
        env._action_smooth_prev2 = cur_action.clone()
        return torch.zeros(cur_action.shape[0], device=cur_action.device, dtype=torch.float32)

    # Reset per-env history at episode start so penalty starts from zero.
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is not None:
        reset_mask = episode_length_buf == 0
        if torch.any(reset_mask):
            env._action_smooth_prev[reset_mask] = cur_action[reset_mask]
            env._action_smooth_prev2[reset_mask] = cur_action[reset_mask]

    prev = env._action_smooth_prev
    prev2 = env._action_smooth_prev2
    diff2 = cur_action - 2.0 * prev + prev2
    penalty = torch.sum(torch.square(diff2), dim=-1).to(dtype=torch.float32)

    # Shift history forward (in-place to avoid allocations)
    env._action_smooth_prev2.copy_(env._action_smooth_prev)
    env._action_smooth_prev.copy_(cur_action)

    return penalty
