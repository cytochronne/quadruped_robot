# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL Base."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL Base.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--resume_path", type=str, default=None, help="Path to the model checkpoint to resume training from.")
parser.add_argument("--track_waypoints", action="store_true", default=False, help="Enable waypoint tracking mode.")
parser.add_argument(
    "--waypoint_ratio",
    type=float,
    default=1.0,
    help=(
        "Fraction of environments that follow waypoint targets when "
        "--track_waypoints is enabled (0.0-1.0). The rest use random commands."
    ),
)
parser.add_argument(
    "--rl_algorithm",
    type=str,
    default="auto",
    choices=["auto", "td3", "sac"],
    help=(
        "RL algorithm selector. "
        "'auto' keeps existing RL Base behavior, "
        "'td3' and 'sac' run local off-policy implementations in rl_base."
    ),
)
parser.add_argument(
    "--offpolicy_total_timesteps",
    type=int,
    default=1_000_000,
    help="Total environment timesteps for TD3/SAC training.",
)
parser.add_argument(
    "--offpolicy_learning_rate",
    type=float,
    default=3.0e-4,
    help="Learning rate for TD3/SAC.",
)
parser.add_argument(
    "--offpolicy_buffer_size",
    type=int,
    default=1_000_000,
    help="Replay buffer size for TD3/SAC.",
)
parser.add_argument(
    "--offpolicy_learning_starts",
    type=int,
    default=10000,
    help="Number of warmup steps before TD3/SAC updates start.",
)
parser.add_argument(
    "--offpolicy_batch_size",
    type=int,
    default=256,
    help="Batch size for TD3/SAC updates.",
)
parser.add_argument(
    "--offpolicy_train_freq",
    type=int,
    default=1,
    help="Training frequency (steps) for TD3/SAC.",
)
parser.add_argument(
    "--offpolicy_gradient_steps",
    type=int,
    default=1,
    help="Gradient steps per update for TD3/SAC.",
)
parser.add_argument(
    "--offpolicy_tau",
    type=float,
    default=0.005,
    help="Polyak averaging coefficient for TD3/SAC target networks.",
)
parser.add_argument(
    "--offpolicy_gamma",
    type=float,
    default=0.99,
    help="Discount factor for TD3/SAC.",
)
parser.add_argument(
    "--offpolicy_log_interval",
    type=int,
    default=10,
    help="Logging interval used by local TD3/SAC learn().",
)
parser.add_argument(
    "--offpolicy_save_interval",
    type=int,
    default=5000,
    help="Checkpoint save interval (in off-policy segments) for local TD3/SAC.",
)



# append RL Base cli arguments
cli_args.add_rl_base_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

import os as _os
if getattr(args_cli, "device", None) and isinstance(args_cli.device, str) \
        and args_cli.device.startswith("cuda:") and not args_cli.distributed:
    try:
        _idx = int(args_cli.device.split(":")[1])
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(_idx)
        # 可选：缓解碎片化
        _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # 进程内改用 cuda:0（此时的 0 对应物理 GPU N）
        args_cli.device = "cuda:0"
        print(f"[INFO] Remapped requested device to single visible GPU: "
              f"CUDA_VISIBLE_DEVICES={_idx}, internal --device=cuda:0")
    except Exception as _e:
        print(f"[WARN] Failed to remap --device '{args_cli.device}': {_e}")


# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RL Base version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rl-base version
RL_BASE_VERSION = "2.3.1"
installed_version = metadata.version("rl-base-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RL_BASE_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rl-base-lib=={RL_BASE_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rl-base-lib=={RL_BASE_VERSION}"]
    print(
        f"Please install the correct version of RL Base.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RL_BASE_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import numpy as np
import os
import shutil
import torch
from datetime import datetime

from rl_base.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from rl_base.isaaclab_support import RlBaseOnPolicyRunnerCfg, RlBaseVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

class WaypointWrapper(gym.Wrapper):
    def __init__(self, env, waypoint_manager):
        super().__init__(env)
        self.waypoint_manager = waypoint_manager
        self.command_manager = self.unwrapped.command_manager

    def step(self, action):
        # Compute waypoints before physics step
        with torch.no_grad():
            robot_pos = self.unwrapped.scene["robot"].data.root_pos_w
            # Get yaw
            try:
                from isaaclab.utils.math import euler_xyz_from_quat
                quat = self.unwrapped.scene["robot"].data.root_quat_w
                _, _, yaw = euler_xyz_from_quat(quat)
            except:
                heading = getattr(self.unwrapped.scene["robot"].data, "heading_w", None)
                if heading is not None:
                     yaw = heading.squeeze(-1)
                else:
                     yaw = torch.zeros(self.unwrapped.num_envs, device=self.unwrapped.device)

            cmd_vel, _ = self.waypoint_manager.compute_command(robot_pos, yaw)
            
            # Override command
            term = self.command_manager.get_term("base_velocity")
            if hasattr(term, "vel_command_b"):
                # Only override a subset of environments according to the
                # waypoint manager's mask. The remaining envs keep using
                # the original random commands from ranges.
                use_wp = getattr(self.waypoint_manager, "use_waypoint", None)
                if isinstance(use_wp, torch.Tensor) and use_wp.dtype == torch.bool:
                    if use_wp.any():
                        #print("INFO: use_wp:",use_wp)
                        term.vel_command_b[use_wp] = cmd_vel[use_wp]
                else:
                    # Fallback to previous behavior (all envs use waypoints)
                    term.vel_command_b[:] = cmd_vel


        # Step
        ret = self.env.step(action)
        
        # Reset logic for waypoints
        # Gym API: obs, rew, terminated, truncated, info
        obs, rew, terminated, truncated, info = ret
        
        dones = terminated | truncated
        if dones.any():
            reset_ids = torch.nonzero(dones).flatten()
            self.waypoint_manager.reset(reset_ids)
            
        return ret


def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, dict):
        return {k: _to_numpy(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_numpy(v) for v in data]
    if isinstance(data, tuple):
        return tuple(_to_numpy(v) for v in data)
    return data


def _squeeze_env_dim(data):
    if isinstance(data, np.ndarray):
        if data.ndim > 0 and data.shape[0] == 1:
            return data[0]
        return data
    if isinstance(data, dict):
        return {k: _squeeze_env_dim(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_squeeze_env_dim(v) for v in data]
    if isinstance(data, tuple):
        return tuple(_squeeze_env_dim(v) for v in data)
    return data


def _to_scalar(data, cast_type=float):
    value = _to_numpy(data)
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        if flat.size == 0:
            return cast_type(0)
        return cast_type(flat[0])
    return cast_type(value)


class OffPolicyVecEnvWrapper(gym.Wrapper):
    """Convert IsaacLab tensor-based vector env to local off-policy Gym API."""

    def __init__(self, env):
        super().__init__(env)
        num_envs = int(getattr(self.unwrapped, "num_envs", 1))
        if num_envs < 1:
            raise ValueError(f"Invalid num_envs={num_envs}. num_envs must be >= 1.")
        self.num_envs = num_envs
        self._sim_device = getattr(self.unwrapped, "device", "cpu")
        # Flatten Dict spaces (which also handles batch dimensions)
        obs_space = self._flatten_obs_space(self.env.observation_space, self.num_envs)
        # Strip batch dimension if still present (for non-Dict spaces)
        self.single_observation_space = self._strip_batch_dim(obs_space, self.num_envs)
        self.single_action_space = self._clip_action_bounds(self._strip_batch_dim(self.env.action_space, self.num_envs))
        # Keep common Gym naming while preserving compatibility with local algorithms.
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        if not isinstance(self.single_observation_space, gym.spaces.Box):
            raise TypeError(
                "TD3/SAC mode currently supports Box observation spaces only. "
                f"Got: {type(self.single_observation_space)}"
            )
        if not isinstance(self.single_action_space, gym.spaces.Box):
            raise TypeError(
                "TD3/SAC mode currently supports Box action spaces only. "
                f"Got: {type(self.single_action_space)}"
            )

    @staticmethod
    def _flatten_obs_space(space, num_envs=None):
        """Flatten a Dict observation space into a Box space."""
        if isinstance(space, gym.spaces.Dict):
            # Flatten Dict space to Box space
            low = []
            high = []
            for subspace in space.spaces.values():
                if isinstance(subspace, gym.spaces.Box):
                    # Strip batch dimension if present
                    if num_envs is not None and len(subspace.shape) > 0 and subspace.shape[0] == num_envs:
                        single_low = np.array(subspace.low[0], copy=True)
                        single_high = np.array(subspace.high[0], copy=True)
                    else:
                        single_low = np.array(subspace.low, copy=True)
                        single_high = np.array(subspace.high, copy=True)
                    low.extend(single_low.flatten())
                    high.extend(single_high.flatten())
                else:
                    raise TypeError(f"Unsupported space type inside Dict: {type(subspace)}")
            low = np.array(low, dtype=np.float32)
            high = np.array(high, dtype=np.float32)
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return space

    def _flatten_obs(self, obs):
        """Flatten a Dict observation into a flat array."""
        if isinstance(obs, dict):
            # Process each observation in the dict
            flattened_list = []
            for key in sorted(obs.keys()):  # Sort keys for consistent ordering
                value = obs[key]
                if isinstance(value, torch.Tensor):
                    value_np = value.cpu().numpy()
                else:
                    value_np = np.asarray(value)

                # Handle batch dimension
                if value_np.ndim > 1 and value_np.shape[0] == self.num_envs:
                    # Shape: (num_envs, ...) -> (num_envs, flat_obs_dim)
                    flat_obs = value_np.reshape(self.num_envs, -1)
                else:
                    # Single observation - flatten and add batch dimension
                    flat_obs = value_np.flatten()
                    flat_obs = np.expand_dims(flat_obs, axis=0).repeat(self.num_envs, axis=0)

                flattened_list.append(flat_obs)

            # Concatenate all observations along the feature dimension
            # Result shape: (num_envs, total_obs_dim)
            return np.concatenate(flattened_list, axis=1)
        return obs

    @staticmethod
    def _strip_batch_dim(space, num_envs: int):
        if isinstance(space, gym.spaces.Box) and len(space.shape) > 0 and space.shape[0] == num_envs:
            low = np.array(space.low[0], copy=True)
            high = np.array(space.high[0], copy=True)
            return gym.spaces.Box(low=low, high=high, dtype=space.dtype)
        return space

    @staticmethod
    def _clip_action_bounds(space):
        """Clip infinite action bounds to [-1, 1] for TD3/SAC compatibility."""
        if isinstance(space, gym.spaces.Box):
            low = np.array(space.low, copy=True)
            high = np.array(space.high, copy=True)
            # Replace -inf with -1.0 and inf with 1.0
            low = np.where(np.isinf(low) & (low < 0), -1.0, low)
            high = np.where(np.isinf(high) & (high > 0), 1.0, high)
            return gym.spaces.Box(low=low, high=high, dtype=space.dtype)
        return space

    def _format_action_for_env(self, action):
        action_np = np.asarray(action, dtype=np.float32)
        single_shape = self.single_action_space.shape
        batched_shape = (self.num_envs, *single_shape)
        if action_np.shape == single_shape:
            if self.num_envs != 1:
                raise ValueError(
                    f"Expected batched actions with shape {batched_shape} for num_envs={self.num_envs}, "
                    f"but got single action shape {single_shape}."
                )
            action_np = np.expand_dims(action_np, axis=0)
        elif action_np.shape != batched_shape:
            action_np = action_np.reshape(batched_shape)
        return torch.as_tensor(action_np, device=self._sim_device)

    def _to_batched_obs(self, obs):
        # First flatten Dict observations
        obs = self._flatten_obs(obs)
        obs = _to_numpy(obs)
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == len(self.single_observation_space.shape):
            obs_arr = np.expand_dims(obs_arr, axis=0)
        if np.issubdtype(obs_arr.dtype, np.floating):
            obs_arr = obs_arr.astype(np.float32, copy=False)
        return obs_arr

    def _to_batched_scalar(self, values, dtype):
        arr = np.asarray(_to_numpy(values))
        if arr.ndim == 0:
            arr = np.repeat(arr.reshape(1), self.num_envs)
        arr = arr.reshape(self.num_envs)
        return arr.astype(dtype, copy=False)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._to_batched_obs(obs)
        info = _to_numpy(info)
        return obs, info

    def step(self, action):
        env_action = self._format_action_for_env(action)

        # Check episode_length_buf BEFORE step to detect resets
        ep_len_before = None
        if hasattr(self.env.unwrapped, "episode_length_buf"):
            ep_len_before = self.env.unwrapped.episode_length_buf.clone() if isinstance(self.env.unwrapped.episode_length_buf, torch.Tensor) else self.env.unwrapped.episode_length_buf.copy()

        # Handle both old (4-value) and new (5-value) Gym APIs
        result = self.env.step(env_action)

        if len(result) == 4:
            # Old Gym API: (obs, reward, done, info)
            obs, reward, done, info = result
            terminated = done
            truncated = torch.zeros_like(done) if isinstance(done, torch.Tensor) else np.zeros_like(done, dtype=bool)
        elif len(result) == 5:
            # New Gym API: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            done = terminated | truncated
        else:
            raise ValueError(f"Unexpected number of return values from env.step(): {len(result)}")

        # Detect resets by comparing episode_length_buf before and after
        # If ep_len decreased, it means the environment was reset internally
        if ep_len_before is not None and hasattr(self.env.unwrapped, "episode_length_buf"):
            ep_len_after = self.env.unwrapped.episode_length_buf
            if isinstance(ep_len_after, torch.Tensor) and isinstance(ep_len_before, torch.Tensor):
                # Detect which envs had their episode_length decrease (means reset happened)
                just_reset = ep_len_after < ep_len_before

                if just_reset.any():
                    just_reset_np = just_reset.cpu().numpy()
                    # Convert terminated/truncated to numpy first (handle CUDA tensors)
                    if isinstance(terminated, torch.Tensor):
                        terminated = terminated.cpu().numpy()
                    if isinstance(truncated, torch.Tensor):
                        truncated = truncated.cpu().numpy()
                    terminated = np.asarray(terminated, dtype=bool).reshape(self.num_envs)
                    truncated = np.asarray(truncated, dtype=bool).reshape(self.num_envs)
                    # For IsaacLab, timeout is the main reason for reset
                    truncated = truncated | just_reset_np

        # Convert observations
        obs = self._to_batched_obs(obs)
        info = _to_numpy(info)
        reward = self._to_batched_scalar(reward, np.float32) 

        # Convert terminated/truncated to numpy
        if isinstance(terminated, torch.Tensor):
            terminated = _to_numpy(terminated)
        if isinstance(truncated, torch.Tensor):
            truncated = _to_numpy(truncated)

        # Ensure they're boolean arrays of shape (num_envs,)
        terminated = np.asarray(terminated, dtype=bool).reshape(self.num_envs)
        truncated = np.asarray(truncated, dtype=bool).reshape(self.num_envs)

        return obs, reward, terminated, truncated, info


def _run_local_offpolicy_training(
    env,
    algo_name: str,
    log_dir: str,
    seed: int | None,
    device: str | None,
    wandb_project: str | None = None,
    save_interval: int = 5000,
):
    from rl_base.algorithms import SAC, TD3
    import wandb
    from collections import deque
    import statistics
    import time

    # Wrap environment to collect metrics - similar to PPO's approach
    class WandbLoggingWrapper(gym.Wrapper):
        def __init__(self, env, num_envs):
            super().__init__(env)
            self.num_envs = num_envs
            # Episode storage (like PPO)
            self.ep_infos = []
            self.rewbuffer = deque(maxlen=100)
            self.lenbuffer = deque(maxlen=100)

            self.cur_reward_sum = np.zeros(num_envs, dtype=np.float32)
            self.cur_episode_length = np.zeros(num_envs, dtype=np.float32)
            self._total_episodes_completed = 0
            self.start_time = time.time()

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self.cur_reward_sum[:] = 0
            self.cur_episode_length[:] = 0
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            dones = terminated | truncated

            # Update episode stats
            self.cur_reward_sum += reward
            self.cur_episode_length += 1

            # Process completed episodes
            done_ids = np.where(dones)[0]
            if len(done_ids) > 0:
                self._total_episodes_completed += len(done_ids)

                for i in done_ids:
                    self.rewbuffer.append(self.cur_reward_sum[i])
                    self.lenbuffer.append(self.cur_episode_length[i])
                    self.cur_reward_sum[i] = 0
                    self.cur_episode_length[i] = 0

                # Only collect log info for done envs, like PPO does
                if "log" in info:
                    ep_info = {}
                    for key, val in info["log"].items():
                        if isinstance(val, torch.Tensor):
                            ep_info[key] = val[done_ids].cpu().float()
                        elif isinstance(val, np.ndarray):
                            ep_info[key] = (val[done_ids] if val.ndim > 0 else val).astype(np.float32)
                        elif isinstance(val, (int, float)):
                            ep_info[key] = float(val)
                    if ep_info:
                        self.ep_infos.append(ep_info)

            return obs, reward, terminated, truncated, info

        def get_metrics(self):
            """Process metrics like PPO's log method."""
            metrics = {}

            # Episode-based metrics
            if len(self.rewbuffer) > 0:
                metrics["Train/mean_reward"] = float(np.mean(self.rewbuffer))
                metrics["Train/std_reward"] = float(np.std(self.rewbuffer))
            if len(self.lenbuffer) > 0:
                metrics["Train/mean_episode_length"] = float(np.mean(self.lenbuffer))
                metrics["Train/episode_length_max"] = float(np.max(self.lenbuffer))

            # Total episode count
            if hasattr(self, '_total_episodes_completed'):
                metrics["Train/num_episodes"] = self._total_episodes_completed

            # Process ep_infos like PPO does (lines 300-319 in on_policy_runner.py)
            if self.ep_infos:
                # Get all unique keys from all log dicts
                all_keys = set()
                for ep_info in self.ep_infos:
                    if isinstance(ep_info, dict):
                        all_keys.update(ep_info.keys())

                # For each key, compute mean across all episodes
                for key in all_keys:
                    values = []
                    for ep_info in self.ep_infos:
                        if isinstance(ep_info, dict) and key in ep_info:
                            val = ep_info[key]
                            # Handle different value types
                            if isinstance(val, (int, float, np.number)):
                                values.append(float(val))
                            elif isinstance(val, torch.Tensor):
                                values.append(float(val.float().mean().item()))
                            elif isinstance(val, np.ndarray):
                                values.append(float(np.mean(val)))

                    if values:
                        mean_val = float(np.mean(values))
                        # Use PPO's naming convention
                        if "/" in key:
                            metrics[key] = mean_val
                        else:
                            metrics[f"Episode/{key}"] = mean_val

            # Clear ep_infos after processing
            self.ep_infos.clear()

            return metrics

    # Wrap the environment
    wrapped_env = OffPolicyVecEnvWrapper(env)
    logged_env = WandbLoggingWrapper(wrapped_env, wrapped_env.num_envs)

    # Print environment info
    print(f"[INFO] Environment: num_envs={logged_env.num_envs}")
    if hasattr(env.unwrapped, "max_episode_length"):
        print(f"[INFO] max_episode_length: {env.unwrapped.max_episode_length}")
    if hasattr(env.unwrapped, "episode_length_s"):
        print(f"[INFO] episode_length_s: {env.unwrapped.episode_length_s}")
    if hasattr(env, "episode_length_buf"):
        print(f"[INFO] episode_length_buf: {env.episode_length_buf}")
        print(f"[INFO] episode_length_buf shape: {env.episode_length_buf.shape}")
        print(f"[INFO] episode_length_buf dtype: {env.episode_length_buf.dtype}")
    if hasattr(env.unwrapped, "common_step_counter"):
        print(f"[INFO] common_step_counter: {env.unwrapped.common_step_counter}")

    # Initialize wandb
    run_name = os.path.split(log_dir)[-1]
    wandb_project = wandb_project or "unitree_rl_lab"
    wandb_entity = os.environ.get("WANDB_USERNAME") or os.environ.get("WANDB_ENTITY")

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        dir=log_dir,
        config={
            "log_dir": log_dir,
            "algorithm": algo_name,
            "total_timesteps": args_cli.offpolicy_total_timesteps,
            "learning_rate": args_cli.offpolicy_learning_rate,
            "buffer_size": args_cli.offpolicy_buffer_size,
            "batch_size": args_cli.offpolicy_batch_size,
            "learning_starts": args_cli.offpolicy_learning_starts,
            "num_envs": logged_env.num_envs,
            "save_interval": save_interval,
        }
    )

    # Common kwargs for TD3/SAC
    common_kwargs = dict(
        policy="MlpPolicy",
        env=logged_env,
        learning_rate=args_cli.offpolicy_learning_rate,
        buffer_size=args_cli.offpolicy_buffer_size,
        learning_starts=args_cli.offpolicy_learning_starts,
        batch_size=args_cli.offpolicy_batch_size,
        tau=args_cli.offpolicy_tau,
        gamma=args_cli.offpolicy_gamma,
        train_freq=args_cli.offpolicy_train_freq,
        gradient_steps=args_cli.offpolicy_gradient_steps,
        verbose=1,
        seed=seed,
        device=device or "auto",
    )

    algo_name = algo_name.lower()
    if algo_name == "td3":
        model = TD3(**common_kwargs)
        tb_name = "TD3"
    elif algo_name == "sac":
        model = SAC(**common_kwargs)
        tb_name = "SAC"
    else:
        raise ValueError(f"Unsupported off-policy algorithm: {algo_name}")

    print(
        f"[INFO] Starting local {tb_name} training with wandb: "
        f"total_timesteps={args_cli.offpolicy_total_timesteps}, "
        f"log_dir={log_dir}, "
        f"wandb_project={wandb_project}"
    )

    # Custom learn with wandb logging
    total_timesteps = args_cli.offpolicy_total_timesteps
    save_interval = max(1, int(save_interval))
    # Use a reasonable log interval (e.g., every 5000 environment steps)
    wandb_log_interval = 5000

    print(
        f"[INFO] Starting training: {total_timesteps} timesteps, "
        f"log every {wandb_log_interval} steps, "
        f"checkpoint every {save_interval} iterations"
    )

    offpolicy_it = 0

    while model.total_timesteps < total_timesteps:
        # Calculate next target
        next_target = min(model.total_timesteps + wandb_log_interval, total_timesteps)

        # Train for this segment
        model.learn(
            total_timesteps=next_target,
            log_interval=999999,  # Disable internal logging
        )

        # Log metrics after each segment
        metrics = logged_env.get_metrics()
        metrics["Train/timestep"] = model.total_timesteps

        # Check environment episode length for debugging
        if hasattr(logged_env.env.unwrapped, "episode_length_buf"):
            ep_len_buf = logged_env.env.unwrapped.episode_length_buf.cpu().numpy()
            metrics["Train/episode_length_max"] = float(np.max(ep_len_buf))
            metrics["Train/episode_length_mean"] = float(np.mean(ep_len_buf))
            if hasattr(logged_env.env.unwrapped, "max_episode_length"):
                max_ep_len = logged_env.env.unwrapped.max_episode_length
                metrics["Train/episode_length_max_ratio"] = float(np.max(ep_len_buf) / max_ep_len)

        # Performance metrics
        elapsed_time = time.time() - logged_env.start_time
        fps = model.total_timesteps / elapsed_time if elapsed_time > 0 else 0
        metrics["Perf/total_fps"] = fps

        # Log model-specific metrics (losses, etc.)
        if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
            logger_metrics_count = 0
            for key, value in model.logger.name_to_value.items():
                if not np.isnan(value):
                    logger_metrics_count += 1
                    # Remove algorithm prefix
                    clean_key = key
                    for algo in ["TD3", "SAC", "td3", "sac"]:
                        if key.startswith(f"{algo}/"):
                            clean_key = key[len(f"{algo}/"):]
                            break
                    # Categorize metrics
                    if "critic_loss" in clean_key or "value_loss" in clean_key:
                        metrics["Loss/value_loss"] = value
                    elif "actor_loss" in clean_key or "policy_loss" in clean_key or "surrogate" in clean_key:
                        metrics["Loss/policy_loss"] = value
                    elif "ent_coef" in clean_key and "loss" not in clean_key:
                        metrics["Loss/entropy_coef"] = value
                    elif "ent_coef_loss" in clean_key:
                        metrics["Loss/entropy_coef_loss"] = value
                    elif "episode_reward" in clean_key:
                        metrics["Train/episode_reward"] = value
                    elif "episode_length" in clean_key:
                        metrics["Train/episode_length"] = value
                    else:
                        metrics[f"Train/{clean_key}"] = value

        wandb.log(metrics)

        # Print summary
        mean_reward = metrics.get('Train/mean_reward', 'N/A')
        if mean_reward != 'N/A':
            reward_str = f"{mean_reward:.2f}"
        else:
            reward_str = 'N/A'

        loss_info = []
        if 'Loss/value_loss' in metrics:
            loss_info.append(f"critic={metrics['Loss/value_loss']:.4f}")
        if 'Loss/policy_loss' in metrics:
            loss_info.append(f"actor={metrics['Loss/policy_loss']:.4f}")
        if 'Loss/entropy_coef' in metrics:
            loss_info.append(f"ent={metrics['Loss/entropy_coef']:.4f}")

        loss_str = ", ".join(loss_info) if loss_info else "no losses yet"

        env_info = []
        for key in list(metrics.keys()):
            if key.startswith("Env/") or key.startswith("Curriculum/") or key.startswith("Reward/"):
                k = key.split('/', 1)[1] if '/' in key else key
                env_info.append(f"{k}={metrics[key]:.3f}")
                if len(env_info) >= 5:
                    break

        env_str = ", ".join(env_info) if env_info else ""

        # Episode stats
        max_ep_len = metrics.get('Train/episode_length_max', 0)
        ep_stats = f", EpDone={metrics.get('Train/num_episodes', 0)}, MaxEpLen={max_ep_len:.0f}"

        print(f"[INFO] Step {model.total_timesteps}/{total_timesteps}, "
              f"Reward: {reward_str}, "
              f"FPS: {fps:.0f}, "
              f"{loss_str}"
              + (f", {env_str}" if env_str else "")
              + ep_stats)

        # Save checkpoints like PPO: model_{it}.pt every save_interval iterations.
        if offpolicy_it % save_interval == 0:
            checkpoint_path = os.path.join(log_dir, f"model_{offpolicy_it}.pt")
            model.save(checkpoint_path)
            wandb.save(checkpoint_path, base_path=log_dir)
            print(f"[INFO] Saved checkpoint: {checkpoint_path}")

        offpolicy_it += 1

    # Save final checkpoint like PPO: model_{final_it}.pt
    final_it = max(offpolicy_it - 1, 0)
    final_checkpoint_path = os.path.join(log_dir, f"model_{final_it}.pt")
    if not os.path.exists(final_checkpoint_path):
        model.save(final_checkpoint_path)
        wandb.save(final_checkpoint_path, base_path=log_dir)
        print(f"[INFO] Saved final checkpoint: {final_checkpoint_path}")

    # Save model
    model_path = os.path.join(log_dir, f"{algo_name}_final_model")
    replay_path = os.path.join(log_dir, f"{algo_name}_replay_buffer.pkl")
    model.save(model_path)

    # Save to wandb
    wandb.save(model_path + ".pt", base_path=log_dir)

    # Save replay buffer for optional continuation
    save_replay_buffer = getattr(model, "save_replay_buffer", None)
    if callable(save_replay_buffer):
        save_replay_buffer(replay_path)
        wandb.save(replay_path, base_path=log_dir)

    print(f"[INFO] Saved {tb_name} model to: {model_path}.pt")

    # Finish wandb run
    wandb.finish()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rl_base_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RlBaseOnPolicyRunnerCfg):
    """Train with RL Base agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rl_base_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
     
    # Force Weights & Biases logging by default.
    setattr(agent_cfg, "logger", "wandb")
    os.environ["WANDB_BASE_URL"]='https://api.bandw.top'

    # Configure Weights & Biases logging when requested by the runner configuration.
    if getattr(agent_cfg, "logger", None) == "wandb":
        if not getattr(agent_cfg, "wandb_project", None):
            default_project = (
                args_cli.log_project_name
                or agent_cfg.experiment_name
                or args_cli.task
                or "unitree_rl_lab"
            )
            setattr(agent_cfg, "wandb_project", str(default_project).replace("/", "_"))
        print(
            "[INFO] Enabling Weights & Biases logging: "
            f"project='{agent_cfg.wandb_project}', run='{agent_cfg.run_name or 'auto'}'"
        )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    
    log_root_path = os.path.join(args_cli.log_root, "rl_base_mle", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = args_cli.resume_path

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-base

    # Initialize Waypoint Manager if enabled
    if args_cli.track_waypoints:
        try:
            from unitree_rl_lab.tasks.locomotion.robots.go2.policy_training4ensemble_deepmimic.velocity_env_cfg import WaypointCfg, WaypointManager
            
            # Get all env origins
            env_origins = getattr(env.unwrapped.scene, "env_origins", None)
            if env_origins is None:
                raise RuntimeError("env_origins missing")
            
            # Initialize vectorized manager
            waypoint_cfg = WaypointCfg()
            waypoint_cfg = WaypointCfg()
            # Override ratio from CLI if provided
            if getattr(args_cli, "waypoint_ratio", None) is not None:
                try:
                    waypoint_cfg.waypoint_ratio = float(args_cli.waypoint_ratio)
                except Exception:
                    pass
            cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
            waypoint_manager = WaypointManager(
                waypoint_cfg,
                env_origins,
                env.unwrapped.num_envs,
                env.unwrapped.device,
                command_term=cmd_term,
            )
            
            # Wrap environment
            env = WaypointWrapper(env, waypoint_manager)
            print("[INFO] Enabled Waypoint Tracking for Training.")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize Waypoint Manager: {e}")
            import traceback
            traceback.print_exc()

    # Optional local off-policy branch
    if args_cli.rl_algorithm in {"td3", "sac"}:
        if args_cli.distributed:
            raise ValueError("TD3/SAC mode does not support --distributed.")
        if args_cli.resume_path:
            print(
                "[WARN] --resume_path is ignored in TD3/SAC mode. "
                "Use local model loading workflow if you need resuming."
            )

        # dump config snapshots for reproducibility
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
        shutil.copy(
            inspect.getfile(env_cfg.__class__),
            os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
        )

        _run_local_offpolicy_training(
            env=env,
            algo_name=args_cli.rl_algorithm,
            log_dir=log_dir,
            seed=agent_cfg.seed,
            device=args_cli.device,
            wandb_project=agent_cfg.wandb_project,
            save_interval=args_cli.offpolicy_save_interval,
        )
        env.close()
        return

    env = RlBaseVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rl-base
    runner_cfg = agent_cfg.to_dict()
    
    runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )
    
    

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Finalize wandb run if it was created by the runner.
    if getattr(agent_cfg, "logger", None) == "wandb" and getattr(runner, "writer", None) is not None:
        stop_fn = getattr(runner.writer, "stop", None)
        if callable(stop_fn):
            stop_fn()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
