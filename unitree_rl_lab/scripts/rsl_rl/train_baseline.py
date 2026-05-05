# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

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
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
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
        "'auto' keeps existing RSL-RL behavior, "
        "'td3' and 'sac' run local off-policy implementations in rsl_rl_woUncertainty."
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



# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
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

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
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

from rsl_rl_woUncertainty.runners import OnPolicyRunner

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
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
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
        self.single_observation_space = self._strip_batch_dim(self.env.observation_space, self.num_envs)
        self.single_action_space = self._strip_batch_dim(self.env.action_space, self.num_envs)
        # Keep SB3-style naming while preserving compatibility with local algorithms.
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
    def _strip_batch_dim(space, num_envs: int):
        if isinstance(space, gym.spaces.Box) and len(space.shape) > 0 and space.shape[0] == num_envs:
            low = np.array(space.low[0], copy=True)
            high = np.array(space.high[0], copy=True)
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
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        obs = self._to_batched_obs(obs)
        info = _to_numpy(info)
        reward = self._to_batched_scalar(reward, np.float32)
        terminated = self._to_batched_scalar(terminated, np.bool_)
        truncated = self._to_batched_scalar(truncated, np.bool_)
        return obs, reward, terminated, truncated, info


def _run_local_offpolicy_training(env, algo_name: str, log_dir: str, seed: int | None, device: str | None):
    from rsl_rl_woUncertainty.algorithms import SAC, TD3

    wrapped_env = OffPolicyVecEnvWrapper(env)
    tb_log_dir = os.path.join(log_dir, "tb")
    os.makedirs(tb_log_dir, exist_ok=True)

    common_kwargs = dict(
        policy="MlpPolicy",
        env=wrapped_env,
        learning_rate=args_cli.offpolicy_learning_rate,
        buffer_size=args_cli.offpolicy_buffer_size,
        learning_starts=args_cli.offpolicy_learning_starts,
        batch_size=args_cli.offpolicy_batch_size,
        tau=args_cli.offpolicy_tau,
        gamma=args_cli.offpolicy_gamma,
        train_freq=args_cli.offpolicy_train_freq,
        gradient_steps=args_cli.offpolicy_gradient_steps,
        tensorboard_log=tb_log_dir,
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
        f"[INFO] Starting local {tb_name} training: "
        f"total_timesteps={args_cli.offpolicy_total_timesteps}, "
        f"log_dir={log_dir}"
    )
    model.learn(
        total_timesteps=args_cli.offpolicy_total_timesteps,
        log_interval=args_cli.offpolicy_log_interval,
        tb_log_name=tb_name,
    )

    model_path = os.path.join(log_dir, f"{algo_name}_final_model")
    replay_path = os.path.join(log_dir, f"{algo_name}_replay_buffer.pkl")
    model.save(model_path)
    # save replay buffer for optional continuation
    save_replay_buffer = getattr(model, "save_replay_buffer", None)
    if callable(save_replay_buffer):
        save_replay_buffer(replay_path)
    print(f"[INFO] Saved {tb_name} model to: {model_path}.pt")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
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
    
    log_root_path = os.path.join(args_cli.log_root, "rsl_rl_MLE", agent_cfg.experiment_name)
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

    # wrap around environment for rsl-rl

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
        )
        env.close()
        return

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
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
