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
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
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
    "--resume_path",
    type=str,
    default=None,
    help="Absolute or relative path to a checkpoint to resume training from (overrides load_run/load_checkpoint).",
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
import os
import shutil
import torch
from datetime import datetime


from rl_base.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

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
                        term.vel_command_b[use_wp] = cmd_vel[use_wp]
                else:
                    # Fallback to previous behavior (all envs use waypoints)
                    term.vel_command_b[:] = cmd_vel

        # Step
        ret = self.env.step(action)
        
        # Reset logic for waypoints
        # Gym API: obs, rew, terminated, truncated, info
        obs, rew, terminated, truncated, info = ret

        # # Add a sizable bonus when a waypoint is reached in waypoint mode
        # bonus_mask = getattr(self.waypoint_manager, "just_reached", None)
        # if bonus_mask is not None:
        #     bonus_value = getattr(self.waypoint_manager.cfg, "goal_bonus", 0.0)
        #     if not torch.is_tensor(rew):
        #         rew = torch.as_tensor(rew, device=bonus_mask.device)
        #     rew = rew + bonus_value * bonus_mask.float()
        
        dones = terminated | truncated
        if dones.any():
            reset_ids = torch.nonzero(dones).flatten()
            self.waypoint_manager.reset(reset_ids)
        
        #return (obs, rew, terminated, truncated, info)
        return ret

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
    
    log_root_path = os.path.join(args_cli.log_root, "rl_base", agent_cfg.experiment_name)
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
    if args_cli.resume_path or agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if args_cli.resume_path:
            resume_path = os.path.abspath(args_cli.resume_path)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

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

    # Initialize Waypoint Manager if enabled
    if args_cli.track_waypoints:
        try:
            from unitree_rl_lab.tasks.locomotion.robots.go2.teacher.velocity_env_cfg import WaypointCfg, WaypointManager
            
            # Get all env origins
            env_origins = getattr(env.unwrapped.scene, "env_origins", None)
            if env_origins is None:
                raise RuntimeError("env_origins missing")
            
            # Initialize vectorized manager
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

    # wrap around environment for rl-base
    env = RlBaseVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

 

    # create runner from rl-base
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if args_cli.resume_path or agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
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
