# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL Base."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL Base.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--track_waypoints", action="store_true", default=False, help="Enable waypoint tracking mode.")
# append RL Base cli arguments
cli_args.add_rl_base_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rl_base.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from rl_base.isaaclab_support import RlBaseOnPolicyRunnerCfg, RlBaseVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RL Base agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RlBaseOnPolicyRunnerCfg = cli_args.parse_rl_base_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join(args_cli.log_root, "rl_base", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_base", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-base
    env = RlBaseVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    loaded_dict = torch.load(resume_path, weights_only=False)
    is_off_policy = "actor" in loaded_dict and "model_state_dict" not in loaded_dict

    if is_off_policy:
        # Off-policy checkpoint (SAC/TD3): reconstruct actor and load weights directly
        algo_name = loaded_dict.get("config", {}).get("algo", "unknown")
        print(f"[INFO]: Detected off-policy checkpoint ({algo_name}), loading actor directly.")

        # Infer dimensions from actor state_dict
        actor_state = loaded_dict["actor"]
        action_low = actor_state["action_low"].cpu().numpy()
        action_high = actor_state["action_high"].cpu().numpy()
        action_dim = len(action_low)

        if "actor_target" in loaded_dict:
            # TD3: obs_dim from backbone.0.weight shape
            from rl_base.algorithms.td3 import TD3Actor
            obs_dim = actor_state["backbone.0.weight"].shape[1]
            actor = TD3Actor(obs_dim, action_dim, action_low, action_high).to(env.unwrapped.device)
        else:
            # SAC: obs_dim from backbone.0.weight shape
            from rl_base.algorithms.sac import SACActor
            obs_dim = actor_state["backbone.0.weight"].shape[1]
            actor = SACActor(obs_dim, action_dim, action_low, action_high).to(env.unwrapped.device)

        actor.load_state_dict(actor_state)
        actor.eval()

        def _flatten_obs(obs, extras=None):
            obs_dict = extras.get("observations") if extras else None
            if obs_dict is None:
                obs_dict = obs if isinstance(obs, dict) else None
            if obs_dict is not None:
                parts = [obs_dict[k].reshape(obs_dict[k].shape[0], -1) for k in sorted(obs_dict.keys())]
                return torch.cat(parts, dim=-1)
            return obs

        _actor = actor
        if hasattr(_actor, "action_log_prob"):
            policy = lambda obs, extras=None: _actor(_flatten_obs(obs, extras), deterministic=True)
        else:
            policy = lambda obs, extras=None: _actor(_flatten_obs(obs, extras))
        policy_nn = actor
        normalizer = None
        runner = None
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

    # Initialize Waypoint Manager if enabled
    waypoint_manager = None
    marker_visualizer = None
    if args_cli.track_waypoints:
        try:
            from unitree_rl_lab.tasks.locomotion.robots.go2.student.velocity_env_cfg import (
                WaypointCfg,
                WaypointManager,
            )
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            import isaaclab.sim as sim_utils

            env_origins = getattr(env.unwrapped.scene, "env_origins", None)
            if env_origins is None:
                raise RuntimeError("env_origins missing")

            waypoint_cfg = WaypointCfg()
            cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
            waypoint_manager = WaypointManager(waypoint_cfg, env_origins, env.num_envs, env.device, command_term=cmd_term)

            marker_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/Waypoints",
                markers={
                    "current": sim_utils.SphereCfg(
                        radius=waypoint_cfg.marker_radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "completed": sim_utils.SphereCfg(
                        radius=waypoint_cfg.marker_radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                    "pending": sim_utils.SphereCfg(
                        radius=waypoint_cfg.marker_radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    ),
                },
            )
            marker_visualizer = VisualizationMarkers(marker_cfg)

            try:
                env.unwrapped.command_manager.set_debug_vis(True)
            except Exception as e:
                print(f"[WARN] Failed to enable command debug vis: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Waypoint Manager: {e}")
            import traceback

            traceback.print_exc()
            waypoint_manager = None
            marker_visualizer = None

    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    ret = env.get_observations()
    if isinstance(ret, tuple):
        obs, extras = ret
    else:
        obs = ret
        extras = {}
    if version("rl-base-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    last_cmd_tensor = None
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # Waypoint tracking (command override + visualization)
        if waypoint_manager is not None:
            robot_pos = env.unwrapped.scene["robot"].data.root_pos_w
            try:
                from isaaclab.utils.math import euler_xyz_from_quat

                quat = env.unwrapped.scene["robot"].data.root_quat_w
                _, _, robot_yaw = euler_xyz_from_quat(quat)
            except Exception:
                heading = getattr(env.unwrapped.scene["robot"].data, "heading_w", None)
                robot_yaw = heading.squeeze(-1) if heading is not None else torch.zeros(env.num_envs, device=env.device)

            try:
                cmd_vel, target_pos = waypoint_manager.compute_command(robot_pos, robot_yaw)
            except Exception as e:
                print(f"[ERROR] compute_command failed: {e}")
                cmd_vel = torch.zeros(env.num_envs, 3, device=env.device)
                target_pos = robot_pos

            term_cmd = env.unwrapped.command_manager.get_term("base_velocity")
            if hasattr(term_cmd, "vel_command_b"):
                term_cmd.vel_command_b[:] = cmd_vel
            else:
                print("[WARN] 'base_velocity' term does not have 'vel_command_b' attribute. Command override failed.")

            last_cmd_tensor = cmd_vel

            if marker_visualizer is not None:
                try:
                    positions, colors = waypoint_manager.get_marker_data()
                    indices = torch.full((colors.shape[0],), 2, dtype=torch.long, device=env.device)
                    indices[colors[:, 0] > 0.5] = 0
                    indices[colors[:, 1] > 0.5] = 1
                    marker_visualizer.visualize(translations=positions, marker_indices=indices)
                except Exception:
                    pass
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs, extras) if is_off_policy else policy(obs)
            # env stepping
            ret = env.step(actions)
            if isinstance(ret, tuple) and len(ret) == 4:
                obs, _, term, extras = ret
            else:
                obs, _, term, extras = ret

            if waypoint_manager is not None:
                if term.any():
                    reset_ids = torch.nonzero(term).flatten()
                    waypoint_manager.reset(reset_ids)
                if last_cmd_tensor is not None:
                    try:
                        term_cmd = env.unwrapped.command_manager.get_term("base_velocity")
                        term_cmd.vel_command_b[:] = last_cmd_tensor
                        if hasattr(term_cmd, "is_standing_env"):
                            term_cmd.is_standing_env[:] = False
                        if hasattr(term_cmd, "is_heading_env"):
                            term_cmd.is_heading_env[:] = False
                    except Exception as e:
                        print(f"[WARN] Failed to set velocity command: {e}")
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
