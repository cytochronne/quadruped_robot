# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate an RL agent from RL Base with standardized metrics."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import gc
import os
import time
import torch
import numpy as np
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RL Base.")
parser.add_argument(
    "--export_only",
    action="store_true",
    default=False,
    help="Export the policy to ONNX/JIT without launching Isaac Sim or creating an environment.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
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
parser.add_argument("--use_ensemble", action="store_true", default=True, help="Use ensemble encoders for uncertainty estimation.")
parser.add_argument("--encoder_size", type=int, default=3, help="Number of ensemble encoders to use.")
parser.add_argument("--track_waypoints", action="store_true", default=False, help="Enable waypoint tracking mode.")
parser.add_argument("--eval_duration", type=float, default=20.0, help="Duration of evaluation in seconds.")
parser.add_argument(
    "--student_activation",
    type=str,
    default="elu",
    help="Activation used to reconstruct the student MLPs for export-only mode.",
)
parser.add_argument(
    "--student_height_obs_dim",
    type=int,
    default=0,
    help="Height observation tail size in student observations for export-only mode.",
)
parser.add_argument(
    "--fail_horizon_steps",
    type=int,
    default=100,
    help="Horizon H (in sim steps) for FailWithinH@BurstStart metric.",
)


# append RL Base cli arguments
cli_args.add_rl_base_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True



def _make_activation(name: str) -> torch.nn.Module:
    name = name.lower()
    if name == "elu":
        return torch.nn.ELU()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "selu":
        return torch.nn.SELU()
    if name == "crelu":
        return torch.nn.ReLU()
    if name == "lrelu":
        return torch.nn.LeakyReLU()
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    raise ValueError(f"Unsupported activation for export reconstruction: {name}")


def _build_mlp_from_state_dict(prefix: str, state_dict: dict[str, torch.Tensor], activation_name: str) -> torch.nn.Sequential:
    linear_indices = sorted(
        {
            int(key[len(prefix) + 1 :].split(".")[0])
            for key in state_dict
            if key.startswith(f"{prefix}.") and key.endswith(".weight")
        }
    )
    if not linear_indices:
        raise ValueError(f"No layers found for prefix '{prefix}' in checkpoint.")

    layers = []
    for i, layer_idx in enumerate(linear_indices):
        weight = state_dict[f"{prefix}.{layer_idx}.weight"]
        bias = state_dict[f"{prefix}.{layer_idx}.bias"]
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if i < len(linear_indices) - 1:
            layers.append(_make_activation(activation_name))
    return torch.nn.Sequential(*layers)


class _StudentPolicyExporter(torch.nn.Module):
    """Export-only student policy wrapper for recurrent distillation checkpoints."""

    def __init__(
        self,
        rnn: torch.nn.Module,
        student_encoder: torch.nn.Module,
        student_policy_head: torch.nn.Module,
        *,
        student_height_dim: int = 0,
        normalizer: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.rnn = copy.deepcopy(rnn)
        self.student_encoder = copy.deepcopy(student_encoder)
        self.student_policy_head = copy.deepcopy(student_policy_head)
        self.student_height_dim = int(max(0, student_height_dim))
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()

        self.rnn.cpu()
        self.student_encoder.cpu()
        self.student_policy_head.cpu()
        self.normalizer.cpu()
        self.eval()

        self.rnn_type = type(self.rnn).__name__.lower()
        if self.rnn_type not in {"lstm", "gru"}:
            raise NotImplementedError(f"Unsupported recurrent policy type for ONNX export: {self.rnn_type}")

    def _normalize_and_split(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.normalizer(obs)
        if self.student_height_dim > 0:
            return obs[..., :-self.student_height_dim]
        return obs

    def forward(self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor | None = None):
        core_obs = self._normalize_and_split(obs)
        if self.rnn_type == "lstm":
            if c_in is None:
                raise ValueError("LSTM export requires c_in.")
            features, (h_out, c_out) = self.rnn(core_obs.unsqueeze(0), (h_in, c_in))
        else:
            features, h_out = self.rnn(core_obs.unsqueeze(0), h_in)
            c_out = None
        latent = self.student_encoder(features.squeeze(0))
        actions = self.student_policy_head(latent)
        if c_out is None:
            return actions, h_out
        return actions, h_out, c_out

    def export_onnx(self, path: str, filename: str = "policy.onnx"):
        os.makedirs(path, exist_ok=True)
        obs = torch.zeros(1, self.rnn.input_size, dtype=torch.float32)
        h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, dtype=torch.float32)
        full_path = os.path.join(path, filename)

        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, dtype=torch.float32)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                full_path,
                export_params=True,
                opset_version=11,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            torch.onnx.export(
                self,
                (obs, h_in),
                full_path,
                export_params=True,
                opset_version=11,
                input_names=["obs", "h_in"],
                output_names=["actions", "h_out"],
                dynamic_axes={},
            )
        return full_path


def _maybe_get_student_exporter(policy_nn, normalizer=None):
    if not all(hasattr(policy_nn, attr) for attr in ("memory_s", "student_encoder", "student_policy_head")):
        return None
    rnn = getattr(policy_nn.memory_s, "rnn", None)
    if rnn is None:
        return None
    student_height_dim = getattr(policy_nn, "student_height_dim", 0)
    return _StudentPolicyExporter(
        rnn=rnn,
        student_encoder=policy_nn.student_encoder,
        student_policy_head=policy_nn.student_policy_head,
        student_height_dim=student_height_dim,
        normalizer=normalizer,
    )


def _resolve_export_checkpoint(args_cli, agent_cfg):
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.utils import get_checkpoint_path

    log_root_path = os.path.join(args_cli.log_root, "rl_base", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_base", args_cli.task)
        if not resume_path:
            raise RuntimeError("Pretrained checkpoint is unavailable for this task.")
        return resume_path
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _export_checkpoint_without_env(args_cli):
    if not args_cli.checkpoint:
        raise RuntimeError("--export_only currently requires an explicit --checkpoint path.")
    resume_path = os.path.abspath(args_cli.checkpoint)
    checkpoint = torch.load(resume_path, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]

    activation_name = args_cli.student_activation
    student_height_dim = int(args_cli.student_height_obs_dim)

    rnn_state_dict = {k.removeprefix("memory_s.rnn."): v for k, v in model_state_dict.items() if k.startswith("memory_s.rnn.")}
    if not rnn_state_dict:
        raise RuntimeError("Checkpoint does not contain student recurrent state; cannot export without environment.")

    input_size = rnn_state_dict["weight_ih_l0"].shape[1]
    hidden_size = rnn_state_dict["weight_hh_l0"].shape[1]
    num_layers = len([k for k in rnn_state_dict if k.startswith("weight_ih_l")])
    rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    rnn.load_state_dict(rnn_state_dict)

    student_encoder = _build_mlp_from_state_dict("student_encoder", model_state_dict, activation_name)
    student_policy_head = _build_mlp_from_state_dict("student_policy_head", model_state_dict, activation_name)
    exporter = _StudentPolicyExporter(
        rnn=rnn,
        student_encoder=student_encoder,
        student_policy_head=student_policy_head,
        student_height_dim=student_height_dim,
    )

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    onnx_path = exporter.export_onnx(export_model_dir, "policy.onnx")
    print(f"[INFO] Exported ONNX policy to: {os.path.abspath(onnx_path)}")
    return onnx_path


def main():
    """Evaluate RL Base agent."""
    if args_cli.export_only:
        _export_checkpoint_without_env(args_cli)
        return

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym

    from rl_base.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from rl_base.isaaclab_support import (
        RlBaseOnPolicyRunnerCfg,
        RlBaseVecEnvWrapper,
        export_policy_as_jit,
        export_policy_as_onnx,
    )
    from isaaclab_tasks.utils import get_checkpoint_path
    import unitree_rl_lab.tasks  # noqa: F401
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    def _iter_nested(obj):
        if obj is None:
            return
        if isinstance(obj, dict):
            for value in obj.values():
                yield from _iter_nested(value)
            return
        if isinstance(obj, (list, tuple, set)):
            for value in obj:
                yield from _iter_nested(value)
            return
        yield obj

    def _discover_burst_models(eval_env, num_envs: int):
        """Discover live burst noise model objects that carry per-env burst state."""
        models = []
        seen = set()

        # 1) Preferred path: observation manager internals
        try:
            obs_mgr = getattr(eval_env.unwrapped, "observation_manager", None)
            if obs_mgr is not None:
                term_container = getattr(obs_mgr, "_group_obs_class_instances", None)
                for term_obj in _iter_nested(term_container):
                    noise_container = getattr(term_obj, "_noise_model", None)
                    for noise_model in _iter_nested(noise_container):
                        if noise_model is None:
                            continue
                        burst_steps_left = getattr(noise_model, "burst_steps_left", None)
                        if not torch.is_tensor(burst_steps_left):
                            continue
                        if burst_steps_left.numel() != num_envs:
                            continue
                        model_id = id(noise_model)
                        if model_id in seen:
                            continue
                        seen.add(model_id)
                        models.append(noise_model)
        except Exception:
            pass

        # 2) Fallback: scan live objects to handle private layout changes
        if len(models) == 0:
            try:
                for obj in gc.get_objects():
                    try:
                        burst_steps_left = getattr(obj, "burst_steps_left", None)
                    except Exception:
                        continue
                    if not torch.is_tensor(burst_steps_left):
                        continue
                    if burst_steps_left.numel() != num_envs:
                        continue
                    if not hasattr(obj, "steps_until_burst"):
                        continue
                    model_id = id(obj)
                    if model_id in seen:
                        continue
                    seen.add(model_id)
                    models.append(obj)
            except Exception:
                pass

        return models

    def _get_burst_active_mask(eval_env, num_envs: int, device: torch.device, burst_models) -> torch.Tensor:
        """Return per-env burst-active mask (True if any burst model is active for that env)."""
        burst_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)

        try:
            # A) Direct from discovered live model instances
            for noise_model in burst_models:
                burst_steps_left = getattr(noise_model, "burst_steps_left", None)
                if not torch.is_tensor(burst_steps_left):
                    continue
                if burst_steps_left.numel() != num_envs:
                    continue
                burst_mask |= (burst_steps_left > 0).to(device=device)

            obs_mgr = getattr(eval_env.unwrapped, "observation_manager", None)
            if obs_mgr is None:
                return burst_mask

            # B) Backup from observation manager instance path
            term_container = getattr(obs_mgr, "_group_obs_class_instances", None)
            seen_models = set()
            for term_obj in _iter_nested(term_container):
                noise_container = getattr(term_obj, "_noise_model", None)
                for noise_model in _iter_nested(noise_container):
                    model_id = id(noise_model)
                    if model_id in seen_models:
                        continue
                    seen_models.add(model_id)

                    burst_steps_left = getattr(noise_model, "burst_steps_left", None)
                    if not torch.is_tensor(burst_steps_left):
                        continue
                    if burst_steps_left.numel() != num_envs:
                        continue

                    burst_mask |= (burst_steps_left > 0).to(device=device)

            # C) Backup from noise cfg class shared states
            term_cfg_groups = getattr(obs_mgr, "_group_obs_term_cfgs", None)
            for term_cfg in _iter_nested(term_cfg_groups):
                noise_cfg = getattr(term_cfg, "noise", None)
                class_type = getattr(noise_cfg, "class_type", None)
                shared_states = getattr(class_type, "_shared_states", None)
                if not isinstance(shared_states, dict):
                    continue
                for state in shared_states.values():
                    if not isinstance(state, dict):
                        continue
                    burst_steps_left = state.get("burst_steps_left", None)
                    if not torch.is_tensor(burst_steps_left):
                        continue
                    if burst_steps_left.numel() != num_envs:
                        continue
                    burst_mask |= (burst_steps_left > 0).to(device=device)
        except Exception:
            pass
        return burst_mask

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    print("INFO: cli.device:", args_cli.device)
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
            "video_folder": os.path.join(log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-base
    env = RlBaseVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    loaded_dict = torch.load(resume_path, weights_only=False)
    is_off_policy = "actor" in loaded_dict and "model_state_dict" not in loaded_dict

    if is_off_policy:
        algo_name = loaded_dict.get("config", {}).get("algo", "unknown")
        print(f"[INFO]: Detected off-policy checkpoint ({algo_name}), loading actor directly.")

        actor_state = loaded_dict["actor"]
        action_low = actor_state["action_low"].cpu().numpy()
        action_high = actor_state["action_high"].cpu().numpy()
        action_dim = len(action_low)

        if "actor_target" in loaded_dict:
            from rl_base.algorithms.td3 import TD3Actor
            obs_dim = actor_state["backbone.0.weight"].shape[1]
            actor = TD3Actor(obs_dim, action_dim, action_low, action_high).to(env.unwrapped.device)
        else:
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
        runner_cfg = agent_cfg.to_dict()
        if args_cli.use_ensemble:
            runner_cfg.setdefault("policy", {})
            runner_cfg["policy"]["ensemble_size"] = int(args_cli.encoder_size)
        print("INFO: agent_cfg.device: ", agent_cfg.device)
        runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=env.device)
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
    print(f"[CHECK] Checkpoint path: {resume_path}")

    export_model_dir = os.path.join(log_dir, "exported")
    onnx_path = None
    if not is_off_policy:
        try:
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        except Exception as jit_err:
            print(f"[WARN] Default JIT export failed: {jit_err}")

        try:
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
            onnx_path = os.path.abspath(os.path.join(export_model_dir, "policy.onnx"))
        except Exception as onnx_err:
            print(f"[WARN] Default ONNX export failed: {onnx_err}")
            student_exporter = _maybe_get_student_exporter(policy_nn, normalizer=normalizer)
            if student_exporter is None:
                raise
            onnx_path = os.path.abspath(student_exporter.export_onnx(export_model_dir, "policy.onnx"))
    if onnx_path is not None:
        print(f"[INFO] Exported ONNX policy to: {onnx_path}")
    
    # Initialize Waypoint Manager if enabled
    waypoint_manager = None
    marker_visualizer = None
    if args_cli.track_waypoints:
        try:
            from unitree_rl_lab.tasks.locomotion.robots.go2.policy_evaluation.velocity_env_cfg import (
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
            
            # Setup visualization if needed, but for eval we might skip to save perf, 
            # unless video is on. We'll keep it for visual confirmation if user watches.
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
            waypoint_manager = None
            marker_visualizer = None

    dt = env.unwrapped.step_dt
    burst_models = _discover_burst_models(env, env.num_envs)
    print(f"[INFO] Burst detector found {len(burst_models)} live burst model(s).")
    try:
        obs_mgr = getattr(env.unwrapped, "observation_manager", None)
        policy_cfg = getattr(getattr(obs_mgr, "cfg", None), "policy", None)
        corruption_flag = getattr(policy_cfg, "enable_corruption", None)
        print(f"[INFO] Policy observation corruption enabled: {corruption_flag}")
    except Exception:
        pass
    
    # Validation Metrics Storage
    total_steps = 0
    # accum_acc_lin = 0.0 # Deprecated
    # accum_success = 0.0 # Deprecated
    
    # New Metrics
    accum_tracking_error = 0.0
    accum_energy = 0.0
    accum_action_smoothness = 0.0

    # Segmented metrics by burst-active vs non-burst-active (env-step level)
    burst_tracking_error_sum = 0.0
    burst_tracking_error_count = 0
    non_burst_tracking_error_sum = 0.0
    non_burst_tracking_error_count = 0

    burst_energy_sum = 0.0
    burst_energy_count = 0
    non_burst_energy_sum = 0.0
    non_burst_energy_count = 0

    burst_survived_steps = 0.0
    burst_total_steps = 0
    non_burst_survived_steps = 0.0
    non_burst_total_steps = 0

    # FailWithinH@BurstStart:
    # denominator = number of per-env burst-start events
    # numerator = number of those events whose env terminates within H steps
    fail_horizon_steps = max(1, int(args_cli.fail_horizon_steps))
    burst_start_count = 0
    fail_within_h_count = 0
    fail_within_h_noreset_count = 0
    burst_window_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    burst_window_remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    prev_burst_active_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    survival_ratios = []
    
    # Track episode lengths manually to be safe
    current_episode_lengths = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    max_episode_length = getattr(env.unwrapped, "max_episode_length", 1000)
    
    # Waypoint per-episode aggregation
    completed_episode_goals_sum = 0.0
    completed_episode_count = 0

    # reset environment
    ret = env.get_observations()
    if isinstance(ret, tuple):
        obs, extras = ret
    else:
        obs = ret
        extras = {}
    
    # Get reward params for metrics calculation
    # (std_lin and lin_thresh are no longer used for primary metrics but kept to avoid breakage if used elsewhere)
    try:
        rm = getattr(env.unwrapped, "reward_manager", None)
        track_cfg = rm.get_term_cfg("track_lin_vel_xy") if rm is not None else None
        std_lin = float(track_cfg.params.get("std", 0.5)) if track_cfg is not None else 0.5
        succ_cfg = rm.get_term_cfg("vel_tracking_success") if rm is not None else None
        lin_thresh = float(succ_cfg.params.get("lin_thresh", 0.1)) if succ_cfg is not None else 0.1
    except Exception:
        std_lin, lin_thresh = 0.5, 0.1

    # Calculate target simulation steps
    target_sim_steps = int(args_cli.eval_duration / dt)
    print(f"[INFO] Starting evaluation for {args_cli.eval_duration}s simulation time ({target_sim_steps} steps, dt={dt:.4f}s)")
    start_eval_time = time.time()
    sim_step_count = 0
    
    last_cmd_tensor = None
    prev_actions = None

    while simulation_app.is_running():
        # Check if target simulation steps reached
        if sim_step_count >= target_sim_steps:
            elapsed_real = time.time() - start_eval_time
            sim_time = sim_step_count * dt
            print(f"\n[INFO] Evaluation finished: {sim_step_count} steps ({sim_time:.2f}s sim, {elapsed_real:.2f}s real)")
            break
            
        # --- Waypoint Tracking Logic ---
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
                cmd_vel = torch.zeros(env.num_envs, 3, device=env.device)
            
            term_cmd = env.unwrapped.command_manager.get_term("base_velocity")
            if hasattr(term_cmd, "vel_command_b"):
                term_cmd.vel_command_b[:] = cmd_vel
            
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

        # run in inference mode
        with torch.inference_mode():
            actions = policy(obs, extras) if is_off_policy else policy(obs)
            ret = env.step(actions)
            
            # Correctly unpack env.step() return values (5-tuple in newer gym)
            if isinstance(ret, tuple) and len(ret) == 5:
                obs, _, terminated, truncated, extras = ret
                term = terminated | truncated  # Combine both signals
                terminated_mask = torch.as_tensor(terminated, device=env.device)
                if terminated_mask.dtype != torch.bool:
                    terminated_mask = terminated_mask != 0
            elif isinstance(ret, tuple) and len(ret) == 4:
                obs, _, term, extras = ret
                # Fallback: legacy API has no explicit terminated/truncated split.
                terminated_mask = torch.as_tensor(term, device=env.device)
                if terminated_mask.dtype != torch.bool:
                    terminated_mask = terminated_mask != 0
            else:
                obs, _, term, extras = ret
                terminated_mask = torch.as_tensor(term, device=env.device)
                if terminated_mask.dtype != torch.bool:
                    terminated_mask = terminated_mask != 0

            term = torch.as_tensor(term, device=env.device)
            if term.dtype != torch.bool:
                term = term != 0

            # Re-discover periodically in case manager lazily creates/replaces noise models.
            if sim_step_count % 200 == 0:
                burst_models = _discover_burst_models(env, env.num_envs)

            burst_active_mask = _get_burst_active_mask(env, env.num_envs, env.device, burst_models)
            non_burst_mask = ~burst_active_mask

            # Detect per-env burst starts (False -> True transition)
            burst_start_mask = burst_active_mask & torch.logical_not(prev_burst_active_mask)
            burst_start_num = int(burst_start_mask.sum().item())
            if burst_start_num > 0:
                burst_start_count += burst_start_num
                burst_window_active[burst_start_mask] = True
                burst_window_remaining[burst_start_mask] = fail_horizon_steps

            # Update episode lengths
            current_episode_lengths += 1

            # Step-wise survival by segment (alive=1 if not terminated on this step)
            alive = torch.logical_not(term).float()
            burst_count_step = int(burst_active_mask.sum().item())
            non_burst_count_step = int(non_burst_mask.sum().item())
            if burst_count_step > 0:
                burst_survived_steps += alive[burst_active_mask].sum().item()
                burst_total_steps += burst_count_step
            if non_burst_count_step > 0:
                non_burst_survived_steps += alive[non_burst_mask].sum().item()
                non_burst_total_steps += non_burst_count_step

            if sim_step_count < 10:
                max_burst_left = 0
                for noise_model in burst_models:
                    burst_steps_left = getattr(noise_model, "burst_steps_left", None)
                    if torch.is_tensor(burst_steps_left) and burst_steps_left.numel() == env.num_envs:
                        max_burst_left = max(max_burst_left, int(burst_steps_left.max().item()))
                # print(
                #     f"[BURST-DEBUG] step={sim_step_count:04d} active_envs={burst_count_step}/{env.num_envs} "
                #     f"max_burst_steps_left={max_burst_left}"
                # )

            # Update FailWithinH@BurstStart windows using termination on this step.
            failed_in_window_mask = term & burst_window_active
            failed_in_window_num = int(failed_in_window_mask.sum().item())
            if failed_in_window_num > 0:
                fail_within_h_count += failed_in_window_num

            failed_in_window_noreset_mask = terminated_mask & burst_window_active
            failed_in_window_noreset_num = int(failed_in_window_noreset_mask.sum().item())
            if failed_in_window_noreset_num > 0:
                fail_within_h_noreset_count += failed_in_window_noreset_num

            if failed_in_window_mask.any():
                burst_window_active[failed_in_window_mask] = False
                burst_window_remaining[failed_in_window_mask] = 0

            active_window_mask = burst_window_active
            if active_window_mask.any():
                burst_window_remaining[active_window_mask] -= 1
                expired_mask = burst_window_active & (burst_window_remaining <= 0)
                if expired_mask.any():
                    burst_window_active[expired_mask] = False
                    burst_window_remaining[expired_mask] = 0

            prev_burst_active_mask = burst_active_mask.clone()
            
            # Handle terminations for survival rate
            if term.any():
                terminated_ids = torch.nonzero(term).flatten()
                # Survival Rate = Survival Time / Max Episode Length
                # Using 1.0 as upper bound in case of slight potential overrun
                ratios = torch.clamp(current_episode_lengths[terminated_ids] / max_episode_length, max=1.0)
                survival_ratios.extend(ratios.cpu().tolist())
                current_episode_lengths[terminated_ids] = 0

            if waypoint_manager is not None:
                if term.any():
                    reset_ids = torch.nonzero(term).flatten()
                    try:
                        if "goals_reached" in waypoint_manager.metrics:
                            completed_episode_goals_sum += waypoint_manager.metrics["goals_reached"][reset_ids].sum().item()
                            completed_episode_count += int(reset_ids.numel())
                    except Exception:
                        pass
                    waypoint_manager.reset(reset_ids)
                if last_cmd_tensor is not None:
                    try:
                        term_cmd = env.unwrapped.command_manager.get_term("base_velocity")
                        term_cmd.vel_command_b[:] = last_cmd_tensor
                    except Exception:
                        pass
            
            # --- Collection Metrics ---
            # Tracking Error: ||v_xy - v_cmd_xy||^2 (MSE)
            try:
                cmd_b = env.unwrapped.command_manager.get_command("base_velocity")
                vel_b = env.unwrapped.scene["robot"].data.root_lin_vel_b[:, :2]
                # MSE of XY velocity
                tracking_error = torch.sum(torch.square(vel_b - cmd_b[:, :2]), dim=1)
                accum_tracking_error += tracking_error.mean().item()

                if burst_count_step > 0:
                    burst_tracking_error_sum += tracking_error[burst_active_mask].sum().item()
                    burst_tracking_error_count += burst_count_step
                if non_burst_count_step > 0:
                    non_burst_tracking_error_sum += tracking_error[non_burst_mask].sum().item()
                    non_burst_tracking_error_count += non_burst_count_step
            except Exception:
                pass

            # Energy Proxy: sum(|qvel| * |qfrc|) over joints
            try:
                robot_data = env.unwrapped.scene["robot"].data
                qvel = getattr(robot_data, "joint_vel", None)
                if qvel is None:
                    qvel = getattr(robot_data, "dof_vel", None)
                qfrc = getattr(robot_data, "applied_torque", None)
                if qfrc is None:
                    qfrc = getattr(robot_data, "dof_torque", None)
                if qvel is not None and qfrc is not None:
                    energy_step = torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=1)
                    accum_energy += energy_step.mean().item()

                    if burst_count_step > 0:
                        burst_energy_sum += energy_step[burst_active_mask].sum().item()
                        burst_energy_count += burst_count_step
                    if non_burst_count_step > 0:
                        non_burst_energy_sum += energy_step[non_burst_mask].sum().item()
                        non_burst_energy_count += non_burst_count_step
            except Exception:
                pass

            # Action Smoothness: ||a_t - a_{t-1}||^2 (MSE)
            try:
                if prev_actions is not None:
                    delta_actions = actions - prev_actions
                    smoothness_step = torch.sum(torch.square(delta_actions), dim=1)
                    accum_action_smoothness += smoothness_step.mean().item()
                prev_actions = actions.clone()
            except Exception:
                pass
            
            total_steps += 1
            sim_step_count += 1
            
            # Progress Print
            if sim_step_count % 50 == 0:
                elapsed_real = time.time() - start_eval_time
                sim_time = sim_step_count * dt
                ratio = sim_time / elapsed_real if elapsed_real > 0 else 0.0
                print(f"Progress: {sim_step_count}/{target_sim_steps} steps | Sim: {sim_time:.1f}s | Real: {elapsed_real:.1f}s | Ratio: {ratio:.2f}x", 
                      end="\r", flush=True)

    # Calculate final metrics
    avg_tracking_error = accum_tracking_error / max(1, total_steps)
    avg_energy = accum_energy / max(1, total_steps)
    avg_action_smoothness = accum_action_smoothness / max(1, total_steps)

    burst_tracking_error = burst_tracking_error_sum / burst_tracking_error_count if burst_tracking_error_count > 0 else None
    non_burst_tracking_error = (
        non_burst_tracking_error_sum / non_burst_tracking_error_count if non_burst_tracking_error_count > 0 else None
    )
    burst_energy = burst_energy_sum / burst_energy_count if burst_energy_count > 0 else None
    non_burst_energy = non_burst_energy_sum / non_burst_energy_count if non_burst_energy_count > 0 else None
    burst_alive_rate = burst_survived_steps / burst_total_steps if burst_total_steps > 0 else None
    non_burst_alive_rate = non_burst_survived_steps / non_burst_total_steps if non_burst_total_steps > 0 else None
    fail_within_h_at_burst_start = fail_within_h_count / burst_start_count if burst_start_count > 0 else None
    fail_within_h_at_burst_start_noreset = (
        fail_within_h_noreset_count / burst_start_count if burst_start_count > 0 else None
    )
    
    # Calculate average survival rate (Survival Duration / Max Duration)
    # Include currently running episodes as well for better estimate if needed, 
    # but strictly "Survival Rate" implies completed/failed episodes.
    # If no episodes finished, use current lengths as lower bound or N/A.
    if len(survival_ratios) > 0:
        avg_survival_rate = sum(survival_ratios) / len(survival_ratios)
    else:
        # Fallback: Estimate from active episodes (Ratio of time survived so far)
        avg_survival_rate = (current_episode_lengths / max_episode_length).mean().item()
        avg_survival_rate = min(1.0, avg_survival_rate)
    
    total_goals_reached = 0
    goals_per_episode_avg = None
    
    if waypoint_manager is not None and hasattr(waypoint_manager, "metrics"):
        if "goals_reached" in waypoint_manager.metrics:
            total_goals_reached = completed_episode_goals_sum
            if completed_episode_count > 0:
                goals_per_episode_avg = completed_episode_goals_sum / completed_episode_count
    
    # Calculate final timing metrics
    final_sim_time = sim_step_count * dt
    final_real_time = time.time() - start_eval_time
    sim_real_ratio = final_sim_time / final_real_time if final_real_time > 0 else 0.0
    
    print("\n" + "="*40)
    print(f"EVALUATION RESULTS")
    print("="*40)
    print(f"Simulation Time          : {final_sim_time:.2f}s")
    print(f"Real-World Time          : {final_real_time:.2f}s")
    print(f"Sim/Real Ratio           : {sim_real_ratio:.2f}x")
    print(f"Total Steps              : {sim_step_count}")
    print("-"*40)
    print(f"Avg Tracking Error (MSE) : {avg_tracking_error:.4f} (m/s)^2")
    print(f"Avg Energy Proxy          : {avg_energy:.4f} (|qvel|*|qfrc|)")
    print(f"Avg Action Smoothness     : {avg_action_smoothness:.4f} (||Δa||^2)")
    print(f"Avg Survival Rate        : {avg_survival_rate:.4f} (Time/MaxTime)")
    print("-"*40)
    print(
        "Tracking Error @Burst    : "
        + (f"{burst_tracking_error:.4f} (m/s)^2" if burst_tracking_error is not None else "N/A")
    )
    print(
        "Tracking Error @NoBurst  : "
        + (f"{non_burst_tracking_error:.4f} (m/s)^2" if non_burst_tracking_error is not None else "N/A")
    )
    print(
        "Energy Proxy @Burst      : "
        + (f"{burst_energy:.4f} (|qvel|*|qfrc|)" if burst_energy is not None else "N/A")
    )
    print(
        "Energy Proxy @NoBurst    : "
        + (f"{non_burst_energy:.4f} (|qvel|*|qfrc|)" if non_burst_energy is not None else "N/A")
    )
    # print(
    #     "Alive Rate @Burst        : "
    #     + (
    #         f"{burst_alive_rate:.4f} ({burst_survived_steps:.0f}/{burst_total_steps})"
    #         if burst_alive_rate is not None
    #         else "N/A"
    #     )
    # )
    # print(
    #     "Alive Rate @NoBurst      : "
    #     + (
    #         f"{non_burst_alive_rate:.4f} ({non_burst_survived_steps:.0f}/{non_burst_total_steps})"
    #         if non_burst_alive_rate is not None
    #         else "N/A"
    #     )
    # )
    print(
        f"Burst Sample Count       : tracking={burst_tracking_error_count}, energy={burst_energy_count}, alive={burst_total_steps}"
    )
    print(
        f"NoBurst Sample Count     : tracking={non_burst_tracking_error_count}, energy={non_burst_energy_count}, alive={non_burst_total_steps}"
    )
    print(
        f"FailWithinH@BurstStart   : "
        + (
            f"{fail_within_h_at_burst_start:.4f} ({fail_within_h_count}/{burst_start_count}, H={fail_horizon_steps} steps)"
            if fail_within_h_at_burst_start is not None
            else f"N/A (0 burst starts, H={fail_horizon_steps} steps)"
        )
    )
    print(
        f"FailWithinH@BurstStart@NoReset: "
        + (
            f"{fail_within_h_at_burst_start_noreset:.4f} ({fail_within_h_noreset_count}/{burst_start_count}, H={fail_horizon_steps} steps)"
            if fail_within_h_at_burst_start_noreset is not None
            else f"N/A (0 burst starts, H={fail_horizon_steps} steps)"
        )
    )
    
    if args_cli.track_waypoints:
        print(f"Total Goals Reached    : {int(total_goals_reached)}")
        print(f"Completed Episodes     : {int(completed_episode_count)}")
        if goals_per_episode_avg is not None:
            print(f"Avg Goals per Episode  : {goals_per_episode_avg:.4f}")
        else:
            print("Avg Goals per Episode  : N/A (no completed episodes)")
        print(f"Goals Reached per Sec  : {total_goals_reached / args_cli.eval_duration:.4f}")
    print("="*40)

    # close the simulator
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
