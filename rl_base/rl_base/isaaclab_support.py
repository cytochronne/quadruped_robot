from __future__ import annotations

import copy
import os
from dataclasses import MISSING
from typing import Literal

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.utils import configclass

from rl_base.env import VecEnv


@configclass
class RlBasePpoActorCriticCfg:
    """Configuration for PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    init_noise_std: float = MISSING
    noise_std_type: Literal["scalar", "log"] = "scalar"
    actor_hidden_dims: list[int] = MISSING
    critic_hidden_dims: list[int] = MISSING
    activation: str = MISSING


@configclass
class RlBasePpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    value_loss_coef: float = MISSING
    use_clipped_value_loss: bool = MISSING
    clip_param: float = MISSING
    entropy_coef: float = MISSING
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    schedule: str = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    desired_kl: float = MISSING
    max_grad_norm: float = MISSING
    normalize_advantage_per_mini_batch: bool = False
    symmetry_cfg: dict | None = None
    rnd_cfg: dict | None = None


@configclass
class RlBaseDistillationAlgorithmCfg:
    """Configuration for the distillation algorithm."""

    class_name: str = "Distillation"
    num_learning_epochs: int = MISSING
    learning_rate: float = MISSING
    gradient_length: int = MISSING
    max_grad_norm: None | float = None
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    loss_type: Literal["mse", "huber", "BCEWithLogits", "Wasserstein"] = "mse"


@configclass
class RlBaseOnPolicyRunnerCfg:
    """Configuration for on-policy runner."""

    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    empirical_normalization: bool | None = None
    clip_actions: float | None = None
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    class_name: str = "OnPolicyRunner"
    policy: object = MISSING
    algorithm: object = MISSING


class RlBaseVecEnvWrapper(VecEnv):
    """Wrap Isaac Lab environment for rl_base runner."""

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must inherit ManagerBasedRLEnv or DirectRLEnv. "
                f"Environment type: {type(env)}"
            )

        self.env = env
        self.clip_actions = clip_actions

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])

        if hasattr(self.unwrapped, "observation_manager") and "critic" in self.unwrapped.observation_manager.group_obs_dim:
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = 0

        self._modify_action_space()
        self.env.reset()

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        return self.env.unwrapped

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return obs, rew, dones, extras

    def close(self):
        return self.env.close()

    def _modify_action_space(self):
        if self.clip_actions is None:
            return

        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions,
            high=self.clip_actions,
            shape=(self.num_actions,),
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space,
            self.num_envs,
        )


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file."""

    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file."""

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of policy into a JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent

        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")

        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            if self.rnn_type == "lstm":
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = self.reset_memory
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
                self.reset = self.reset_memory
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward_gru(self, x):
        x = self.normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of policy into an ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent

        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")

        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()

        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
