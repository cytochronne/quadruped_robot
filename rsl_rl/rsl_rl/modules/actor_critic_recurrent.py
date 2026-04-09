# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import warnings
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation


class ActorCriticRecurrent(nn.Module):
    """Recurrent actor-critic with student-style actor and terrain-aware critic.
    
    Actor: RNN -> Encoder -> Policy Head (recurrent, no height input)
    Critic: Core + Height CNN -> Fusion -> Value Head (non-recurrent, terrain-aware)
    """
    
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        critic_height_obs_dim=88,
        height_cnn_channels=(16, 32),
        height_map_shape=None,
        student_encoder_hidden_dims=(256, 256),
        student_policy_hidden_dims=(256, 256, 256),
        fusion_encoder_dims=(256, 128, 96),
        student_latent_dim=96,
        noise_std_type="scalar",
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(list(kwargs.keys())),
            )

        super().__init__()

        if critic_height_obs_dim < 0:
            raise ValueError("critic_height_obs_dim must be non-negative.")

        # Validate critic observation split
        if critic_height_obs_dim == 0:
            self.critic_height_dim = 0
        elif critic_height_obs_dim >= num_critic_obs:
            print(
                "[WARN] Requested critic_height_obs_dim exceeds critic observations. "
                "Falling back to zero height features for the critic."
            )
            self.critic_height_dim = 0
        else:
            self.critic_height_dim = critic_height_obs_dim

        self.critic_core_dim = num_critic_obs - self.critic_height_dim
        if self.critic_core_dim <= 0:
            raise ValueError("Critic core observation dimension must be positive.")

        activation_name = activation
        self.student_latent_dim = student_latent_dim
        self.noise_std_type = noise_std_type

        # ============================================================
        # ACTOR COMPONENTS: Recurrent Student-Style
        # ============================================================
        # Actor RNN processes full actor observations
        self.memory_a = Memory(
            num_actor_obs, 
            type=rnn_type, 
            num_layers=rnn_num_layers, 
            hidden_size=rnn_hidden_dim
        )

        # Student encoder: RNN output -> latent features
        self.student_encoder = self._build_mlp(
            rnn_hidden_dim,
            student_encoder_hidden_dims,
            student_latent_dim,
            activation_name,
        )

        # Student policy head: latent features -> actions
        self.student_policy_head = self._build_mlp(
            student_latent_dim,
            student_policy_hidden_dims,
            num_actions,
            activation_name,
        )

        print(f"Actor RNN: {self.memory_a}")
        print(f"Student encoder: {self.student_encoder}")
        print(f"Student policy head: {self.student_policy_head}")

        # ============================================================
        # CRITIC COMPONENTS: Non-Recurrent Terrain-Aware
        # ============================================================
        # Height encoder (CNN)
        self.height_map_shape = self._resolve_height_map_shape(
            self.critic_height_dim, height_map_shape
        )
        self.height_encoder, self.height_embedding_dim = self._build_height_cnn(
            self.height_map_shape, height_cnn_channels, activation_name
        )

        # Fusion encoder: combines core + height features
        critic_fusion_in_dim = self.critic_core_dim + self.height_embedding_dim
        self.critic_fusion_encoder, self.critic_fusion_dim = self._build_fusion_encoder(
            critic_fusion_in_dim, fusion_encoder_dims, activation_name
        )

        # Critic head: fusion features -> value
        self.critic = self._build_head(
            self.critic_fusion_dim, critic_hidden_dims, 1, activation_name
        )

        print(f"Terrain encoder CNN: {self.height_encoder}")
        print(f"Critic fusion encoder: {self.critic_fusion_encoder}")
        print(f"Critic head: {self.critic}")

        # ============================================================
        # ACTION NOISE CONFIGURATION
        # ============================================================
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError("Unknown standard deviation type. Should be 'scalar' or 'log'.")

        Normal.set_default_validate_args(False)
        self.distribution = None

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_dims: Sequence[int] | None,
        output_dim: int,
        activation_name: str,
    ) -> nn.Sequential:
        """Build a simple MLP with activation after each hidden layer."""
        if hidden_dims is None or len(hidden_dims) == 0:
            return nn.Sequential(nn.Linear(input_dim, output_dim))

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for idx, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(resolve_nn_activation(activation_name))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_head(
        input_dim: int, 
        hidden_dims: Sequence[int], 
        output_dim: int, 
        activation_name: str
    ) -> nn.Sequential:
        """Build network head (same as _build_mlp but explicit naming for clarity)."""
        layers: list[nn.Module] = []
        prev_dim = input_dim
        hidden_dims = tuple(hidden_dims)
        if not hidden_dims:
            layers.append(nn.Linear(prev_dim, output_dim))
            return nn.Sequential(*layers)
    
        for idx, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(resolve_nn_activation(activation_name))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _resolve_height_map_shape(
        height_dim: int, 
        explicit_shape: Tuple[int, int] | None
    ) -> Tuple[int, int]:
        """Resolve 2D shape for height map."""
        if height_dim == 0:
            return 0, 0

        if explicit_shape is not None:
            if explicit_shape[0] * explicit_shape[1] != height_dim:
                raise ValueError(
                    "Provided height_map_shape does not match height_obs_dim. "
                    f"Got {explicit_shape[0]}x{explicit_shape[1]} != {height_dim}."
                )
            return explicit_shape

        factors: list[Tuple[int, int]] = []
        for h in range(1, int(math.sqrt(height_dim)) + 1):
            if height_dim % h == 0:
                factors.append((h, height_dim // h))

        if not factors:
            raise ValueError(f"Unable to factorize height_obs_dim={height_dim} into a 2D map shape.")

        # Choose the pair with the smallest aspect ratio difference
        best_h, best_w = min(factors, key=lambda hw: abs(hw[0] - hw[1]))
        return best_h, best_w

    @staticmethod
    def _build_height_cnn(
        map_shape: Tuple[int, int],
        channels: Sequence[int],
        activation_name: str,
    ) -> tuple[nn.Module, int]:
        """Build CNN for height map encoding."""
        height, width = map_shape
        if height == 0 or width == 0:
            return nn.Identity(), 0

        layers: list[nn.Module] = []
        in_channels = 1
        current_h, current_w = height, width
        for idx, out_channels in enumerate(channels):
            stride = 2 if current_h >= 4 and current_w >= 4 else 1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
            layers.append(resolve_nn_activation(activation_name))
            in_channels = out_channels
            current_h = math.floor((current_h + 2 - 3) / stride + 1)
            current_w = math.floor((current_w + 2 - 3) / stride + 1)
        layers.append(nn.Flatten())
        embedding_dim = in_channels * max(current_h, 1) * max(current_w, 1)
        return nn.Sequential(*layers), embedding_dim

    @staticmethod
    def _build_fusion_encoder(
        input_dim: int, 
        hidden_dims: Sequence[int] | None, 
        activation_name: str
    ) -> tuple[nn.Module, int]:
        """Build fusion encoder MLP."""
        if input_dim == 0:
            return nn.Identity(), 0

        if not hidden_dims:
            return nn.Identity(), input_dim

        dims = list(hidden_dims)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for idx, dim in enumerate(dims):
            layers.append(nn.Linear(prev_dim, dim))
            if idx < len(dims) - 1:
                layers.append(resolve_nn_activation(activation_name))
            prev_dim = dim
        return nn.Sequential(*layers), dims[-1]

    def _split_obs(
        self, 
        obs: torch.Tensor, 
        height_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split observations into core and height components."""
        if height_dim == 0:
            return obs, torch.empty(obs.shape[:-1] + (0,), device=obs.device, dtype=obs.dtype)
        return obs[..., :-height_dim], obs[..., -height_dim:]

    def _encode_height(self, height: torch.Tensor, height_dim: int) -> torch.Tensor:
        """Encode height observations through CNN."""
        if height_dim == 0 or height.shape[-1] == 0:
            return height[..., :0]

        batch_shape = height.shape[:-1]
        height_flat = height.reshape(-1, height_dim)
        height_map = height_flat.view(-1, 1, *self.height_map_shape)
        encoded = self.height_encoder(height_map)
        return encoded.view(*batch_shape, -1)

    def update_distribution(self, policy_output: torch.Tensor) -> None:
        """Update action distribution from policy head output."""
        mean = policy_output
        if self.noise_std_type == "scalar":
            raw = self.std.to(device=mean.device, dtype=mean.dtype)
            std_unclamped = F.softplus(raw)
            std = (std_unclamped + 1e-6).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).to(device=mean.device, dtype=mean.dtype).expand_as(mean)
        else:
            raise ValueError("Unknown standard deviation type. Should be 'scalar' or 'log'.")
        self.distribution = Normal(mean, std)

    def reset(self, dones=None):
        """Reset actor RNN hidden states."""
        self.memory_a.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        """Sample actions from policy during training."""
        # Actor: observations -> RNN -> encoder -> policy head
        rnn_out = self.memory_a(observations, masks, hidden_states)
        latent = self.student_encoder(rnn_out.squeeze(0))
        policy_output = self.student_policy_head(latent)
        
        self.update_distribution(policy_output)
        return self.distribution.sample()

    def act_inference(self, observations):
        """Get deterministic actions during inference."""
        # Actor: observations -> RNN -> encoder -> policy head
        rnn_out = self.memory_a(observations)
        latent = self.student_encoder(rnn_out.squeeze(0))
        return self.student_policy_head(latent)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Evaluate value function (non-recurrent terrain-aware critic)."""
        # For recurrent training: critic_observations shape is [time, batch, obs_dim]
        # For non-recurrent training: critic_observations shape is [batch, obs_dim]
        # Since our critic is non-recurrent, we need to flatten time and batch dimensions
        
        original_shape = critic_observations.shape
        if len(original_shape) == 3:
            # Recurrent format: [time, batch, obs_dim] -> flatten to [time*batch, obs_dim]
            time_steps, batch_size, obs_dim = original_shape
            critic_observations = critic_observations.reshape(time_steps * batch_size, obs_dim)
            need_reshape = True
        else:
            # Already in [batch, obs_dim] format
            need_reshape = False
        
        # Critic: split obs -> encode height -> fuse -> value
        core, height = self._split_obs(critic_observations, self.critic_height_dim)
        height_feat = self._encode_height(height, self.critic_height_dim)
        
        # Fuse features
        if height_feat.numel() != 0:
            fusion_input = torch.cat((core, height_feat), dim=-1)
        else:
            fusion_input = core
        
        fusion_out = self.critic_fusion_encoder(fusion_input)
        values = self.critic(fusion_out)
        
        # Reshape back to [time, batch, 1] if needed
        if need_reshape:
            values = values.reshape(time_steps, batch_size, 1)
        
        return values

    def get_hidden_states(self):
        """Get RNN hidden states (only actor has RNN)."""
        # 返回 actor 的真实 hidden states 和 critic 的空占位符
        # critic 的占位符应该匹配存储系统期望的结构
        actor_hidden = self.memory_a.hidden_states
        
        # 创建与 actor hidden states 结构相同但大小为0的占位符
        if actor_hidden is None:
            return None, None
        
        # 对于 LSTM: hidden_states 是 (h, c) 的元组
        if isinstance(actor_hidden, tuple):
            # 创建空的 (h, c) 占位符
            dummy_h = torch.zeros(0, actor_hidden[0].shape[1], actor_hidden[0].shape[2], 
                                device=actor_hidden[0].device, dtype=actor_hidden[0].dtype)
            dummy_c = torch.zeros(0, actor_hidden[1].shape[1], actor_hidden[1].shape[2],
                                device=actor_hidden[1].device, dtype=actor_hidden[1].dtype)
            critic_hidden = (dummy_h, dummy_c)
        else:
            # 对于 GRU: hidden_states 是单个 tensor
            dummy = torch.zeros(0, actor_hidden.shape[1], actor_hidden.shape[2],
                            device=actor_hidden.device, dtype=actor_hidden.dtype)
            critic_hidden = dummy
        
        return actor_hidden, critic_hidden

    def detach_hidden_states(self, dones=None):
        """Detach hidden states for actor RNN."""
        self.memory_a.detach_hidden_states(dones)

    def get_actions_log_prob(self, actions):
        """Compute log probability of actions under current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        """Get mean of action distribution."""
        return self.distribution.mean

    @property
    def action_std(self):
        """Get standard deviation of action distribution."""
        return self.distribution.stddev

    @property
    def entropy(self):
        """Get entropy of action distribution."""
        return self.distribution.entropy().sum(dim=-1)
