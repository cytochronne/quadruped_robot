# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .terrain_aware_actor_critic import TerrainAwareActorCritic
from rsl_rl_woUncertainty.networks import Memory
from rsl_rl_woUncertainty.utils import resolve_nn_activation


class TerrainAwareStudentTeacher(nn.Module):
    """Student-teacher module with a terrain-aware teacher and recurrent student."""

    is_recurrent = True

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        *,
        teacher_height_obs_dim: int,
        student_height_obs_dim: int = 0,
        fusion_encoder_dims: Sequence[int] | None = (256, 128, 96),
        height_cnn_channels: Sequence[int] = (16, 32),
        height_map_shape: Tuple[int, int] | None = None,
        height_encoder_dims: Sequence[int] | None = None,
        teacher_actor_hidden_dims: Sequence[int] = (512, 256, 128),
        teacher_critic_hidden_dims: Sequence[int] = (512, 256, 128),
        student_encoder_hidden_dims: Sequence[int] | None = None,
        student_policy_hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(
                "TerrainAwareStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )

        if teacher_height_obs_dim < 0:
            raise ValueError("teacher_height_obs_dim must be non-negative.")
        if student_height_obs_dim < 0:
            raise ValueError("student_height_obs_dim must be non-negative.")

        if teacher_height_obs_dim == 0:
            self.teacher_height_dim = 0
        elif teacher_height_obs_dim >= num_teacher_obs:
            print(
                "[WARN] Requested teacher_height_obs_dim exceeds teacher observations. "
                "Falling back to zero height features for the teacher."
            )
            self.teacher_height_dim = 0
        else:
            self.teacher_height_dim = teacher_height_obs_dim

        if student_height_obs_dim == 0:
            self.student_height_dim = 0
        elif student_height_obs_dim >= num_student_obs:
            print(
                "[WARN] Requested student_height_obs_dim exceeds student observations. "
                "Falling back to zero height features for the student."
            )
            self.student_height_dim = 0
        else:
            self.student_height_dim = student_height_obs_dim

        self.student_core_dim = num_student_obs - self.student_height_dim
        if self.student_core_dim <= 0:
            raise ValueError("Student core observation dimension must be positive.")

        activation_name = activation
        self.loaded_teacher = False

        # Teacher model (kept in eval mode, gradients disabled)
        self.teacher = TerrainAwareActorCritic(
            num_actor_obs=num_teacher_obs,
            num_critic_obs=num_teacher_obs,
            num_actions=num_actions,
            height_obs_dim=self.teacher_height_dim,
            actor_hidden_dims=teacher_actor_hidden_dims,
            critic_hidden_dims=teacher_critic_hidden_dims,
            fusion_encoder_dims=fusion_encoder_dims,
            height_cnn_channels=height_cnn_channels,
            height_map_shape=height_map_shape,
            activation=activation_name,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            height_encoder_dims=height_encoder_dims,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            build_critic=True,
        )
        # Don't set eval or disable gradients here - let Distillation class control this

        self.teacher_latent_dim = self.teacher.actor_fusion_dim

        # Student components: recurrent encoder -> latent -> policy head
        self.memory_s = Memory(
            self.student_core_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        # Single deterministic encoder (no ensemble, no uncertainty)
        self.student_encoder = self._build_mlp(
            rnn_hidden_dim,
            student_encoder_hidden_dims,
            self.teacher_latent_dim,
            activation_name,
        )

        self.student_policy_head = self._build_mlp(
            self.teacher_latent_dim,
            student_policy_hidden_dims,
            num_actions,
            activation_name,
        )
        
        # Expose student modules for optimizer/gradient utilities
        self.student = nn.ModuleDict({"encoder": self.student_encoder, "policy": self.student_policy_head})

        print(f"Student recurrent encoder: {self.memory_s}")
        print(f"Student latent encoder: {self.student_encoder}")
        print(f"Student policy head: {self.student_policy_head}")

        # Action noise configuration for the student policy
        self.noise_std_type = noise_std_type
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
    def _split_obs(obs: torch.Tensor, height_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        if height_dim == 0:
            empty = torch.empty(obs.shape[:-1] + (0,), device=obs.device, dtype=obs.dtype)
            return obs, empty
        return obs[..., :-height_dim], obs[..., -height_dim:]

    def _student_core_features(self, observations: torch.Tensor, masks=None, hidden_states=None) -> torch.Tensor:
        core_obs, _ = self._split_obs(observations, self.student_height_dim)
        core_features = self.memory_s(core_obs, masks=masks, hidden_states=hidden_states)
        return core_features.squeeze(0)

    def update_distribution(self, features: torch.Tensor) -> None:
        mean = self.student_policy_head(features)
        if self.noise_std_type == "scalar":
            raw = self.std.to(device=mean.device, dtype=mean.dtype)
            std = (F.softplus(raw) + 1.0e-6).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).to(device=mean.device, dtype=mean.dtype).expand_as(mean)
        else:
            raise ValueError("Unknown standard deviation type. Should be 'scalar' or 'log'.")
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, masks=None, hidden_states=None) -> torch.Tensor:
        latent = self.get_student_latent(observations, masks=masks, hidden_states=hidden_states)
        self.update_distribution(latent)
        return self.distribution.sample()

    def act_inference(
        self, observations: torch.Tensor, *, return_latent: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        latent = self.get_student_latent(observations)
        actions_mean = self.student_policy_head(latent)
        if return_latent:
            return actions_mean, latent
        return actions_mean

    #teacher policy inference
    def evaluate(self, teacher_observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher.act_inference(teacher_observations)

    #get teacher's latent features
    def evaluate_feature(self, teacher_observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher._prepare_features(
                teacher_observations,
                self.teacher.actor_height_dim,
                self.teacher.actor_fusion_encoder,
            )

    def get_student_latent(self, observations: torch.Tensor, masks=None, hidden_states=None) -> torch.Tensor:
        """Get deterministic latent features from student encoder."""
        core = self._student_core_features(observations, masks=masks, hidden_states=hidden_states)
        latent = self.student_encoder(core)
        return latent

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            hidden_states = (None,)
        self.memory_s.reset(dones, hidden_states[0])

    def get_hidden_states(self):
        return self.memory_s.hidden_states, None

    def detach_hidden_states(self, dones=None):
        self.memory_s.detach_hidden_states(dones)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def load_state_dict(self, state_dict, strict: bool = False):
        
        def _drop_critic_head(sd: dict, prefix: str = "critic.") -> dict:
            """Drop critic head weights so they stay randomly initialized.

            Args:
                sd: state dict to filter.
                prefix: key prefix that identifies critic head params
                    (e.g. ``"critic."`` for raw teacher, ``"teacher.critic."`` for
                    full student-teacher checkpoint).
            """
            cleaned = {}
            dropped = 0
            for k, v in sd.items():
                if k.startswith(prefix):
                    dropped += 1
                    continue
                cleaned[k] = v
            if dropped > 0:
                print(f"[INFO] Dropped {dropped} critic head params (prefix='{prefix}') "
                      "to keep it randomly initialized.")
            return cleaned

        # Check if we have teacher-prefixed keys
        has_teacher_prefix = any(k.startswith("teacher.") for k in state_dict.keys())

        if has_teacher_prefix:
            # Resume case: full student-teacher checkpoint → keep critic head as-is
            super().load_state_dict(state_dict, strict=strict)
            print("[INFO] Resumed training: loaded teacher + student (critic head preserved).")
            self.loaded_teacher = True
            return True
        else:
            # Raw teacher checkpoint → drop critic head so it gets freshly initialized
            filtered = {k: v for k, v in state_dict.items()
                        if not (k.startswith("student") or k.startswith("memory_s"))}
            filtered = _drop_critic_head(filtered, prefix="critic.")
            try:
                self.teacher.load_state_dict(filtered, strict=False)
            except Exception:
                pass
            # Also try via super() with "teacher." prefix added
            remapped = {"teacher." + k: v for k, v in filtered.items()}
            super().load_state_dict(remapped, strict=False)
            print("[INFO] Loaded raw teacher checkpoint (critic head randomly initialized).")

        self.loaded_teacher = True
        return False
