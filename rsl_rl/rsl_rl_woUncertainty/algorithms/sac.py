# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local SAC implementation adapted from Stable-Baselines3 ideas.

This file is self-contained and does not import the external stable_baselines3 package.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .offpolicy_common import (
    ReplayBuffer,
    TensorboardLogger,
    build_mlp,
    dump_pickle,
    ensure_pt_path,
    polyak_update,
    resolve_device,
    set_global_seed,
)

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SACActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("SAC actor requires at least one hidden layer.")

        self.backbone = build_mlp(obs_dim, hidden_dims[-1], hidden_dims[:-1], activation=nn.ReLU)
        self.mu = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)

        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2.0)

    def _distribution_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.backbone(obs)
        mean = self.mu(latent)
        log_std = torch.clamp(self.log_std(latent), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action, _ = self.action_log_prob(obs, deterministic=deterministic)
        return action

    def action_log_prob(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._distribution_params(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            raw_action = mean
        else:
            raw_action = dist.rsample()

        squashed_action = torch.tanh(raw_action)
        action = self.action_bias + self.action_scale * squashed_action

        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        correction = torch.log(torch.clamp(1.0 - squashed_action.pow(2), min=1.0e-6)).sum(dim=-1, keepdim=True)
        scale_correction = torch.log(torch.clamp(self.action_scale, min=1.0e-6)).sum()
        log_prob = log_prob - correction - scale_correction

        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        input_dim = obs_dim + action_dim
        self.q1 = build_mlp(input_dim, 1, hidden_dims, activation=nn.ReLU)
        self.q2 = build_mlp(input_dim, 1, hidden_dims, activation=nn.ReLU)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class SAC:
    def __init__(
        self,
        policy: str,
        env: gym.Env,
        learning_rate: float = 3.0e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        tensorboard_log: str | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: str | torch.device | None = "auto",
        ent_coef: str | float = "auto",
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
    ):
        if policy != "MlpPolicy":
            raise ValueError(f"Only MlpPolicy is supported in local SAC. Got: {policy}")

        obs_space = getattr(env, "single_observation_space", env.observation_space)
        action_space = getattr(env, "single_action_space", env.action_space)

        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"SAC requires Box observation space. Got: {type(obs_space)}")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"SAC requires Box action space. Got: {type(action_space)}")

        self.env = env
        self.device = resolve_device(device)
        self.verbose = int(verbose)
        self.seed = seed
        set_global_seed(seed)

        self.learning_rate = float(learning_rate)
        self.buffer_size = int(buffer_size)
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.target_update_interval = int(target_update_interval)

        self.ent_coef_setting = ent_coef
        self.target_entropy_setting = target_entropy

        self.n_envs = int(getattr(env, "num_envs", 1))

        obs_shape = tuple(int(x) for x in obs_space.shape)
        action_shape = tuple(int(x) for x in action_space.shape)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(np.prod(action_shape))

        action_low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        self.action_low_np = action_low
        self.action_high_np = action_high

        self.actor = SACActor(self.obs_dim, self.action_dim, action_low, action_high).to(self.device)
        self.critic = SACCritic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = SACCritic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        if self.target_entropy_setting == "auto":
            self.target_entropy = float(-self.action_dim)
        else:
            self.target_entropy = float(self.target_entropy_setting)

        self.log_ent_coef: torch.Tensor | None = None
        self.ent_coef_optimizer: torch.optim.Optimizer | None = None
        self.ent_coef_tensor: torch.Tensor | None = None
        self._setup_entropy_coef()

        self.replay_buffer = ReplayBuffer(obs_shape, action_shape, self.buffer_size, self.device)
        self.logger = TensorboardLogger(tensorboard_log)

        self.total_timesteps = 0
        self.total_env_steps = 0
        self.total_updates = 0

    def _setup_entropy_coef(self) -> None:
        if isinstance(self.ent_coef_setting, str) and self.ent_coef_setting.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef_setting:
                init_value = float(self.ent_coef_setting.split("_")[1])
            self.log_ent_coef = torch.log(
                torch.ones(1, device=self.device, dtype=torch.float32) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
            self.ent_coef_tensor = None
        else:
            self.log_ent_coef = None
            self.ent_coef_optimizer = None
            self.ent_coef_tensor = torch.tensor(float(self.ent_coef_setting), dtype=torch.float32, device=self.device)

    def _ensure_batched_obs(self, obs) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == len(self.obs_shape):
            obs_arr = np.expand_dims(obs_arr, axis=0)
        if obs_arr.shape[0] != self.n_envs:
            obs_arr = obs_arr.reshape(self.n_envs, *self.obs_shape)
        return obs_arr

    def _reshape_action_batch(self, action_batch_flat: np.ndarray) -> np.ndarray:
        return np.asarray(action_batch_flat, dtype=np.float32).reshape(self.n_envs, *self.action_shape)

    def _sample_random_actions(self) -> np.ndarray:
        return np.random.uniform(self.action_low_np, self.action_high_np, size=(self.n_envs, self.action_dim)).astype(
            np.float32
        )

    def _apply_terminal_obs(self, infos, next_obs_batch: np.ndarray, done_batch: np.ndarray) -> np.ndarray:
        if not isinstance(infos, dict):
            return next_obs_batch
        terminal_obs_batch = infos.get("terminal_observation")
        if terminal_obs_batch is None:
            return next_obs_batch
        if isinstance(terminal_obs_batch, np.ndarray) and terminal_obs_batch.shape[0] == self.n_envs:
            iterable = [terminal_obs_batch[i] for i in range(self.n_envs)]
        elif isinstance(terminal_obs_batch, (list, tuple)) and len(terminal_obs_batch) == self.n_envs:
            iterable = terminal_obs_batch
        else:
            return next_obs_batch
        for idx in range(self.n_envs):
            if done_batch[idx] and iterable[idx] is not None:
                next_obs_batch[idx] = np.asarray(iterable[idx], dtype=np.float32).reshape(self.obs_shape)
        return next_obs_batch

    def _predict_actions(self, obs_batch: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs_batch.reshape(self.n_envs, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_t, _ = self.actor.action_log_prob(obs_t, deterministic=deterministic)
        action = action_t.cpu().numpy()
        action = np.clip(action, self.action_low_np, self.action_high_np)
        return action.astype(np.float32)

    def _current_ent_coef(self) -> torch.Tensor:
        if self.log_ent_coef is not None:
            return torch.exp(self.log_ent_coef)
        assert self.ent_coef_tensor is not None
        return self.ent_coef_tensor

    def _train_step(self) -> dict[str, float]:
        batch = self.replay_buffer.sample(self.batch_size)

        observations = batch.observations.reshape(self.batch_size, -1)
        actions = batch.actions.reshape(self.batch_size, -1)
        next_observations = batch.next_observations.reshape(self.batch_size, -1)

        actions_pi, log_prob = self.actor.action_log_prob(observations, deterministic=False)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            ent_coef = torch.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
        else:
            ent_coef = self._current_ent_coef()

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.action_log_prob(next_observations, deterministic=False)
            next_q1, next_q2 = self.critic_target(next_observations, next_actions)
            next_q = torch.min(next_q1, next_q2) - ent_coef * next_log_prob
            target_q = batch.rewards + (1.0 - batch.dones) * self.gamma * next_q

        current_q1, current_q2 = self.critic(observations, actions)
        critic_loss = 0.5 * (F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        q1_pi, q2_pi = self.critic(observations, actions_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (ent_coef * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.total_updates % self.target_update_interval == 0:
            polyak_update(self.critic, self.critic_target, self.tau)

        self.total_updates += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "ent_coef": float(ent_coef.item()),
            "ent_coef_loss": float(ent_coef_loss.item()) if ent_coef_loss is not None else 0.0,
        }

    def learn(self, total_timesteps: int, log_interval: int = 10, tb_log_name: str = "SAC") -> "SAC":
        # Only reset at the start of training, not on subsequent calls
        if self.total_timesteps == 0:
            if self.seed is not None:
                obs, _ = self.env.reset(seed=self.seed)
            else:
                obs, _ = self.env.reset()
            self.obs_batch = self._ensure_batched_obs(obs)
            # Initialize episode tracking variables
            self.episode_rewards = np.zeros(self.n_envs, dtype=np.float64)
            self.episode_lengths = np.zeros(self.n_envs, dtype=np.int64)
            self.ep_count = 0
            self.ep_reward_window: deque[float] = deque(maxlen=100)

        obs_batch = self.obs_batch

        while self.total_timesteps < int(total_timesteps):
            if self.total_timesteps < self.learning_starts:
                action_batch_flat = self._sample_random_actions()
            else:
                action_batch_flat = self._predict_actions(obs_batch, deterministic=False)

            env_action_batch = self._reshape_action_batch(action_batch_flat)
            next_obs, reward, terminated, truncated, infos = self.env.step(env_action_batch)
            next_obs_batch = self._ensure_batched_obs(next_obs)

            reward_batch = np.asarray(reward, dtype=np.float32).reshape(self.n_envs)
            terminated_batch = np.asarray(terminated, dtype=bool).reshape(self.n_envs)
            truncated_batch = np.asarray(truncated, dtype=bool).reshape(self.n_envs)
            done_batch = np.logical_or(terminated_batch, truncated_batch)
            timeout_batch = np.logical_and(truncated_batch, np.logical_not(terminated_batch))
            next_obs_batch = self._apply_terminal_obs(infos, next_obs_batch, done_batch)

            self.replay_buffer.add_batch(
                observations=obs_batch,
                actions=action_batch_flat.reshape(self.n_envs, *self.action_shape),
                rewards=reward_batch,
                next_observations=next_obs_batch,
                dones=done_batch,
                timeouts=timeout_batch,
            )

            self.obs_batch = next_obs_batch
            obs_batch = next_obs_batch
            self.episode_rewards += reward_batch
            self.episode_lengths += 1

            if self.total_timesteps >= self.learning_starts and (self.total_env_steps + 1) % self.train_freq == 0:
                if len(self.replay_buffer) >= self.batch_size:
                    gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else self.train_freq * self.n_envs
                    if gradient_steps > 0:
                        for _ in range(gradient_steps):
                            metrics = self._train_step()
                        self.logger.add_scalar(f"{tb_log_name}/critic_loss", metrics["critic_loss"], self.total_timesteps)
                        self.logger.add_scalar(f"{tb_log_name}/actor_loss", metrics["actor_loss"], self.total_timesteps)
                        self.logger.add_scalar(f"{tb_log_name}/ent_coef", metrics["ent_coef"], self.total_timesteps)
                        self.logger.add_scalar(
                            f"{tb_log_name}/ent_coef_loss", metrics["ent_coef_loss"], self.total_timesteps
                        )

            finished_indices = np.nonzero(done_batch)[0]
            for idx in finished_indices.tolist():
                self.ep_count += 1
                ep_reward = float(self.episode_rewards[idx])
                ep_len = int(self.episode_lengths[idx])
                self.ep_reward_window.append(ep_reward)
                self.logger.add_scalar(f"{tb_log_name}/episode_reward", ep_reward, self.total_timesteps)
                self.logger.add_scalar(f"{tb_log_name}/episode_length", ep_len, self.total_timesteps)
                self.episode_rewards[idx] = 0.0
                self.episode_lengths[idx] = 0

            if log_interval > 0 and self.ep_count > 0 and self.ep_count % log_interval == 0 and len(self.ep_reward_window) > 0:
                print(
                    f"[SAC] timesteps={self.total_timesteps} "
                    f"episodes={self.ep_count} "
                    f"mean_reward_100={float(np.mean(self.ep_reward_window)):.3f}"
                )

            self.total_env_steps += 1
            self.total_timesteps += self.n_envs

        self.logger.close()
        return self

    def save(self, save_path: str | Path) -> None:
        path = ensure_pt_path(save_path)
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_ent_coef": self.log_ent_coef.detach().cpu() if self.log_ent_coef is not None else None,
            "ent_coef_optimizer": self.ent_coef_optimizer.state_dict() if self.ent_coef_optimizer is not None else None,
            "ent_coef_tensor": self.ent_coef_tensor.detach().cpu() if self.ent_coef_tensor is not None else None,
            "config": {
                "learning_rate": self.learning_rate,
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "batch_size": self.batch_size,
                "tau": self.tau,
                "gamma": self.gamma,
                "train_freq": self.train_freq,
                "gradient_steps": self.gradient_steps,
                "target_update_interval": self.target_update_interval,
                "ent_coef_setting": self.ent_coef_setting,
                "target_entropy_setting": self.target_entropy_setting,
                "target_entropy": self.target_entropy,
                "total_timesteps": self.total_timesteps,
                "total_env_steps": self.total_env_steps,
                "total_updates": self.total_updates,
            },
        }
        torch.save(payload, path)

    def save_replay_buffer(self, path: str | Path) -> None:
        dump_pickle(path, self.replay_buffer.state_dict())
