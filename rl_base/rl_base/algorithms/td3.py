# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local TD3 implementation adapted from Stable-Baselines3 ideas.

This file is self-contained and does not import the external external off-policy packages package.
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


class TD3Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: tuple[int, ...] = (400, 300),
    ):
        super().__init__()
        self.backbone = build_mlp(obs_dim, action_dim, hidden_dims, activation=nn.ReLU)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action_unit = torch.tanh(self.backbone(obs))
        return self.action_bias + self.action_scale * action_unit


class TD3Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (400, 300)):
        super().__init__()
        input_dim = obs_dim + action_dim
        self.q1 = build_mlp(input_dim, 1, hidden_dims, activation=nn.ReLU)
        self.q2 = build_mlp(input_dim, 1, hidden_dims, activation=nn.ReLU)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


class TD3:
    def __init__(
        self,
        policy: str,
        env: gym.Env,
        learning_rate: float = 1.0e-3,
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
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        exploration_noise_std: float = 0.1,
    ):
        if policy != "MlpPolicy":
            raise ValueError(f"Only MlpPolicy is supported in local TD3. Got: {policy}")

        obs_space = getattr(env, "single_observation_space", env.observation_space)
        action_space = getattr(env, "single_action_space", env.action_space)

        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"TD3 requires Box observation space. Got: {type(obs_space)}")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"TD3 requires Box action space. Got: {type(action_space)}")

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
        self.policy_delay = int(policy_delay)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.exploration_noise_std = float(exploration_noise_std)

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
        self.action_scale_np = (action_high - action_low) / 2.0

        self.actor = TD3Actor(self.obs_dim, self.action_dim, action_low, action_high).to(self.device)
        self.actor_target = TD3Actor(self.obs_dim, self.action_dim, action_low, action_high).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = TD3Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(obs_shape, action_shape, self.buffer_size, self.device)
        self.logger = TensorboardLogger(tensorboard_log)

        # total collected transitions across all envs
        self.total_timesteps = 0
        # number of vector env steps (each step advances n_envs transitions)
        self.total_env_steps = 0
        self.total_updates = 0

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

    def _apply_terminal_obs_and_timeouts(
        self,
        infos,
        next_obs_batch: np.ndarray,
        done_batch: np.ndarray,
        timeout_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align with common timeout and terminal-observation info fields when available."""
        if infos is None:
            return next_obs_batch, timeout_batch

        if isinstance(infos, (list, tuple)):
            for idx in range(min(self.n_envs, len(infos))):
                if not done_batch[idx]:
                    continue
                info_i = infos[idx]
                if not isinstance(info_i, dict):
                    continue
                if "TimeLimit.truncated" in info_i:
                    timeout_batch[idx] = bool(info_i["TimeLimit.truncated"])
                terminal_obs = info_i.get("terminal_observation")
                if terminal_obs is not None:
                    terminal_obs_arr = np.asarray(terminal_obs, dtype=np.float32).reshape(self.obs_shape)
                    next_obs_batch[idx] = terminal_obs_arr
            return next_obs_batch, timeout_batch

        if isinstance(infos, dict):
            if "TimeLimit.truncated" in infos:
                timeout_batch = np.asarray(infos["TimeLimit.truncated"], dtype=bool).reshape(self.n_envs)

            terminal_obs_batch = infos.get("terminal_observation")
            if terminal_obs_batch is not None:
                if isinstance(terminal_obs_batch, np.ndarray) and terminal_obs_batch.shape[0] == self.n_envs:
                    iterable = [terminal_obs_batch[i] for i in range(self.n_envs)]
                elif isinstance(terminal_obs_batch, (list, tuple)) and len(terminal_obs_batch) == self.n_envs:
                    iterable = terminal_obs_batch
                else:
                    iterable = None

                if iterable is not None:
                    for idx in range(self.n_envs):
                        if not done_batch[idx]:
                            continue
                        term_obs = iterable[idx]
                        if term_obs is None:
                            continue
                        term_obs_arr = np.asarray(term_obs, dtype=np.float32).reshape(self.obs_shape)
                        next_obs_batch[idx] = term_obs_arr

        return next_obs_batch, timeout_batch

    def _predict_actions(self, obs_batch: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs_batch.reshape(self.n_envs, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_t = self.actor(obs_t)
        action = action_t.cpu().numpy()
        if not deterministic:
            noise = np.random.normal(0.0, self.exploration_noise_std * self.action_scale_np, size=action.shape)
            action = action + noise.astype(np.float32)
        action = np.clip(action, self.action_low_np, self.action_high_np)
        return action.astype(np.float32)

    def _train_step(self) -> dict[str, float]:
        batch = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            noise = torch.randn_like(batch.actions.reshape(self.batch_size, -1)) * (
                self.target_policy_noise * self.actor.action_scale.view(1, -1)
            )
            noise_clip = self.target_noise_clip
            noise = torch.clamp(noise, -noise_clip, noise_clip)

            next_obs = batch.next_observations.reshape(self.batch_size, -1)
            next_actions = self.actor_target(next_obs) + noise
            next_actions = torch.max(next_actions, self.actor.action_low.view(1, -1))
            next_actions = torch.min(next_actions, self.actor.action_high.view(1, -1))

            next_q1, next_q2 = self.critic_target(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = batch.rewards + (1.0 - batch.dones) * self.gamma * next_q

        observations = batch.observations.reshape(self.batch_size, -1)
        actions = batch.actions.reshape(self.batch_size, -1)

        current_q1, current_q2 = self.critic(observations, actions)
        critic_loss = 0.5 * (F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        update_index = self.total_updates + 1
        if update_index % self.policy_delay == 0:
            pred_actions = self.actor(observations)
            actor_loss = -self.critic.q1_forward(observations, pred_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            polyak_update(self.critic, self.critic_target, self.tau)
            polyak_update(self.actor, self.actor_target, self.tau)
            actor_loss_value = float(actor_loss.item())

        self.total_updates = update_index
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
        }

    def learn(self, total_timesteps: int, log_interval: int = 10, tb_log_name: str = "TD3") -> "TD3":
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
            next_obs_batch, timeout_batch = self._apply_terminal_obs_and_timeouts(
                infos=infos,
                next_obs_batch=next_obs_batch,
                done_batch=done_batch,
                timeout_batch=timeout_batch,
            )

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

            post_step_timesteps = self.total_timesteps + self.n_envs
            if post_step_timesteps > self.learning_starts and (self.total_env_steps + 1) % self.train_freq == 0:
                if len(self.replay_buffer) >= self.batch_size:
                    gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else self.train_freq * self.n_envs
                    if gradient_steps > 0:
                        for _ in range(gradient_steps):
                            metrics = self._train_step()
                        self.logger.add_scalar(f"{tb_log_name}/critic_loss", metrics["critic_loss"], self.total_timesteps)
                        self.logger.add_scalar(f"{tb_log_name}/actor_loss", metrics["actor_loss"], self.total_timesteps)

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
                    f"[TD3] timesteps={self.total_timesteps} "
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
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": {
                "learning_rate": self.learning_rate,
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "batch_size": self.batch_size,
                "tau": self.tau,
                "gamma": self.gamma,
                "train_freq": self.train_freq,
                "gradient_steps": self.gradient_steps,
                "policy_delay": self.policy_delay,
                "target_policy_noise": self.target_policy_noise,
                "target_noise_clip": self.target_noise_clip,
                "exploration_noise_std": self.exploration_noise_std,
                "total_timesteps": self.total_timesteps,
                "total_env_steps": self.total_env_steps,
                "total_updates": self.total_updates,
            },
        }
        torch.save(payload, path)

    def save_replay_buffer(self, path: str | Path) -> None:
        dump_pickle(path, self.replay_buffer.state_dict())
