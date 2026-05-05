# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for local off-policy algorithms (TD3/SAC).

This module provides a lightweight subset inspired by Stable-Baselines3 internals,
implemented locally to avoid external SB3 runtime dependency.
"""

from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Iterable[int],
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation())
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


@dataclass
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...], capacity: int, device: torch.device):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = int(capacity)
        self.device = device

        self.observations = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        self.observations[self.pos] = np.asarray(obs, dtype=np.float32)
        self.actions[self.pos] = np.asarray(action, dtype=np.float32)
        self.rewards[self.pos, 0] = float(reward)
        self.next_observations[self.pos] = np.asarray(next_obs, dtype=np.float32)
        self.dones[self.pos, 0] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, observations, actions, rewards, next_observations, dones) -> None:
        obs_batch = np.asarray(observations, dtype=np.float32)
        action_batch = np.asarray(actions, dtype=np.float32)
        reward_batch = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        next_obs_batch = np.asarray(next_observations, dtype=np.float32)
        done_batch = np.asarray(dones, dtype=np.float32).reshape(-1, 1)

        if obs_batch.ndim == len(self.obs_shape):
            obs_batch = np.expand_dims(obs_batch, axis=0)
            action_batch = np.expand_dims(action_batch, axis=0)
            reward_batch = reward_batch.reshape(1, 1)
            next_obs_batch = np.expand_dims(next_obs_batch, axis=0)
            done_batch = done_batch.reshape(1, 1)

        batch_size = int(obs_batch.shape[0])
        if batch_size <= 0:
            return

        if batch_size > self.capacity:
            obs_batch = obs_batch[-self.capacity :]
            action_batch = action_batch[-self.capacity :]
            reward_batch = reward_batch[-self.capacity :]
            next_obs_batch = next_obs_batch[-self.capacity :]
            done_batch = done_batch[-self.capacity :]
            batch_size = self.capacity

        indices = (np.arange(batch_size) + self.pos) % self.capacity
        self.observations[indices] = obs_batch
        self.actions[indices] = action_batch
        self.rewards[indices] = reward_batch
        self.next_observations[indices] = next_obs_batch
        self.dones[indices] = done_batch

        self.pos = (self.pos + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        indices = np.random.randint(0, self.size, size=int(batch_size))
        return ReplayBatch(
            observations=torch.as_tensor(self.observations[indices], device=self.device),
            actions=torch.as_tensor(self.actions[indices], device=self.device),
            rewards=torch.as_tensor(self.rewards[indices], device=self.device),
            next_observations=torch.as_tensor(self.next_observations[indices], device=self.device),
            dones=torch.as_tensor(self.dones[indices], device=self.device),
        )

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict:
        return {
            "obs_shape": self.obs_shape,
            "action_shape": self.action_shape,
            "capacity": self.capacity,
            "pos": self.pos,
            "size": self.size,
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "dones": self.dones,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.pos = int(state_dict["pos"])
        self.size = int(state_dict["size"])
        self.observations = state_dict["observations"]
        self.actions = state_dict["actions"]
        self.rewards = state_dict["rewards"]
        self.next_observations = state_dict["next_observations"]
        self.dones = state_dict["dones"]


class TensorboardLogger:
    def __init__(self, log_dir: str | None):
        self.writer = None
        if log_dir is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        except Exception:
            self.writer = None

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), int(step))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def ensure_pt_path(path: str | os.PathLike) -> Path:
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def dump_pickle(path: str | os.PathLike, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)
