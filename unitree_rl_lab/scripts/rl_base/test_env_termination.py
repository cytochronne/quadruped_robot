#!/usr/bin/env python
"""Test script to check episode termination behavior."""

import gymnasium as gym
import numpy as np
import torch

def main():
    # Make the environment
    env = gym.make("Unitree-Go2-Velocity-lab-Rough-Env-v0", num_envs=8, headless=True)

    # Reset
    obs, info = env.reset(seed=42)

    print(f"Environment type: {type(env)}")
    print(f"Environment unwrapped type: {type(env.unwrapped)}")

    # Check max episode length
    if hasattr(env.unwrapped, "max_episode_length"):
        print(f"max_episode_length: {env.unwrapped.max_episode_length}")

    if hasattr(env.unwrapped, "episode_length_s"):
        print(f"episode_length_s: {env.unwrapped.episode_length_s}")

    # Check termination manager
    if hasattr(env.unwrapped, "termination_manager"):
        print(f"Termination manager: {env.unwrapped.termination_manager}")
        print(f"  Active terms: {env.unwrapped.termination_manager.active_terms}")

    # Run for 1600 steps (should see episodes ending)
    print("\nRunning for 1600 steps...")
    completed_episodes = 0
    max_ep_len_seen = 0

    for step in range(1600):
        action = env.action_space.sample()
        result = env.step(action)

        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = np.zeros_like(done)
        elif len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated | truncated
        else:
            print(f"Unexpected result length: {len(result)}")
            break

        # Check episode_length_buf
        if hasattr(env.unwrapped, "episode_length_buf"):
            ep_len = env.unwrapped.episode_length_buf.cpu().numpy()
            max_ep_len_seen = max(max_ep_len_seen, int(np.max(ep_len)))

        # Count completed episodes
        done_count = int(np.sum(done))
        completed_episodes += done_count

        # Print debug info at specific steps
        if step in [0, 1, 1498, 1499, 1500, 1501, 1502]:
            print(f"\nStep {step}:")
            print(f"  episode_length_buf: {env.unwrapped.episode_length_buf.cpu().numpy()[:5]}")
            if hasattr(env.unwrapped, "termination_manager"):
                to = env.unwrapped.termination_manager.time_outs.cpu().numpy()
                ter = env.unwrapped.termination_manager.terminated.cpu().numpy()
                print(f"  time_outs: {to[:5]}")
                print(f"  terminated: {ter[:5]}")
            print(f"  done signals: {done[:5]}")
            print(f"  sum(done): {np.sum(done)}")

    print(f"\n=== Results ===")
    print(f"Total completed episodes: {completed_episodes}")
    print(f"Max episode length seen: {max_ep_len_seen}")

    env.close()

if __name__ == "__main__":
    main()
