import gymnasium as gym


gym.register(
    id="Unitree-Go2-Velocity-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:UnitreeGo2RoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:UnitreeGo2RoughEnvCfg_PLAY",
        "rl_base_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rl_base_ppo_cfg:TerrainAwarePPORunnerCfg",
    },
)
