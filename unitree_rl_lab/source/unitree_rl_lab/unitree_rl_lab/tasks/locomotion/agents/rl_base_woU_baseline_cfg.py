# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from rl_base.isaaclab_support import RlBaseOnPolicyRunnerCfg, RlBasePpoActorCriticCfg, RlBasePpoAlgorithmCfg,RlBaseDistillationAlgorithmCfg


@configclass
class TerrainAwarePpoActorCriticCfg(RlBasePpoActorCriticCfg):
    class_name = "TerrainAwareActorCritic"
    height_obs_dim: int = 0
    height_encoder_dims = (256, 128)
    fusion_encoder_dims = (256, 128, 96)
    rnn_type = "lstm"
    rnn_hidden_dim = 256
    rnn_num_layers = 1
    noise_std_type = "scalar"


@configclass
class TerrainAwareStudentTeacherCfg(RlBasePpoActorCriticCfg):
    class_name = "TerrainAwareStudentTeacher"
    teacher_height_obs_dim: int = 0
    student_height_obs_dim: int = 0
    height_encoder_dims = (256, 128)
    fusion_encoder_dims = (256, 128, 96)
    height_cnn_channels = (16, 32)
    rnn_type = "lstm"
    rnn_hidden_dim = 256
    rnn_num_layers = 1
    noise_std_type = "scalar"
    student_encoder_hidden_dims = (256, 256)
    student_policy_hidden_dims = (256, 256, 256)
    ensemble_size: int = 1
    encoder_seeds = None


@configclass
class BasePPORunnerCfg(RlBaseOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    # obs_groups = {
    #     "policy": ["policy"],
    #     # optional: you may explicitly set critic; if omitted, resolve_obs_groups() will fill it
    #     # "critic": ["critic"],
    # }

    policy = RlBasePpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RlBasePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TerrainAwarePPORunnerCfg(BasePPORunnerCfg):
    policy = TerrainAwarePpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        height_obs_dim=187,
    )


@configclass
class TerrainAwareDistillationAlgorithmCfg(RlBaseDistillationAlgorithmCfg):
    class_name = "Distillation"
    learning_rate = 5.0e-4
    gradient_length = 15
    num_learning_epochs = 5
    max_grad_norm = 1.0
    #BCEWithLogits or Wasserstein
    # loss_type = "BCEWithLogits"
    # discriminator_cfg = {
    #     "hidden_layer_sizes": [256, 256],
    #     "learning_rate": 5.0e-4,
    #     "use_minibatch_std": False,
    #     "grad_penalty_lambda": 0.05,
    # }
    # adv_loss_weight = 1.0

    #PPO settings
    value_loss_coef=1.0
    use_clipped_value_loss=True
    clip_param=0.2
    entropy_coef=0.01
    num_mini_batches=4
    schedule="adaptive"
    gamma=0.99
    lam=0.95
    desired_kl=0.01
    
    #RL settings
    RL_loss_coef=0.0
    entropy_coef=0.0
    
    #setting4BC
    bc_loss_coef=1.0
    use_mse_loss = True

    # BC -> RL curriculum
    curriculum_enable = True
    curriculum_start_iter = 1000
    curriculum_ramp_iters = 500
    curriculum_final_rl_coef = 1.0
    curriculum_final_bc_coef = 0.0
    curriculum_noise_start = 0.744
    curriculum_noise_target = 0.744
    curriculum_noise_handover_to_rl = True
    curriculum_type = "linear"

    #deepmimic style imitation loss
    use_action_imitation_reward = False
    action_imitation_reward_coef = 1.0
    
    


@configclass
class TerrainAwareDistillationRunnerCfg(BasePPORunnerCfg):
    policy = TerrainAwareStudentTeacherCfg(
        init_noise_std=0.1,
        teacher_height_obs_dim=88,
        student_height_obs_dim=0,
        activation="elu",
    
        
    )
    algorithm = TerrainAwareDistillationAlgorithmCfg()

@configclass
class UnitreeGo2RoughPPORunnerCfg(RlBaseOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_rough"
    empirical_normalization = False
    policy = RlBasePpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RlBasePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
