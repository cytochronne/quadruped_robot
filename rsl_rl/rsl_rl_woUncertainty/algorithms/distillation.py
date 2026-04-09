# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# rsl-rl
from rsl_rl_woUncertainty.modules import Discriminator, StudentTeacher, StudentTeacherRecurrent
from rsl_rl_woUncertainty.storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        bc_loss_coef=1.0,
        RL_loss_coef=1.0,
        use_action_imitation_reward=False,
        action_imitation_reward_coef=1.0,
        use_mse_loss=True,
        device="cpu",
        uncertainty_abs_coef = 1.0,
        uncertainty_delta_coef = 2.0,
        uncertainty_delta_threshold = 0.03,
        uncertainty_max = 1.0,
        uncertainty_min = 0.0,
        uncertainty_warmup_iters: int = 500,
        uncertainty_ema_beta: float = 0.01,
        uncertainty_eps: float = 1e-6,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class
        
        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        
        # Collect all student parameters (encoder + policy head)
        self.student_parameters: list[nn.Parameter] = []
        
        # 1. Encoder parameters
        if hasattr(self.policy, "memory_s"):
            self.student_parameters.extend(p for p in self.policy.memory_s.parameters())
        student_module = getattr(self.policy, "student", None)
        if student_module is not None and hasattr(student_module, "get"):
            encoder_module = student_module.get("encoder")
            if encoder_module is not None:
                self.student_parameters.extend(p for p in encoder_module.parameters())
        elif hasattr(self.policy, "student_encoder"):
            self.student_parameters.extend(p for p in self.policy.student_encoder.parameters())

        # 2. Policy head parameters
        policy_head = getattr(self.policy, "student_policy_head", None)
        if policy_head is not None:
            # Ensure they are trainable
            for param in policy_head.parameters():
                param.requires_grad_(True)
            self.student_parameters.extend(p for p in policy_head.parameters())

        # 2b. Action noise std parameter (must be optimized for exploration decay)
        if hasattr(self.policy, "std"):
            self.student_parameters.append(self.policy.std)
        elif hasattr(self.policy, "log_std"):
            self.student_parameters.append(self.policy.log_std)

        # 3. Teacher Critic (Complete parameter collection)
        self.teacher_critic_parameters: list[nn.Parameter] = []
        if hasattr(self.policy, "teacher") and hasattr(self.policy.teacher, "critic"):
             critic_modules = []
             
             # Critic head
             if self.policy.teacher.critic is not None:
                 critic_modules.append(self.policy.teacher.critic)
             
             # Critic fusion encoder
             if hasattr(self.policy.teacher, "critic_fusion_encoder"):
                 critic_modules.append(self.policy.teacher.critic_fusion_encoder)
             
             # Height encoder (shared between actor and critic)
             if hasattr(self.policy.teacher, "height_encoder") and self.policy.teacher.critic_height_dim > 0:
                 critic_modules.append(self.policy.teacher.height_encoder)
             
             # Enable gradients and collect parameters
             for module in critic_modules:
                 module.train()  # Set to training mode
                 for param in module.parameters():
                     param.requires_grad_(True)
                 self.teacher_critic_parameters.extend(p for p in module.parameters())
             
             # Disable gradients for actor
             if hasattr(self.policy.teacher, "actor"):
                 self.policy.teacher.actor.eval()
                 for param in self.policy.teacher.actor.parameters():
                     param.requires_grad_(False)
             
             if hasattr(self.policy.teacher, "actor_fusion_encoder"):
                 self.policy.teacher.actor_fusion_encoder.eval()
                 for param in self.policy.teacher.actor_fusion_encoder.parameters():
                     param.requires_grad_(False)
             
             print(f"[INFO] Collected {len(self.teacher_critic_parameters)} critic parameters for training")
             print(f"[INFO] Critic modules: {[type(m).__name__ for m in critic_modules]}")

        if not self.student_parameters:
            raise ValueError("No student parameters found to optimise.")

        self.optimizer = optim.Adam(self.student_parameters, lr=learning_rate)
        if self.teacher_critic_parameters:
            self.critic_optimizer = optim.Adam(self.teacher_critic_parameters, lr=learning_rate)
        else:
            self.critic_optimizer = None
        self.transition = RolloutStorage.Transition()

        # PPO / RL parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.bc_loss_coef = bc_loss_coef
        self.RL_loss_coef = RL_loss_coef
        self.use_action_imitation_reward = use_action_imitation_reward
        self.action_imitation_reward_coef = action_imitation_reward_coef
        self.use_mse_loss = use_mse_loss

        # Uncertainty reward parameters
        self.uncertainty_abs_coef = uncertainty_abs_coef
        self.uncertainty_delta_coef = uncertainty_delta_coef
        self.uncertainty_delta_threshold = uncertainty_delta_threshold
        self.u_max = uncertainty_max
        self.u_min = uncertainty_min
        self.uncertainty_warmup_iters = uncertainty_warmup_iters
        self.uncertainty_ema_beta = uncertainty_ema_beta
        self.uncertainty_eps = uncertainty_eps
        self.u_q_low = None
        self.u_q_high = None
        self.last_uncertainty = None
        self.reset_uncertainty_history = None

        self.accumulated_unc_reward = 0.0
        self.accumulated_unc_abs_reward = 0.0
        self.accumulated_unc_delta_reward = 0.0
        self.accumulated_unc_reward_count = 0

        # Action imitation reward statistics
        self.accumulated_imitation_reward = 0.0
        self.accumulated_imitation_reward_count = 0

        self.num_updates = 0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        # Force "rl" type to get values/advantages buffers, but we modified RolloutStorage to also have privileged_actions
        self.storage = RolloutStorage(
            "rl",
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )
        self.reset_uncertainty_history = torch.ones(num_envs, dtype=torch.bool, device=self.device)
    
    def compute_returns(self, last_teacher_obs):
        last_values = self.policy.teacher.evaluate(last_teacher_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    
    def act(self, obs, teacher_obs):
        # 1. Student acts (RL)
        # Compute student actions and values
        # Note: We use teacher's critic for value estimation (Asymmetric Actor-Critic)
        # or if student had a critic we would use it. Here we assume teacher critic.
        
        # Save hidden states BEFORE act() so trajectory reconstruction uses pre-act states
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        # Student action (no uncertainty)
        self.transition.actions = self.policy.act(obs).detach()

        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        
        
        # Value from Teacher (Critic)
        self.transition.values = self.policy.teacher.evaluate(teacher_obs).detach()
        

        # 2. Teacher acts (for BC target)
        # We need teacher's action for BC loss
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach() 

        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Optional action imitation reward (RL-style imitation)
        if self.use_action_imitation_reward:
            action_gap = self.transition.actions - self.transition.privileged_actions
            # imitation reward: exp(-error), where error is mean squared action gap
            error = action_gap.pow(2).mean(dim=-1)
            imitation_reward = self.action_imitation_reward_coef * torch.exp(-error)
            rewards += imitation_reward
            self.accumulated_imitation_reward += imitation_reward.mean().item()
            self.accumulated_imitation_reward_count += 1

        # record the rewards and dones
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # --- FIX START: Bootstrapping on time outs ---
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        # --- FIX END ---

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        self.num_updates += 1
        mean_mse_loss = 0.0
        mean_bc_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_value_loss = 0.0
        
        # Latent statistics
        mean_student_mean_norm = 0.0
        mean_teacher_mean_norm = 0.0
        
        gen_cnt = 0

        # Compute returns and advantages
        # last_values = torch.zeros(self.storage.num_envs, 1, device=self.device)
        # self.storage.compute_returns(last_values, self.gamma, self.lam)

        # Generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, privileged_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch, \
            privileged_actions_batch in generator:

                # 1. Compute Latent (Encoder) - deterministic, no uncertainty
                student_latent = self.policy.get_student_latent(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                
                # 2. MSE Loss (Encoder update only)
                with torch.no_grad():
                    teacher_latent = self.policy.evaluate_feature(privileged_obs_batch)
                
                # Change from gaussian_nll_loss to MSE loss
                mse_loss = F.mse_loss(student_latent, teacher_latent)

                # 3. Detach Latent for Policy Head (Stop gradient from RL/BC to Encoder)
                if self.RL_loss_coef != 0.0:
                    student_latent_detached = student_latent
                else:
                    student_latent_detached = student_latent.detach()

                # 4. Run Policy Head (Head update only)
                # Policy input is just the latent features (no uncertainty)
                self.policy.update_distribution(student_latent_detached)
                
                # 5. Compute PPO/BC variables
                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                mu_batch = self.policy.action_mean
                sigma_batch = self.policy.action_std
                entropy_batch = self.policy.entropy
                
                # 6. BC Loss (MSE) – gated by RL imitation flag
                bc_loss = torch.tensor(0.0, device=self.device)

                if not self.use_action_imitation_reward:
                    # Student action (mu_batch) vs Teacher action (privileged_actions_batch)
                    bc_loss = F.mse_loss(mu_batch, privileged_actions_batch)

                # 7. PPO Loss (Surrogate)
                # Adaptive LR / KL
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                        if self.critic_optimizer is not None:
                            for param_group in self.critic_optimizer.param_groups:
                                param_group["lr"] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # 8. Value Loss (Teacher Critic)
                if self.critic_optimizer is not None:
                    #print("INFO: Computing value loss with critic update")
                    value_batch = self.policy.teacher.evaluate(privileged_obs_batch)
                    # Value function loss
                    if self.use_clipped_value_loss:
                        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                            -self.clip_param, self.clip_param
                        )
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2).mean()
                else:
                    value_loss = torch.tensor(0.0, device=self.device)
                
                # Total Loss
                # mse_loss affects encoder
                # surrogate_loss (+ optional bc_loss) affects policy head (due to detach)
                loss = self.RL_loss_coef * surrogate_loss - self.entropy_coef * entropy_batch.mean()
                
                if not self.use_action_imitation_reward:
                    loss += self.bc_loss_coef * bc_loss
                
                if self.use_mse_loss:
                    loss += mse_loss

                if self.critic_optimizer is not None:
                    loss += self.value_loss_coef * value_loss

                # Gradient step
                self.optimizer.zero_grad()
                if self.critic_optimizer is not None:
                    self.critic_optimizer.zero_grad()
                
                loss.backward()
                
                nn.utils.clip_grad_norm_(self.student_parameters, self.max_grad_norm)
                if self.teacher_critic_parameters:
                    nn.utils.clip_grad_norm_(self.teacher_critic_parameters, self.max_grad_norm)
                
                self.optimizer.step()
                if self.critic_optimizer is not None:
                    self.critic_optimizer.step()

                # Logging
                mean_mse_loss += mse_loss.item()
                mean_bc_loss += bc_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_value_loss += value_loss.item()
                gen_cnt += 1
                
                # Debug stats
                with torch.no_grad():
                    mean_student_mean_norm += student_latent.norm(dim=-1).mean().item()
                    mean_teacher_mean_norm += teacher_latent.norm(dim=-1).mean().item()
                    
                    # Cache last batch data for saving to disk later
                    self.last_debug_data = {
                        "obs": obs_batch.detach().cpu(),
                        "privileged_obs": privileged_obs_batch.detach().cpu(),
                        "student_latent": student_latent.detach().cpu(),
                        "teacher_latent": teacher_latent.detach().cpu(),
                    }

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_mse_loss /= num_updates
        mean_bc_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_value_loss /= num_updates
        mean_student_mean_norm /= num_updates
        mean_teacher_mean_norm /= num_updates

        if self.accumulated_imitation_reward_count > 0:
            mean_imitation_reward = self.accumulated_imitation_reward / self.accumulated_imitation_reward_count
        else:
            mean_imitation_reward = 0.0
        self.accumulated_imitation_reward = 0.0
        self.accumulated_imitation_reward_count = 0

        self.storage.clear()

        # --- Log Histograms to WandB (Rank 0 only) ---
        if self.gpu_global_rank == 0 and self.num_updates % 100 == 0:
            try:
                import wandb
                if wandb.run is not None and hasattr(self, "last_debug_data") and self.last_debug_data is not None:
                    hists = {
                        "latent_dist/student": wandb.Histogram(self.last_debug_data["student_latent"].numpy()),
                        "latent_dist/teacher": wandb.Histogram(self.last_debug_data["teacher_latent"].numpy()),
                    }
                    wandb.log(hists, commit=False)
            except ImportError:
                pass
        # ---------------------------------------------

        # construct the loss dictionary
        loss_dict = {
            "mse_loss": mean_mse_loss,
            "bc_loss": mean_bc_loss * self.bc_loss_coef if not self.use_action_imitation_reward else 0.0,
            "surrogate_loss": mean_surrogate_loss * self.RL_loss_coef,
            "value_function": mean_value_loss,
            "latent/student_mean_norm": mean_student_mean_norm,
            "latent/teacher_mean_norm": mean_teacher_mean_norm,
            "distillation/mean_action_imitation_reward": mean_imitation_reward,
        }

        return loss_dict

    def _normalize_uncertainty(self, u: torch.Tensor) -> torch.Tensor:
        """EMA-based robust quantile normalization to [0, 1]."""
        u_flat = u.detach()
        q_low_now = torch.quantile(u_flat, 0.05).item()
        q_high_now = torch.quantile(u_flat, 0.95).item()

        if self.u_q_low is None:
            self.u_q_low = q_low_now
            self.u_q_high = max(q_high_now, q_low_now + self.uncertainty_eps)
        else:
            beta = self.uncertainty_ema_beta
            self.u_q_low = (1 - beta) * self.u_q_low + beta * q_low_now
            self.u_q_high = (1 - beta) * self.u_q_high + beta * max(q_high_now, q_low_now + self.uncertainty_eps)

        denom = max(self.u_q_high - self.u_q_low, self.uncertainty_eps)
        u_norm = (u - self.u_q_low) / denom
        return torch.clamp(u_norm, 0.0, 1.0)

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self, params=None):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        if not self.is_multi_gpu:
            return
        if params is None:
            params = self.student_parameters
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel

    def _reduce_module_gradients(self, module: nn.Module) -> None:
        if not self.is_multi_gpu:
            return

        grads = [param.grad.view(-1) for param in module.parameters() if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in module.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
