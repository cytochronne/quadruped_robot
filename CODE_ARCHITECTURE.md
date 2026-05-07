# rl_base 代码架构与接口文档

本文档记录 `rl_base` 目录内本地强化学习库的代码架构、训练数据流、模块边界，以及当前代码中所有函数和方法的功能与接口。文档面向后续维护、调试、扩展算法和接入 Isaac Lab 环境使用。

## 1. 项目定位

`rl_base` 是仓库内使用的本地强化学习库，包名为 `rl_base`，项目元信息定义在 `pyproject.toml`：

- 包名：`rl-base-lib`
- 当前版本：`2.3.3`
- Python：`>=3.8`
- 主要依赖：`torch`、`torchvision`、`numpy`、`GitPython`、`onnx`
- 主要用途：基于 PyTorch 实现 PPO、Distillation、TD3、SAC 等算法，并提供 Isaac Lab 环境包装、策略导出、日志记录和 rollout/replay 存储。

整体上，该库包含两条训练路径：

- On-policy 路径：`VecEnv -> OnPolicyRunner -> PPO/Distillation -> RolloutStorage -> policy module`
- Off-policy 路径：`gymnasium Env -> TD3/SAC -> ReplayBuffer -> actor/critic networks`

## 2. 目录结构

```text
rl_base/
  README.md
  pyproject.toml
  setup.py
  rl_base/
    __init__.py
    isaaclab_support.py
    algorithms/
      __init__.py
      ppo.py
      distillation.py
      td3.py
      sac.py
      offpolicy_common.py
    env/
      __init__.py
      vec_env.py
    modules/
      __init__.py
      actor_critic.py
      actor_critic_recurrent.py
      student_teacher.py
      student_teacher_recurrent.py
      terrain_aware_actor_critic.py
      terrain_aware_student_teacher.py
      discriminatorAEP.py
      normalizer.py
      rnd.py
    networks/
      __init__.py
      memory.py
    runners/
      __init__.py
      on_policy_runner.py
    storage/
      __init__.py
      rollout_storage.py
    utils/
      __init__.py
      utils.py
      wandb_utils.py
      neptune_utils.py
```

## 3. 包导出关系

### `rl_base/rl_base/__init__.py`

主包入口，仅声明模块文档字符串，没有额外导出对象。

### `rl_base/rl_base/algorithms/__init__.py`

导出算法类：

- `PPO`
- `Distillation`
- `TD3`
- `SAC`

### `rl_base/rl_base/modules/__init__.py`

导出神经网络组件：

- `ActorCritic`
- `ActorCriticRecurrent`
- `EmpiricalNormalization`
- `RandomNetworkDistillation`
- `StudentTeacher`
- `StudentTeacherRecurrent`
- `TerrainAwareActorCritic`
- `TerrainAwareStudentTeacher`
- `Discriminator`

### `rl_base/rl_base/env/__init__.py`

导出：

- `VecEnv`

### `rl_base/rl_base/networks/__init__.py`

导出：

- `Memory`

### `rl_base/rl_base/runners/__init__.py`

导出：

- `OnPolicyRunner`

### `rl_base/rl_base/storage/__init__.py`

导出：

- `RolloutStorage`

### `rl_base/rl_base/utils/__init__.py`

导出工具函数：

- `resolve_nn_activation`
- `split_and_pad_trajectories`
- `unpad_trajectories`
- `store_code_state`
- `string_to_callable`

## 4. 核心训练数据流

### 4.1 PPO 强化学习路径

1. `OnPolicyRunner.__init__` 从环境读取 actor observation 和可选 critic observation。
2. 根据 `train_cfg["policy"]["class_name"]` 动态构造策略网络，例如 `ActorCritic`、`ActorCriticRecurrent`、`TerrainAwareActorCritic`。
3. 根据 `train_cfg["algorithm"]["class_name"]` 构造 `PPO`。
4. `PPO.init_storage(...)` 创建 `RolloutStorage(training_type="rl")`。
5. `OnPolicyRunner.learn(...)` 每个 iteration 执行：
   - 调用 `PPO.act(obs, critic_obs)` 采样动作、估计 value、记录 log prob。
   - 调用 `env.step(actions)`。
   - 调用 `PPO.process_env_step(rewards, dones, infos)` 写入 rollout。
   - rollout 满后调用 `PPO.compute_returns(last_critic_obs)`。
   - 调用 `PPO.update()` 进行多个 epoch / mini-batch 更新。
   - 记录日志、定期保存 checkpoint。

### 4.2 Student-Teacher 蒸馏路径

1. `OnPolicyRunner.__init__` 在算法为 `Distillation` 时，将 `extras["observations"]["teacher"]` 作为 teacher/privileged observation；如果不存在则退回普通 observation。
2. 策略通常是 `StudentTeacher`、`StudentTeacherRecurrent` 或 `TerrainAwareStudentTeacher`。
3. `runner.load(path)` 加载 teacher 或 distillation checkpoint；蒸馏训练开始前要求 `policy.loaded_teacher == True`。
4. `Distillation.act(obs, teacher_obs)`：
   - student 采样训练动作。
   - teacher actor 生成 imitation target action。
   - teacher critic 估计 value。
5. `Distillation.update()` 同时包含：
   - student latent 对 teacher latent 的 MSE。
   - student action mean 对 teacher action 的 BC loss，除非启用 `use_action_imitation_reward`。
   - PPO surrogate loss。
   - teacher critic value loss。
   - BC -> RL curriculum 系数和 noise std 调度。

### 4.3 TD3/SAC off-policy 路径

TD3 和 SAC 不经过 `OnPolicyRunner`，直接接收 Gymnasium 环境：

1. 构造本地 actor/critic 和 target 网络。
2. 用 `ReplayBuffer` 缓存转移。
3. `learn(total_timesteps)` 循环环境交互。
4. 过 `learning_starts` 后，每 `train_freq` 个 vector step 从 replay buffer 采样并更新。
5. `save(...)` 保存模型、优化器和训练计数；`save_replay_buffer(...)` 保存 replay buffer。

## 5. 环境接口

文件：`rl_base/rl_base/env/vec_env.py`

### `class VecEnv(ABC)`

向量化环境抽象类。`OnPolicyRunner` 只依赖这个接口，因此任意环境只要实现这些属性和方法即可被 on-policy runner 驱动。

约定属性：

- `num_envs: int`：并行环境数量。
- `num_actions: int`：动作维度。
- `max_episode_length: int | torch.Tensor`：最大 episode 长度。
- `episode_length_buf: torch.Tensor`：每个环境当前 episode 步数。
- `device: torch.device`：环境所在设备。
- `cfg: dict | object`：环境配置对象。

约定 `extras` 格式：

- `extras["observations"]`：额外 observation 字典。
  - `critic`：critic privileged observation。
  - `teacher`：distillation teacher observation。
  - `rnd_state`：RND 输入状态。
- `extras["time_outs"]`：由于时间限制而结束的环境。
- `extras["log"]`：episode 或调试日志。

接口：

| 函数 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `get_observations(self) -> tuple[torch.Tensor, dict]` | 返回当前 actor observation 和 extras。 | 无 | `(obs, extras)` |
| `reset(self) -> tuple[torch.Tensor, dict]` | 重置所有环境。 | 无 | `(obs, extras)` |
| `step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]` | 执行动作并推进环境。 | `actions` shape 为 `(num_envs, num_actions)` | `(obs, rewards, dones, extras)` |

## 6. Runner

文件：`rl_base/rl_base/runners/on_policy_runner.py`

### `class OnPolicyRunner`

on-policy 训练主控类，负责环境交互、策略/算法构造、rollout 采集、更新调度、日志、checkpoint、多 GPU 初始化和推理策略导出。

#### 构造函数

```python
OnPolicyRunner(env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu")
```

关键输入：

- `env`：实现 `VecEnv` 的环境。
- `train_cfg`：训练配置，至少包含：
  - `algorithm.class_name`：`"PPO"` 或 `"Distillation"`。
  - `policy.class_name`：策略类名，例如 `"ActorCritic"`。
  - `num_steps_per_env`
  - `save_interval`
  - `empirical_normalization`
- `log_dir`：日志和 checkpoint 目录。
- `device`：训练设备。

构造流程：

- 调用 `_configure_multi_gpu()` 读取 `WORLD_SIZE`、`LOCAL_RANK`、`RANK`。
- 根据算法类名解析 `training_type`：
  - PPO -> `"rl"`
  - Distillation -> `"distillation"`
- 调用环境 `get_observations()` 推断 observation 维度。
- 调用 `_debug_print_observation_breakdown(...)` 打印观测拆分调试信息。
- 根据配置动态构造 policy 和 algorithm。
- 可选构造 observation normalizer。
- 调用 `alg.init_storage(...)` 初始化 rollout storage。

#### 方法接口

| 函数 | 功能 | 输入 | 输出/副作用 |
| --- | --- | --- | --- |
| `learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False)` | 执行完整 on-policy 训练循环。 | 学习迭代数；是否随机化初始 episode 长度。 | 采集 rollout、更新算法、写日志、保存模型。 |
| `log(self, locs: dict, width: int = 80, pad: int = 35)` | 记录 TensorBoard/W&B/Neptune 指标，并打印终端日志。 | `locals()` 字典；输出宽度。 | 更新 writer、打印训练摘要。 |
| `save(self, path: str, infos=None)` | 保存 checkpoint。 | 文件路径和附加信息。 | 写入模型、优化器、RND、normalizer、可选 discriminator、debug data。 |
| `load(self, path: str, load_optimizer: bool = True)` | 加载 checkpoint。 | 文件路径；是否加载优化器。 | 恢复模型、normalizer、优化器和迭代数；返回 checkpoint 中的 `infos`。 |
| `get_inference_policy(self, device=None)` | 获取推理 callable。 | 可选目标设备。 | 返回 `policy.act_inference`，如启用归一化则包一层 normalizer。 |
| `train_mode(self)` | 切换训练模式。 | 无 | PPO 下调用 `policy.train()`；distillation 下只训练 student 和 teacher critic 相关模块。 |
| `eval_mode(self)` | 切换评估模式。 | 无 | `policy.eval()`、RND eval、normalizer eval。 |
| `add_git_repo_to_log(self, repo_file_path)` | 添加需要保存 git diff 的仓库路径。 | 仓库内任意文件路径。 | 追加到 `git_status_repos`。 |
| `_configure_multi_gpu(self)` | 初始化分布式配置。 | 无 | 设置 rank/world size/device；必要时初始化 NCCL。 |
| `_debug_print_observation_breakdown(self, obs: torch.Tensor, extras: dict) -> None` | 尝试推断 policy/critic observation 的历史长度、每帧维度、关节维度、地形维度。 | 当前 obs 和 extras。 | 打印调试信息；只执行一次。 |

#### 日志指标

常见日志命名空间：

- `Loss/*`：算法返回的 loss。
- `Policy/mean_noise_std`：策略动作噪声。
- `Perf/*`：FPS、采样耗时、学习耗时。
- `Train/*`：平均 reward、episode 长度、success rate、速度跟踪指标。
- `Rnd/*`：RND 外部/内部 reward 和权重。
- `VelTracking/*`：从 Isaac Lab reward manager 推断的速度跟踪准确率。

## 7. Rollout 存储

文件：`rl_base/rl_base/storage/rollout_storage.py`

### `class RolloutStorage`

on-policy 采样缓存。按 `[num_transitions_per_env, num_envs, ...]` 存储一个 iteration 内的 rollout。

#### `class Transition`

单步临时容器，`PPO.act` / `Distillation.act` 填入，再由 `process_env_step` 补 rewards/dones，最后写入 `RolloutStorage`。

字段：

- `observations`
- `privileged_observations`
- `actions`
- `privileged_actions`
- `rewards`
- `dones`
- `values`
- `actions_log_prob`
- `action_mean`
- `action_sigma`
- `hidden_states`
- `rnd_state`

接口：

| 函数 | 功能 |
| --- | --- |
| `__init__(self)` | 初始化所有字段为 `None`。 |
| `clear(self)` | 调用 `__init__` 清空字段。 |

#### `RolloutStorage.__init__`

```python
RolloutStorage(
    training_type,
    num_envs,
    num_transitions_per_env,
    obs_shape,
    privileged_obs_shape,
    actions_shape,
    rnd_state_shape=None,
    device="cpu",
)
```

根据 `training_type` 创建不同 buffer：

- `"rl"`：
  - `values`
  - `actions_log_prob`
  - `mu`
  - `sigma`
  - `returns`
  - `advantages`
  - `privileged_actions`
- `"distillation"`：
  - `privileged_actions`
- 任意带 RND：
  - `rnd_state`
- recurrent policy：
  - `saved_hidden_states_a`
  - `saved_hidden_states_c`

#### 方法接口

| 函数 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `add_transitions(self, transition: Transition)` | 将一个 step 的 transition 写入当前 `step` 位置。 | `Transition` | 无；满时抛 `OverflowError`。 |
| `_save_hidden_states(self, hidden_states)` | 保存 actor/critic RNN hidden states，用于 recurrent mini-batch。 | hidden state tuple | 无 |
| `clear(self)` | 重置写入指针。 | 无 | 无 |
| `compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True)` | 用 GAE 计算 returns 和 advantages。 | 最后 value、折扣、GAE lambda、是否整体归一化 advantage。 | 更新 `returns`、`advantages`。 |
| `generator(self)` | distillation 单步生成器。 | 无 | yield `(obs, privileged_obs, actions, privileged_actions, dones)`。 |
| `mini_batch_generator(self, num_mini_batches, num_epochs=8)` | feedforward PPO mini-batch 生成器。 | mini-batch 数和 epoch 数。 | yield PPO batch tuple。 |
| `recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8)` | recurrent PPO mini-batch 生成器。 | mini-batch 数和 epoch 数。 | yield padded trajectory batch、mask、hidden states。 |

`mini_batch_generator` 和 `recurrent_mini_batch_generator` 当前都 yield 13 个元素：

```python
(
    obs_batch,
    privileged_observations_batch,
    actions_batch,
    target_values_batch,
    advantages_batch,
    returns_batch,
    old_actions_log_prob_batch,
    old_mu_batch,
    old_sigma_batch,
    hid_states_batch,
    masks_batch,
    rnd_state_batch,
    privileged_actions_batch,
)
```

## 8. Algorithms

### 8.1 PPO

文件：`rl_base/rl_base/algorithms/ppo.py`

#### `class PPO`

实现 clipped PPO，支持：

- feedforward 和 recurrent policy。
- RND intrinsic reward。
- symmetry data augmentation / mirror loss。
- adaptive KL learning-rate schedule。
- multi-GPU 梯度同步。

#### 构造函数

```python
PPO(
    policy,
    num_learning_epochs=1,
    num_mini_batches=1,
    clip_param=0.2,
    gamma=0.998,
    lam=0.95,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    learning_rate=1e-3,
    max_grad_norm=1.0,
    use_clipped_value_loss=True,
    schedule="fixed",
    desired_kl=0.01,
    device="cpu",
    normalize_advantage_per_mini_batch=False,
    rnd_cfg: dict | None = None,
    symmetry_cfg: dict | None = None,
    multi_gpu_cfg: dict | None = None,
)
```

关键字段：

- `policy`：必须提供 `act`、`evaluate`、`get_actions_log_prob`、`action_mean`、`action_std`、`entropy`。
- `storage`：由 `init_storage` 创建。
- `transition`：当前 step 临时 transition。
- `rnd` / `rnd_optimizer`：可选 RND 模块。
- `symmetry`：可选对称数据增强配置。

#### 方法接口

| 函数 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape)` | 初始化 `RolloutStorage`。 | rollout 形状和训练类型。 | 无 |
| `act(self, obs, critic_obs)` | 用当前 policy 采样动作并记录 value/log prob/action distribution。 | actor obs、critic obs。 | detached actions。 |
| `process_env_step(self, rewards, dones, infos)` | 补全 reward/done，处理 RND intrinsic reward 和 timeout bootstrap，然后写入 storage。 | rewards、dones、extras。 | 无 |
| `compute_returns(self, last_critic_obs)` | 估计最后一步 value 并计算 GAE。 | last critic obs。 | 无 |
| `update(self)` | 执行 PPO 更新。 | 无 | loss 字典：`value_function`、`surrogate`、`entropy`，可选 `rnd`、`symmetry`。 |
| `broadcast_parameters(self)` | 多 GPU 下从 rank 0 广播 policy 和 RND predictor 参数。 | 无 | 无 |
| `reduce_parameters(self)` | 多 GPU 下 all-reduce policy/RND 梯度。 | 无 | 无 |

#### `update()` 内部逻辑

- 从 `RolloutStorage` 选择 feedforward 或 recurrent generator。
- 可选 per-mini-batch advantage normalization。
- 可选 symmetry augmentation：
  - augment obs/actions/critic obs。
  - 复制 old log prob、target values、advantages、returns。
- 重新计算当前策略 log prob、value、entropy。
- 如果 `schedule == "adaptive"`，根据 KL 调整学习率。
- 计算 clipped surrogate loss。
- 计算 clipped 或 unclipped value loss。
- 可选 mirror loss。
- 可选 RND predictor MSE loss。
- 反向传播、梯度裁剪、优化器 step。

### 8.2 Distillation

文件：`rl_base/rl_base/algorithms/distillation.py`

#### `class Distillation`

用于训练 student policy 模仿 teacher policy，同时可混入 PPO 风格 RL 更新。当前实现主要面向 terrain-aware student-teacher：

- student encoder 学 teacher latent。
- student policy head 学 teacher action 或用 PPO surrogate。
- teacher critic 可以继续训练，用于 value estimation。
- 支持 BC -> RL curriculum。
- 支持 action imitation reward。

#### 构造函数

```python
Distillation(
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
    uncertainty_abs_coef=1.0,
    uncertainty_delta_coef=2.0,
    uncertainty_delta_threshold=0.03,
    uncertainty_max=1.0,
    uncertainty_min=0.0,
    uncertainty_warmup_iters=500,
    uncertainty_ema_beta=0.01,
    uncertainty_eps=1e-6,
    curriculum_enable=False,
    curriculum_start_iter=1000,
    curriculum_ramp_iters=1500,
    curriculum_final_rl_coef=1.0,
    curriculum_final_bc_coef=0.0,
    curriculum_noise_start: float | None = None,
    curriculum_noise_target=0.8,
    curriculum_noise_handover_to_rl=True,
    curriculum_type="linear",
    multi_gpu_cfg: dict | None = None,
    **kwargs,
)
```

参数收集规则：

- `student_parameters`：
  - `policy.memory_s` 参数。
  - `policy.student["encoder"]` 或 `policy.student_encoder` 参数。
  - `policy.student_policy_head` 参数。
  - `policy.std` 或 `policy.log_std`。
- `teacher_critic_parameters`：
  - `policy.teacher.critic`
  - `policy.teacher.critic_fusion_encoder`
  - 如果 critic 使用地形高度，则包含共享 `policy.teacher.height_encoder`
- teacher actor 和 actor fusion encoder 会被设为 eval 且禁止梯度。

#### 方法接口

| 函数 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `init_storage(self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape)` | 初始化 storage。当前强制用 `"rl"` 类型，以复用 PPO value/advantage buffer。 | rollout 形状。 | 无 |
| `compute_returns(self, last_teacher_obs)` | 使用 teacher critic 估计最后 value 并计算 GAE。 | last teacher obs。 | 无 |
| `act(self, obs, teacher_obs)` | student 采样动作；teacher actor 生成 privileged action；teacher critic 估值。 | student obs、teacher obs。 | student actions。 |
| `process_env_step(self, rewards, dones, infos)` | 可选加入 action imitation reward，处理 timeout bootstrap，写入 storage。 | rewards、dones、extras。 | 无 |
| `_compute_curriculum_progress(self, update_idx: int) -> float` | 根据 iteration 计算 curriculum 进度。 | 当前 update index。 | `[0, 1]` 进度。 |
| `_mix_with_progress(self, start_value: float, end_value: float, progress: float) -> float` | 线性插值。 | 起点、终点、进度。 | 插值结果。 |
| `_get_current_noise_std(self) -> float` | 读取当前策略动作噪声 std。 | 无 | 标量 std。 |
| `_set_policy_noise_std(self, target_std: float) -> None` | 设置策略动作噪声 std，支持 `log_std` 或 softplus `std`。 | 目标 std。 | 无 |
| `_update_curriculum_state(self) -> None` | 根据 curriculum 配置更新 active RL/BC 系数和 noise std。 | 无 | 更新内部状态。 |
| `update(self)` | 执行蒸馏和 PPO 混合更新。 | 无 | loss 字典。 |
| `_normalize_uncertainty(self, u: torch.Tensor) -> torch.Tensor` | 用 EMA 分位数把 uncertainty 归一化到 `[0, 1]`。 | uncertainty tensor。 | 归一化 tensor。 |
| `broadcast_parameters(self)` | 多 GPU 参数广播。 | 无 | 无 |
| `reduce_parameters(self, params=None)` | 对给定参数列表执行梯度 all-reduce。 | 参数列表，默认 student parameters。 | 无 |
| `_reduce_module_gradients(self, module: nn.Module) -> None` | 对某个 module 的梯度执行 all-reduce。 | module。 | 无 |

#### `update()` 返回 loss 字典

- `mse_loss`
- `bc_loss`
- `surrogate_loss`
- `value_function`
- `latent/student_mean_norm`
- `latent/teacher_mean_norm`
- `distillation/mean_action_imitation_reward`
- `distillation/curriculum_progress`
- `distillation/active_bc_coef`
- `distillation/active_rl_coef`
- `distillation/target_noise_std`

#### 当前实现注意事项

- `uncertainty_*` 参数和 `_normalize_uncertainty(...)` 已保留，但当前主更新路径没有使用 uncertainty reward。
- `Discriminator` 被 import，runner 也保留了 discriminator checkpoint 逻辑，但当前 `Distillation.__init__` 没有实际创建 `self.discriminator`；从 config 传入的 `discriminator_cfg` 和 `adv_loss_weight` 会通过 `**kwargs` 被忽略。

### 8.3 Off-policy 共享工具

文件：`rl_base/rl_base/algorithms/offpolicy_common.py`

#### 顶层函数

| 函数 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `resolve_device(device: str | torch.device | None) -> torch.device` | 解析设备字符串。`None` 或 `"auto"` 自动选 CUDA/CPU。 | device。 | `torch.device`。 |
| `set_global_seed(seed: int | None) -> None` | 设置 Python、NumPy、PyTorch、CUDA 随机种子。 | seed。 | 无 |
| `build_mlp(input_dim: int, output_dim: int, hidden_dims: Iterable[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential` | 构造 MLP。 | 输入维度、输出维度、隐藏层、激活类。 | `nn.Sequential`。 |
| `polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None` | 执行 target 参数 Polyak 更新。 | source、target、tau。 | 无 |
| `ensure_pt_path(path: str | os.PathLike) -> Path` | 确保保存路径带 `.pt` 后缀并创建父目录。 | path。 | `Path`。 |
| `dump_pickle(path: str | os.PathLike, obj) -> None` | pickle 保存对象。 | path、obj。 | 无 |

#### `@dataclass ReplayBatch`

采样 batch 容器：

- `observations: torch.Tensor`
- `actions: torch.Tensor`
- `rewards: torch.Tensor`
- `next_observations: torch.Tensor`
- `dones: torch.Tensor`

#### `class ReplayBuffer`

NumPy 环形 replay buffer，采样时转为 torch tensor。

构造函数：

```python
ReplayBuffer(obs_shape: tuple[int, ...], action_shape: tuple[int, ...], capacity: int, device: torch.device)
```

方法：

| 函数 | 功能 |
| --- | --- |
| `add(self, obs, action, reward: float, next_obs, done: bool, timeout: bool = False) -> None` | 写入单条 transition。 |
| `add_batch(self, observations, actions, rewards, next_observations, dones, timeouts=None) -> None` | 批量写入 transition，支持 vector env。 |
| `sample(self, batch_size: int) -> ReplayBatch` | 随机采样 batch；`dones` 会屏蔽 timeout。 |
| `__len__(self) -> int` | 返回当前有效样本数。 |
| `state_dict(self) -> dict` | 导出 replay buffer 状态。 |
| `load_state_dict(self, state_dict: dict) -> None` | 恢复 replay buffer 状态，兼容无 `timeouts` 的旧 checkpoint。 |

#### `class TensorboardLogger`

轻量 TensorBoard logger。

| 函数 | 功能 |
| --- | --- |
| `__init__(self, log_dir: str | None)` | 如果 `log_dir` 可用则创建 `SummaryWriter`，否则禁用 writer。 |
| `add_scalar(self, tag: str, value: float, step: int) -> None` | 记录最新 scalar，并写 TensorBoard。 |
| `close(self) -> None` | 关闭 writer。 |

### 8.4 TD3

文件：`rl_base/rl_base/algorithms/td3.py`

#### `class TD3Actor`

确定性 actor，输出经过 `tanh` 后映射到 action space。

```python
TD3Actor(obs_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray, hidden_dims: tuple[int, ...] = (400, 300))
```

方法：

- `forward(self, obs: torch.Tensor) -> torch.Tensor`：返回缩放到 `[action_low, action_high]` 的动作。

#### `class TD3Critic`

twin Q critic。

```python
TD3Critic(obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (400, 300))
```

方法：

- `forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`：返回 `(q1, q2)`。
- `q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor`：只计算 Q1，用于 actor loss。

#### `class TD3`

本地 TD3 实现，只支持 `policy="MlpPolicy"` 和 Box observation/action space。

构造函数：

```python
TD3(
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
)
```

方法：

| 函数 | 功能 |
| --- | --- |
| `_ensure_batched_obs(self, obs) -> np.ndarray` | 将 env observation 变成 `(n_envs, *obs_shape)`。 |
| `_reshape_action_batch(self, action_batch_flat: np.ndarray) -> np.ndarray` | 将 flat action reshape 成 env action shape。 |
| `_sample_random_actions(self) -> np.ndarray` | 从 action space 均匀采样随机动作。 |
| `_apply_terminal_obs_and_timeouts(self, infos, next_obs_batch, done_batch, timeout_batch) -> tuple[np.ndarray, np.ndarray]` | 处理 `terminal_observation` 和 `TimeLimit.truncated`。 |
| `_predict_actions(self, obs_batch: np.ndarray, deterministic: bool = False) -> np.ndarray` | actor 推理动作，并在非 deterministic 时添加探索噪声。 |
| `_train_step(self) -> dict[str, float]` | 执行一次 critic 更新，并按 `policy_delay` 延迟更新 actor/target。 |
| `learn(self, total_timesteps: int, log_interval: int = 10, tb_log_name: str = "TD3") -> "TD3"` | 训练循环。 |
| `save(self, save_path: str | Path) -> None` | 保存 actor/critic/target/optimizer/config。 |
| `save_replay_buffer(self, path: str | Path) -> None` | 保存 replay buffer。 |

### 8.5 SAC

文件：`rl_base/rl_base/algorithms/sac.py`

#### `class SACActor`

高斯策略 actor，使用 reparameterization 和 `tanh` squashing。

构造函数：

```python
SACActor(obs_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray, hidden_dims: tuple[int, ...] = (256, 256))
```

方法：

| 函数 | 功能 |
| --- | --- |
| `_distribution_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` | 计算 Gaussian mean 和 clamp 后的 log std。 |
| `forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor` | 返回动作。 |
| `action_log_prob(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]` | 返回 squashed action 和修正后的 log prob。 |

#### `class SACCritic`

twin Q critic。

- `__init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256))`
- `forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`

#### `class SAC`

本地 SAC 实现，只支持 `policy="MlpPolicy"` 和 Box observation/action space。

构造函数：

```python
SAC(
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
)
```

方法：

| 函数 | 功能 |
| --- | --- |
| `_setup_entropy_coef(self) -> None` | 初始化固定或可学习 entropy coefficient。 |
| `_ensure_batched_obs(self, obs) -> np.ndarray` | 规范 observation batch 维度。 |
| `_reshape_action_batch(self, action_batch_flat: np.ndarray) -> np.ndarray` | reshape action 给环境。 |
| `_sample_random_actions(self) -> np.ndarray` | 均匀随机动作。 |
| `_apply_terminal_obs(self, infos, next_obs_batch: np.ndarray, done_batch: np.ndarray) -> np.ndarray` | 用 terminal observation 修正 next obs。 |
| `_predict_actions(self, obs_batch: np.ndarray, deterministic: bool = False) -> np.ndarray` | actor 推理动作。 |
| `_current_ent_coef(self) -> torch.Tensor` | 返回当前 entropy coefficient。 |
| `_train_step(self) -> dict[str, float]` | 执行一次 SAC 更新，包括 entropy coefficient、critic、actor 和 target critic。 |
| `learn(self, total_timesteps: int, log_interval: int = 10, tb_log_name: str = "SAC") -> "SAC"` | 训练循环。 |
| `save(self, save_path: str | Path) -> None` | 保存 actor/critic/target/optimizer/entropy/config。 |
| `save_replay_buffer(self, path: str | Path) -> None` | 保存 replay buffer。 |

## 9. Policy 和神经网络模块

### 9.1 ActorCritic

文件：`rl_base/rl_base/modules/actor_critic.py`

#### `class ActorCritic(nn.Module)`

基础 feedforward actor-critic。`is_recurrent = False`。

构造函数：

```python
ActorCritic(
    num_actor_obs,
    num_critic_obs,
    num_actions,
    actor_hidden_dims=[256, 256, 256],
    critic_hidden_dims=[256, 256, 256],
    activation="elu",
    init_noise_std=1.0,
    noise_std_type: str = "scalar",
    **kwargs,
)
```

行为：

- actor：`num_actor_obs -> actor_hidden_dims -> num_actions`
- critic：`num_critic_obs -> critic_hidden_dims -> 1`
- action distribution：`torch.distributions.Normal(mean, std)`
- `noise_std_type`：
  - `"scalar"`：直接优化 `self.std`
  - `"log"`：优化 `self.log_std` 并 exponentiate

接口：

| 函数/属性 | 功能 |
| --- | --- |
| `init_weights(sequential, scales)` | 静态方法，对 Sequential 中 Linear 层做 orthogonal 初始化。当前未被主路径调用。 |
| `reset(self, dones=None)` | feedforward 策略无状态，当前为空。 |
| `forward(self)` | 未实现，直接调用会抛 `NotImplementedError`。 |
| `action_mean` | 当前 distribution mean。 |
| `action_std` | 当前 distribution stddev。 |
| `entropy` | 当前动作分布 entropy，按 action 维度求和。 |
| `update_distribution(self, observations)` | 用 actor 计算 mean，用噪声参数构造 Normal distribution。 |
| `act(self, observations, **kwargs)` | 更新 distribution 并采样动作。 |
| `get_actions_log_prob(self, actions)` | 返回动作 log prob，按 action 维度求和。 |
| `act_inference(self, observations)` | 返回 actor mean，用于推理。 |
| `evaluate(self, critic_observations, **kwargs)` | 返回 critic value。 |
| `load_state_dict(self, state_dict, strict=True)` | 加载参数并返回 `True`，表示恢复训练。 |

### 9.2 ActorCriticRecurrent

文件：`rl_base/rl_base/modules/actor_critic_recurrent.py`

#### `class ActorCriticRecurrent(ActorCritic)`

RNN actor-critic。`is_recurrent = True`。

构造函数：

```python
ActorCriticRecurrent(
    num_actor_obs,
    num_critic_obs,
    num_actions,
    actor_hidden_dims=[256, 256, 256],
    critic_hidden_dims=[256, 256, 256],
    activation="elu",
    rnn_type="lstm",
    rnn_hidden_dim=256,
    rnn_num_layers=1,
    init_noise_std=1.0,
    **kwargs,
)
```

结构：

- `memory_a = Memory(num_actor_obs, rnn_type, rnn_num_layers, rnn_hidden_dim)`
- `memory_c = Memory(num_critic_obs, rnn_type, rnn_num_layers, rnn_hidden_dim)`
- 父类 actor/critic 的输入维度都变成 `rnn_hidden_dim`。

接口：

| 函数 | 功能 |
| --- | --- |
| `reset(self, dones=None)` | 重置 actor/critic RNN hidden states。 |
| `act(self, observations, masks=None, hidden_states=None)` | 先过 actor memory，再调用父类 `act`。 |
| `act_inference(self, observations)` | 推理时使用 actor memory。 |
| `evaluate(self, critic_observations, masks=None, hidden_states=None)` | 先过 critic memory，再调用父类 `evaluate`。 |
| `get_hidden_states(self)` | 返回 `(memory_a.hidden_states, memory_c.hidden_states)`。 |

### 9.3 StudentTeacher

文件：`rl_base/rl_base/modules/student_teacher.py`

#### `class StudentTeacher(nn.Module)`

基础 feedforward student-teacher 策略。student 训练，teacher 通常从 PPO actor 加载并固定推理。`is_recurrent = False`。

构造函数：

```python
StudentTeacher(
    num_student_obs,
    num_teacher_obs,
    num_actions,
    student_hidden_dims=[256, 256, 256],
    teacher_hidden_dims=[256, 256, 256],
    activation="elu",
    init_noise_std=0.1,
    **kwargs,
)
```

接口：

| 函数/属性 | 功能 |
| --- | --- |
| `reset(self, dones=None, hidden_states=None)` | feedforward 无状态，当前为空。 |
| `forward(self)` | 未实现。 |
| `action_mean` | student distribution mean。 |
| `action_std` | student distribution stddev。 |
| `entropy` | student distribution entropy。 |
| `update_distribution(self, observations)` | 用 student 计算 mean 并创建 Normal distribution。 |
| `act(self, observations)` | student 采样动作。 |
| `act_inference(self, observations)` | student mean action。 |
| `evaluate(self, teacher_observations)` | teacher mean action，`torch.no_grad()`。 |
| `load_state_dict(self, state_dict, strict=True)` | 支持从 PPO actor checkpoint 加载 teacher，或从 distillation checkpoint 加载完整 student/teacher。 |
| `get_hidden_states(self)` | 返回 `None`。 |
| `detach_hidden_states(self, dones=None)` | feedforward 无状态，当前为空。 |

`load_state_dict` 返回值约定：

- 返回 `False`：加载的是 RL teacher，表示不是恢复 distillation 训练。
- 返回 `True`：加载的是 student-teacher checkpoint，表示恢复训练。

### 9.4 StudentTeacherRecurrent

文件：`rl_base/rl_base/modules/student_teacher_recurrent.py`

#### `class StudentTeacherRecurrent(StudentTeacher)`

RNN student，可选 RNN teacher。`is_recurrent = True`。

构造函数：

```python
StudentTeacherRecurrent(
    num_student_obs,
    num_teacher_obs,
    num_actions,
    student_hidden_dims=[256, 256, 256],
    teacher_hidden_dims=[256, 256, 256],
    activation="elu",
    rnn_type="lstm",
    rnn_hidden_dim=256,
    rnn_num_layers=1,
    init_noise_std=0.1,
    teacher_recurrent=False,
    **kwargs,
)
```

接口：

| 函数 | 功能 |
| --- | --- |
| `reset(self, dones=None, hidden_states=None)` | 重置 student memory；如 `teacher_recurrent` 也重置 teacher memory。 |
| `act(self, observations)` | observation 先过 `memory_s`，再由 student 采样。 |
| `act_inference(self, observations)` | student recurrent 推理。 |
| `evaluate(self, teacher_observations)` | teacher 推理；如果 teacher recurrent，先过 `memory_t`。 |
| `get_hidden_states(self)` | 返回 `(student_hidden, teacher_hidden_or_None)`。 |
| `detach_hidden_states(self, dones=None)` | detach student/teacher hidden states。 |

### 9.5 TerrainAwareActorCritic

文件：`rl_base/rl_base/modules/terrain_aware_actor_critic.py`

#### `class TerrainAwareActorCritic(nn.Module)`

地形感知 actor-critic。将 observation 尾部 `height_obs_dim` 视为高度扫描，单独编码后与本体状态融合。当前类声明 `is_recurrent = False`，虽然构造参数保留了 `rnn_type/rnn_hidden_dim/rnn_num_layers`，但当前实现没有使用 RNN。

构造函数：

```python
TerrainAwareActorCritic(
    num_actor_obs: int,
    num_critic_obs: int,
    num_actions: int,
    *,
    height_obs_dim: int,
    actor_hidden_dims: Sequence[int] = (256, 256, 256),
    critic_hidden_dims: Sequence[int] = (256, 256, 256),
    fusion_encoder_dims: Sequence[int] | None = (256, 128, 96),
    height_cnn_channels: Sequence[int] = (16, 32),
    height_map_shape: Tuple[int, int] | None = None,
    activation: str = "elu",
    init_noise_std: float = 1.0,
    noise_std_type: str = "scalar",
    height_encoder_dims: Sequence[int] | None = None,
    rnn_type: str = "lstm",
    rnn_hidden_dim: int = 256,
    rnn_num_layers: int = 1,
    build_critic: bool = True,
    **kwargs,
)
```

结构：

- observation split：
  - core：`obs[..., :-height_dim]`
  - height：`obs[..., -height_dim:]`
- height encoder：2D CNN + flatten。
- fusion encoder：core + height embedding -> latent。
- actor head：latent -> action mean。
- critic head：critic latent -> value，可由 `build_critic=False` 禁用。

接口：

| 函数/属性 | 功能 |
| --- | --- |
| `_resolve_height_map_shape(height_dim, explicit_shape)` | 解析高度图 2D shape；未显式传入时选接近正方形的因子。 |
| `_build_head(input_dim, hidden_dims, output_dim, activation_name)` | 构造 MLP head。 |
| `_build_height_cnn(map_shape, channels, activation_name)` | 构造 height CNN encoder，并返回 embedding dim。 |
| `_build_fusion_encoder(input_dim, hidden_dims, activation_name)` | 构造 fusion MLP。 |
| `reset(self, dones=None)` | 当前无状态，返回 `None`。 |
| `get_hidden_states(self)` | 返回 `(None, None)`。 |
| `detach_hidden_states(self, dones=None)` | 当前无状态，返回 `None`。 |
| `_split_obs(self, obs, height_dim)` | 拆分 core 和 height。 |
| `_encode_height(self, height, height_dim)` | 将 height flat vector reshape 成 map 并编码。 |
| `_prepare_features(self, observations, height_dim, fusion_encoder)` | 生成 actor 或 critic fusion features。 |
| `update_distribution(self, features)` | 根据 actor head 输出 mean 并构造 Normal distribution。 |
| `act(self, observations, masks=None, hidden_states=None)` | actor features -> distribution -> sample。 |
| `act_inference(self, observations)` | actor features -> mean action。 |
| `evaluate(self, critic_observations, masks=None, hidden_states=None)` | critic features -> value；critic 禁用时抛 `RuntimeError`。 |
| `get_actions_log_prob(self, actions)` | 当前 distribution 中动作 log prob。 |
| `action_mean` | distribution mean。 |
| `action_std` | distribution stddev。 |
| `entropy` | distribution entropy sum。 |
| `load_state_dict(self, state_dict, strict: bool = True)` | 如果 `build_critic=False`，过滤 critic 参数后加载；否则完整加载。 |

### 9.6 TerrainAwareStudentTeacher

文件：`rl_base/rl_base/modules/terrain_aware_student_teacher.py`

#### `class TerrainAwareStudentTeacher(nn.Module)`

地形感知 teacher + recurrent student 的蒸馏模块。`is_recurrent = True`。

构造函数：

```python
TerrainAwareStudentTeacher(
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
)
```

结构：

- teacher：`TerrainAwareActorCritic(build_critic=True)`。
- teacher latent dim：`teacher.actor_fusion_dim`。
- student memory：`Memory(student_core_dim, rnn_type, rnn_num_layers, rnn_hidden_dim)`。
- student encoder：`rnn_hidden_dim -> teacher_latent_dim`。
- student policy head：`teacher_latent_dim -> num_actions`。
- `student = ModuleDict({"encoder": student_encoder, "policy": student_policy_head})`，方便 optimizer 查找。

接口：

| 函数/属性 | 功能 |
| --- | --- |
| `_build_mlp(input_dim, hidden_dims, output_dim, activation_name)` | 构造 student encoder/head MLP。 |
| `_split_obs(obs, height_dim)` | 拆分 core 和 height。当前 student 只使用 core。 |
| `_student_core_features(self, observations, masks=None, hidden_states=None)` | 取 student core obs，经过 `memory_s`，返回 RNN features。 |
| `update_distribution(self, features)` | 用 student policy head 构造动作 Normal distribution。 |
| `act(self, observations, masks=None, hidden_states=None)` | student latent -> distribution -> sample。 |
| `act_inference(self, observations, *, return_latent: bool = False)` | 返回 student mean action；可选同时返回 latent。 |
| `evaluate(self, teacher_observations)` | teacher actor 推理动作。 |
| `evaluate_feature(self, teacher_observations)` | 返回 teacher actor fusion latent。 |
| `get_student_latent(self, observations, masks=None, hidden_states=None)` | 返回 deterministic student latent。 |
| `get_actions_log_prob(self, actions)` | 当前 distribution 下 action log prob。 |
| `reset(self, dones=None, hidden_states=None)` | 重置 student memory。 |
| `get_hidden_states(self)` | 返回 `(memory_s.hidden_states, None)`。 |
| `detach_hidden_states(self, dones=None)` | detach student memory hidden states。 |
| `action_mean` | distribution mean。 |
| `action_std` | distribution stddev。 |
| `entropy` | distribution entropy sum。 |
| `load_state_dict(self, state_dict, strict: bool = False)` | 支持恢复完整 student-teacher checkpoint，或加载 raw teacher checkpoint 并丢弃 critic head。 |

`load_state_dict` 返回值：

- full student-teacher checkpoint：返回 `True`，表示恢复训练。
- raw teacher checkpoint：返回 `False`，表示加载 teacher 后开始新蒸馏。

### 9.7 Memory

文件：`rl_base/rl_base/networks/memory.py`

#### `class Memory(torch.nn.Module)`

RNN 封装，支持 LSTM/GRU，兼容在线推理和 padded trajectory batch 更新。

构造函数：

```python
Memory(input_size, type="lstm", num_layers=1, hidden_size=256)
```

接口：

| 函数 | 功能 |
| --- | --- |
| `forward(self, input, masks=None, hidden_states=None)` | `masks is None` 时在线推理并更新内部 hidden state；否则使用传入 hidden state 处理 padded trajectory 并 unpad。 |
| `reset(self, dones=None, hidden_states=None)` | 重置全部或已 done 环境的 hidden states。 |
| `detach_hidden_states(self, dones=None)` | detach 全部或指定 done 环境 hidden states，防止图跨 episode 累积。 |

### 9.8 Normalizer

文件：`rl_base/rl_base/modules/normalizer.py`

#### `class EmpiricalNormalization`

基于运行均值/方差的标准化模块。

构造函数：

```python
EmpiricalNormalization(shape, eps=1e-2, until=None)
```

接口：

| 函数/属性 | 功能 |
| --- | --- |
| `mean` | 返回当前均值副本。 |
| `std` | 返回当前标准差副本。 |
| `forward(self, x)` | 训练模式下先更新统计量，再返回 `(x - mean) / (std + eps)`。 |
| `update(self, x)` | 用 batch 更新运行均值/方差，超过 `until` 后停止学习。 |
| `inverse(self, y)` | 反标准化。 |

#### `class EmpiricalDiscountedVariationNormalization`

奖励尺度标准化，先对 reward 做 discounted average，再用其经验标准差归一化原 reward。

构造函数：

```python
EmpiricalDiscountedVariationNormalization(shape, eps=1e-2, gamma=0.99, until=None)
```

接口：

- `forward(self, rew)`：训练模式下更新 discounted average 和经验统计量，然后返回 `rew / std`。

#### `class DiscountedAverage`

折扣累计平均：

```python
DiscountedAverage(gamma)
```

接口：

- `update(self, rew: torch.Tensor) -> torch.Tensor`：返回 `avg = gamma * avg + rew`。

### 9.9 RandomNetworkDistillation

文件：`rl_base/rl_base/modules/rnd.py`

#### `class RandomNetworkDistillation(nn.Module)`

RND intrinsic reward 模块。包含固定 target network 和可训练 predictor network，intrinsic reward 为 embedding 距离。

构造函数：

```python
RandomNetworkDistillation(
    num_states: int,
    num_outputs: int,
    predictor_hidden_dims: list[int],
    target_hidden_dims: list[int],
    activation: str = "elu",
    weight: float = 0.0,
    state_normalization: bool = False,
    reward_normalization: bool = False,
    device: str = "cpu",
    weight_schedule: dict | None = None,
)
```

`weight_schedule` 支持：

- `{"mode": "constant"}`
- `{"mode": "step", "final_step": int, "final_value": float}`
- `{"mode": "linear", "initial_step": int, "final_step": int, "final_value": float}`

接口：

| 函数 | 功能 |
| --- | --- |
| `get_intrinsic_reward(self, rnd_state) -> tuple[torch.Tensor, torch.Tensor]` | 计算 intrinsic reward，并返回可能已归一化的 `rnd_state`。 |
| `forward(self, *args, **kwargs)` | 未实现；调用会抛异常，要求使用 `get_intrinsic_reward`。 |
| `train(self, mode: bool = True)` | 只切换 predictor 和可选 normalizer；target 保持 eval。 |
| `eval(self)` | 调用 `train(False)`。 |
| `_build_mlp(input_dims, hidden_dims, output_dims, activation_name="elu")` | 构造 predictor/target MLP。 |
| `_constant_weight_schedule(self, step: int, **kwargs)` | 返回初始权重。 |
| `_step_weight_schedule(self, step: int, final_step: int, final_value: float, **kwargs)` | 到达 `final_step` 后切到 final value。 |
| `_linear_weight_schedule(self, step: int, initial_step: int, final_step: int, final_value: float, **kwargs)` | 线性插值权重。 |

### 9.10 Discriminator

文件：`rl_base/rl_base/modules/discriminatorAEP.py`

#### `class Discriminator(nn.Module)`

latent 二分类器，用于 adversarial encoder pre-training 设计：teacher latent 为正类，student latent 为负类。支持 BCEWithLogits 和 Wasserstein 两种 loss。

构造函数：

```python
Discriminator(
    input_dim: int,
    hidden_layer_sizes: list[int],
    *,
    device: str = "cpu",
    loss_type: str = "BCEWithLogits",
    eta_wgan: float = 0.3,
    use_minibatch_std: bool = True,
)
```

接口：

| 函数 | 功能 |
| --- | --- |
| `forward(self, latent: torch.Tensor) -> torch.Tensor` | 返回 discriminator logits。 |
| `classify(self, latent: torch.Tensor) -> torch.Tensor` | 返回 teacher-class probability，即 `sigmoid(logits)`；当前包含 debug print。 |
| `generator_loss(self, student_latent: torch.Tensor) -> torch.Tensor` | student encoder 用于 fool discriminator 的 loss。 |
| `discriminator_loss(self, student_latent: torch.Tensor, teacher_latent: torch.Tensor) -> torch.Tensor` | discriminator 区分 student/teacher latent 的 loss。 |
| `compute_grad_pen(self, teacher_latent, student_latent, lambda_=10.0) -> torch.Tensor` | Wasserstein 下用 WGAN-GP；BCE 下用 R1 风格 penalty。 |
| `_minibatch_std_scalar(self, h: torch.Tensor) -> torch.Tensor` | 计算 minibatch std 特征。 |

实现注意事项：

- 构造函数在 `use_minibatch_std=True` 时会让最后线性层输入维度加 1。
- 当前 `forward` 中拼接 minibatch std 的代码被注释掉，因此默认 `use_minibatch_std=True` 可能导致 `linear` 输入维度不匹配。实际使用时应设为 `False` 或恢复拼接逻辑。

## 10. Isaac Lab 支持

文件：`rl_base/rl_base/isaaclab_support.py`

### 10.1 配置类

这些类使用 Isaac Lab 的 `@configclass`。

#### `class RlBasePpoActorCriticCfg`

PPO actor-critic 网络配置：

- `class_name: str = "ActorCritic"`
- `init_noise_std: float`
- `noise_std_type: Literal["scalar", "log"] = "scalar"`
- `actor_hidden_dims: list[int]`
- `critic_hidden_dims: list[int]`
- `activation: str`

#### `class RlBasePpoAlgorithmCfg`

PPO 算法配置：

- `class_name: str = "PPO"`
- `value_loss_coef`
- `use_clipped_value_loss`
- `clip_param`
- `entropy_coef`
- `num_learning_epochs`
- `num_mini_batches`
- `learning_rate`
- `schedule`
- `gamma`
- `lam`
- `desired_kl`
- `max_grad_norm`
- `normalize_advantage_per_mini_batch: bool = False`
- `symmetry_cfg: dict | None = None`
- `rnd_cfg: dict | None = None`

#### `class RlBaseDistillationAlgorithmCfg`

Distillation 算法配置：

- `class_name: str = "Distillation"`
- `num_learning_epochs`
- `learning_rate`
- `gradient_length`
- `max_grad_norm`
- `optimizer`
- `loss_type`

注意：当前 `Distillation.__init__` 实际使用的参数比该配置类更多，此配置类可能不是完整配置来源。

#### `class RlBaseOnPolicyRunnerCfg`

Runner 配置：

- `seed`
- `device`
- `num_steps_per_env`
- `max_iterations`
- `empirical_normalization`
- `clip_actions`
- `save_interval`
- `experiment_name`
- `run_name`
- `logger`
- `neptune_project`
- `wandb_project`
- `resume`
- `load_run`
- `load_checkpoint`
- `class_name`
- `policy`
- `algorithm`

### 10.2 `class RlBaseVecEnvWrapper(VecEnv)`

将 Isaac Lab `ManagerBasedRLEnv` 或 `DirectRLEnv` 包装成 `VecEnv`。

构造函数：

```python
RlBaseVecEnvWrapper(env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None)
```

行为：

- 从 Isaac Lab 环境推断：
  - `num_envs`
  - `device`
  - `max_episode_length`
  - `num_actions`
  - `num_obs`
  - `num_privileged_obs`
- 如果设置 `clip_actions`，会修改 `single_action_space` 和 batched `action_space`。
- 初始化时调用一次 `env.reset()`。

接口：

| 函数/属性 | 功能 |
| --- | --- |
| `__str__(self)` / `__repr__(self)` | 返回包装器和环境字符串。 |
| `cfg` | 返回 `unwrapped.cfg`。 |
| `render_mode` | 返回底层环境 render mode。 |
| `observation_space` | 返回底层 observation space。 |
| `action_space` | 返回底层 action space。 |
| `class_name(cls) -> str` | 返回类名。 |
| `unwrapped` | 返回底层 Isaac Lab env。 |
| `get_observations(self) -> tuple[torch.Tensor, dict]` | 返回 policy obs 和 `{"observations": obs_dict}`。 |
| `episode_length_buf` getter/setter | 代理到底层环境。 |
| `seed(self, seed: int = -1) -> int` | 设置环境 seed。 |
| `reset(self) -> tuple[torch.Tensor, dict]` | 重置环境并返回 policy obs/extras。 |
| `step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]` | 可选 clip action，调用 Isaac Lab step，并合并 terminated/truncated 为 dones。 |
| `close(self)` | 关闭底层环境。 |
| `_modify_action_space(self)` | 根据 `clip_actions` 修改 action space。 |

### 10.3 策略导出函数

#### `export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt")`

导出 TorchScript JIT 策略。

#### `export_policy_as_onnx(policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False)`

导出 ONNX 策略。

### 10.4 `_TorchPolicyExporter`

TorchScript 导出包装器。

构造函数：

```python
_TorchPolicyExporter(policy, normalizer=None)
```

选择 actor 的规则：

- 如果 policy 有 `actor`：复制 `policy.actor`，recurrent 时复制 `policy.memory_a.rnn`。
- 如果 policy 有 `student`：复制 `policy.student`，recurrent 时复制 `policy.memory_s.rnn`。
- 否则抛 `ValueError`。

接口：

| 函数 | 功能 |
| --- | --- |
| `forward_lstm(self, x)` | LSTM recurrent TorchScript forward，内部维护 hidden/cell state。 |
| `forward_gru(self, x)` | GRU recurrent TorchScript forward，内部维护 hidden state。 |
| `forward(self, x)` | feedforward policy forward。 |
| `reset(self)` | TorchScript export 方法，默认 pass；recurrent 时被替换为 `reset_memory`。 |
| `reset_memory(self)` | 将 recurrent hidden/cell state 清零。 |
| `export(self, path, filename)` | script 并保存到文件。 |

### 10.5 `_OnnxPolicyExporter`

ONNX 导出包装器。

构造函数：

```python
_OnnxPolicyExporter(policy, normalizer=None, verbose=False)
```

接口：

| 函数 | 功能 |
| --- | --- |
| `forward_lstm(self, x_in, h_in, c_in)` | LSTM ONNX forward，显式输入输出 hidden/cell state。 |
| `forward_gru(self, x_in, h_in)` | GRU ONNX forward，显式输入输出 hidden state。 |
| `forward(self, x)` | feedforward ONNX forward。 |
| `export(self, path, filename)` | 根据 recurrent/feedforward 导出 ONNX。 |

## 11. 通用工具

文件：`rl_base/rl_base/utils/utils.py`

| 函数 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `resolve_nn_activation(act_name: str) -> torch.nn.Module` | 将字符串转为激活模块。 | `"elu"`、`"selu"`、`"relu"`、`"crelu"`、`"lrelu"`、`"tanh"`、`"sigmoid"`、`"identity"`。 | 激活层实例；非法字符串抛 `ValueError`。 |
| `split_and_pad_trajectories(tensor, dones)` | 根据 `dones` 切分 trajectory，并 pad 到最长轨迹。 | `tensor` shape `[time, env, ...]`；`dones` shape `[time, env, 1]` 或兼容。 | `(padded_trajectories, masks)`。 |
| `unpad_trajectories(trajectories, masks)` | `split_and_pad_trajectories` 的逆操作。 | padded trajectories 和 masks。 | unpadded tensor。 |
| `store_code_state(logdir, repositories) -> list` | 保存每个 git repo 的 status/diff 到 `logdir/git/*.diff`。 | 日志目录；repo 文件路径列表。 | 已写入 diff 文件路径列表。 |
| `string_to_callable(name: str) -> Callable` | 将 `"module:attribute_name"` 解析为 callable。 | 字符串。 | callable；不可调用或解析失败时抛 `ValueError`。 |

## 12. 外部日志工具

### 12.1 Wandb

文件：`rl_base/rl_base/utils/wandb_utils.py`

#### `class WandbSummaryWriter(SummaryWriter)`

继承 TensorBoard `SummaryWriter`，同时写入 Weights & Biases。

接口：

| 函数 | 功能 |
| --- | --- |
| `__init__(self, log_dir: str, flush_secs: int, cfg)` | 初始化 SummaryWriter 和 `wandb.init`。依赖 `cfg["wandb_project"]`，可选 `WANDB_USERNAME`。 |
| `store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg)` | 写入 runner/policy/algorithm/env 配置到 wandb。 |
| `add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False)` | 同时写 TensorBoard 和 wandb scalar。 |
| `stop(self)` | `wandb.finish()`。 |
| `log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg)` | 调用 `store_config`。 |
| `save_model(self, model_path, iter)` | 上传模型文件。 |
| `save_file(self, path, iter=None)` | 上传任意文件。 |
| `_map_path(self, path)` | 映射不兼容的 tag 名称。 |

### 12.2 Neptune

文件：`rl_base/rl_base/utils/neptune_utils.py`

#### `class NeptuneLogger`

轻量 Neptune run 包装。

| 函数 | 功能 |
| --- | --- |
| `__init__(self, project, token)` | 调用 `neptune.init_run`。 |
| `store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg)` | 保存配置到 Neptune run。 |

#### `class NeptuneSummaryWriter(SummaryWriter)`

继承 TensorBoard `SummaryWriter`，同时写入 Neptune。

接口：

| 函数 | 功能 |
| --- | --- |
| `__init__(self, log_dir: str, flush_secs: int, cfg)` | 初始化 SummaryWriter 和 Neptune run。要求环境变量 `NEPTUNE_API_TOKEN`、`NEPTUNE_USERNAME`。 |
| `_map_path(self, path)` | 映射不兼容 tag。 |
| `add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False)` | 同时写 TensorBoard 和 Neptune scalar。 |
| `stop(self)` | 停止 Neptune run。 |
| `log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg)` | 写入配置。 |
| `save_model(self, model_path, iter)` | 上传 checkpoint。 |
| `save_file(self, path, iter=None)` | 上传文件，主要用于 git diff。 |

## 13. 关键接口契约

### 13.1 Policy 必须提供的接口

PPO 期望 policy 提供：

- `is_recurrent: bool`
- `act(obs, masks=None, hidden_states=None)`
- `evaluate(critic_obs, masks=None, hidden_states=None)`
- `get_actions_log_prob(actions)`
- `action_mean`
- `action_std`
- `entropy`
- `reset(dones)`
- recurrent 时：
  - `get_hidden_states()`

Distillation 额外期望：

- `loaded_teacher: bool`
- `evaluate(teacher_obs)`：teacher action。
- `teacher.evaluate(teacher_obs)`：teacher critic value。
- 对 terrain-aware student：
  - `get_student_latent(obs, masks=None, hidden_states=None)`
  - `evaluate_feature(teacher_obs)`
  - `update_distribution(latent)`

### 13.2 Env extras 契约

PPO：

- actor obs 来自 `env.get_observations()[0]`。
- critic obs 优先使用 `extras["observations"]["critic"]`。
- RND 需要 `extras["observations"]["rnd_state"]`。
- timeout bootstrap 使用 `infos["time_outs"]`。

Distillation：

- student obs 来自普通 actor obs。
- teacher obs 优先使用 `extras["observations"]["teacher"]`。
- 如果没有 teacher obs，则退回普通 obs。

### 13.3 Checkpoint 契约

`OnPolicyRunner.save` 保存字段：

- `model_state_dict`
- `optimizer_state_dict`
- `iter`
- `infos`
- 可选 `discriminator_state_dict`
- 可选 `discriminator_optimizer_state_dict`
- 可选 `rnd_state_dict`
- 可选 `rnd_optimizer_state_dict`
- 可选 `obs_norm_state_dict`
- 可选 `privileged_obs_norm_state_dict`

`policy.load_state_dict(...)` 的返回值会影响 `runner.load(...)`：

- `True`：恢复同类训练，会加载 optimizer 和当前 iteration。
- `False`：加载 teacher 或迁移模型，不加载主 optimizer，不恢复 iteration。

## 14. 实现注意事项和维护建议

1. 动态 `eval(class_name)` 依赖类名已在 `on_policy_runner.py` import 到当前命名空间。新增 policy/algorithm 时需要同步 import。
2. `train_cfg["policy"]` 会被 `pop("class_name")` 修改；如果外部还需要原始 cfg，应传入副本。
3. `TerrainAwareActorCritic` 当前不使用 RNN，虽然保留了 RNN 配置参数。
4. `TerrainAwareStudentTeacher` 的 student 当前只使用 core observation，`student_height_obs_dim` 会被拆分但 height 部分没有进入 student encoder。
5. `RolloutStorage.recurrent_mini_batch_generator` 中 privileged obs 当前保持 `[time, env, dim]` 对齐 actions/returns，而不是 padded trajectory 对齐；这是为了支持 MLP teacher/critic。
6. `Distillation` 中 uncertainty 和 discriminator 相关参数/代码存在保留痕迹，主训练路径未实际使用 adversarial loss。
7. `Discriminator(use_minibatch_std=True)` 默认可能与当前 `forward` 不兼容，使用前需要修复或设为 `False`。
8. `StudentTeacher.__init__` 中 `Normal.set_default_validate_args = False` 是赋值而非调用；`ActorCritic` 使用的是 `Normal.set_default_validate_args(False)`。如依赖该行为，建议统一。
9. `store_code_state` 会写 git diff 到日志目录，有助于复现实验，但需要确保 `log_dir` 存在且可写。
10. TD3/SAC 保存了模型和 replay buffer，但当前没有对应 `load` 方法；需要恢复训练时应补充加载逻辑。

## 15. 新增模块时的接入清单

### 新增 on-policy policy

1. 在 `rl_base/rl_base/modules/` 添加模块。
2. 在 `modules/__init__.py` 导出类。
3. 在 `runners/on_policy_runner.py` import 类，确保 `eval(policy_cfg.pop("class_name"))` 可找到。
4. 实现 PPO/Distillation 所需 policy 契约。
5. 如果 recurrent，确保 `get_hidden_states`、`reset`、`act/evaluate(..., masks, hidden_states)` 与 `RolloutStorage` 兼容。

### 新增 on-policy algorithm

1. 在 `rl_base/rl_base/algorithms/` 添加模块。
2. 在 `algorithms/__init__.py` 导出类。
3. 在 `runners/on_policy_runner.py` import 类。
4. 在 `OnPolicyRunner.__init__` 中为 `class_name` 增加 `training_type` 解析。
5. 实现：
   - `init_storage`
   - `act`
   - `process_env_step`
   - `compute_returns`
   - `update`
   - 可选 `broadcast_parameters`、`reduce_parameters`

### 新增环境包装

1. 继承 `VecEnv`。
2. 提供 `num_envs`、`num_actions`、`max_episode_length`、`episode_length_buf`、`device`、`cfg`。
3. 实现 `get_observations`、`reset`、`step`。
4. 确保 extras 中的 `observations` 字典包含训练所需的 `critic`、`teacher` 或 `rnd_state`。
