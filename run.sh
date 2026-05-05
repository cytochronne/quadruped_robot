#!/usr/bin/env bash
# script: run_train_with_wandb.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNITREE_RL_LAB_DIR="${PROJECT_ROOT}/unitree_rl_lab"

#export WANDB_MODE=offline


# （可选）设置 W&B 项目／实体，如果你想指定
# export WANDB_PROJECT="your_project_name"
# export WANDB_ENTITY="your_team_or_username"

# 激活你对应的 conda 环境（如果需要）
#conda activate SEAMP




export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v anaconda3 | paste -sd:)

cd "${UNITREE_RL_LAB_DIR}"

echo "enter unitree_rl_lab directory"


#运行教师play命令
# python scripts/rsl_rl/play_teacher.py --headless --task Unitree-Go2-Velocity --num_envs 64 \
#                                     --checkpoint ${PROJECT_ROOT}/unitree_rl_lab/logs/rsl_rl/teacher/rsl_rl/unitree_go2_velocity/2026-04-02_20-31-56_TeacherNewTerrain4/model_6200.pt \
#                                     --device cuda:5 \
#                                     --video \
#                                     --video_length 2000 \
#                                     # --track_waypoints \
#                                     # --waypoint_ratio 0.5 \



# 运行教师训练命令
# python scripts/rsl_rl/train_teacher.py --headless --task Unitree-Go2-Velocity --num_envs 4096  --log_root ${PROJECT_ROOT}/unitree_rl_lab/logs/rsl_rl/teacher  \
#                                     --run_name TeacherNewTerrain5 \
#                                     --device cuda:5 \
#                                     --video \
#                                     --video_length 1000 \
#                                     # --track_waypoints \
#                                     # --waypoint_ratio 0.5 \
                                    
                                    
                                    #--resume_path /home/dataset/yanzhe/SEAMPlog/rsl_rl/unitree_go2_velocity/2026-01-09_15-32-23_teacher_addWaypoints/model_10100.pt

#baseline evaluation命令，使用训练好的baseline模型进行评估,可以得到onnx
python scripts/rsl_rl/eval_baseline_policy.py \
    --task Unitree-Go2-Velocity-lab-Rough-Env-v0 \
    --checkpoint  ${PROJECT_ROOT}/unitree_rl_lab/logs/student_baseline/rsl_rl_MLE/unitree_go2_velocity_lab_rough_env_v0/2026-04-21_09-44-33_baseline_bc_initnoise0.1/model_2000.pt \
    --eval_duration 200.0 \
    --num_envs 2048 \
    --device cuda:4 \
    --headless \
    #--track_waypoints




#baseline play命令，使用训练好的baseline模型进行play，并录制视频
# python scripts/rsl_rl/play_baseline.py \
#     --task Unitree-Go2-Velocity-lab-Rough-Env-v0 \
#     --checkpoint ${PROJECT_ROOT}/unitree_rl_lab/logs/student_baseline/rsl_rl_MLE/unitree_go2_velocity_lab_rough_env_v0/2026-04-21_09-44-33_baseline_bc_initnoise0.1/model_2000.pt \
#     --num_envs 256 \
#     --video \
#     --video_length 2000 \
#     --headless \
#     --device cuda:4
# #    --track_waypoints \


#baseline 训练命令
# python scripts/rsl_rl/train_baseline.py --headless --task Unitree-Go2-Velocity-lab-Rough-Env-v0 --num_envs 4096  --log_root ${PROJECT_ROOT}/unitree_rl_lab/logs/student_baseline \
#                                     --resume_path ${PROJECT_ROOT}/unitree_rl_lab/logs/rsl_rl/teacher/rsl_rl/unitree_go2_velocity/2026-04-02_20-31-56_TeacherNewTerrain4/model_6200.pt \
#                                     --run_name baseline_bc+rl2\
#                                     --device cuda:6 \


# baseline 纯BC训练命令（关闭课程，固定BC=1.0, RL=0.0）
# python scripts/rsl_rl/train_baseline.py --headless --task Unitree-Go2-Velocity-lab-Rough-Env-v0 --num_envs 4096 --log_root ${PROJECT_ROOT}/unitree_rl_lab/logs/student_baseline \
#                                     --resume_path ${PROJECT_ROOT}/unitree_rl_lab/logs/rsl_rl/teacher/rsl_rl/unitree_go2_velocity/2026-04-02_20-31-56_TeacherNewTerrain4/model_6200.pt \
#                                     --run_name baseline_pure_bc \
#                                     --device cuda:6 \
#                                     agent.algorithm.curriculum_enable=False \
#                                     agent.algorithm.RL_loss_coef=0.0 \
#                                     agent.algorithm.bc_loss_coef=1.0 \
#                                     agent.algorithm.use_mse_loss=True \
#                                     agent.algorithm.entropy_coef=0.0


# baseline 纯RL训练命令（关闭课程，固定RL=1.0, BC=0.0）
# python scripts/rsl_rl/train_baseline.py --headless --task Unitree-Go2-Velocity-lab-Rough-Env-v0 --num_envs 4096 --log_root ${PROJECT_ROOT}/unitree_rl_lab/logs/student_baseline \
#                                     --resume_path ${PROJECT_ROOT}/unitree_rl_lab/logs/rsl_rl/teacher/rsl_rl/unitree_go2_velocity/2026-04-02_20-31-56_TeacherNewTerrain4/model_6200.pt \
#                                     --run_name baseline_pure_rl \
#                                     --device cuda:6 \
#                                     agent.algorithm.curriculum_enable=False \
#                                     agent.algorithm.RL_loss_coef=1.0 \
#                                     agent.algorithm.bc_loss_coef=0.0 \
#                                     agent.algorithm.use_mse_loss=False \
#                                     agent.algorithm.entropy_coef=0.01


# baseline TD3训练命令（本地 off-policy 实现，多环境）
# python scripts/rsl_rl/train_baseline.py --headless --task Unitree-Go2-Velocity-lab-Rough-Env-v0 --num_envs 4096 --log_root ${PROJECT_ROOT}/unitree_rl_lab/logs/student_baseline \
#                                     --run_name baseline_td3 \
#                                     --device cuda:6 \
#                                     --rl_algorithm td3 \
#                                     --offpolicy_total_timesteps 1000000 \
#                                     --offpolicy_learning_rate 3e-4 \
#                                     --offpolicy_batch_size 256 \
#                                     --offpolicy_learning_starts 10000


# baseline SAC训练命令（本地 off-policy 实现，多环境）
# python scripts/rsl_rl/train_baseline.py --headless --task Unitree-Go2-Velocity-lab-Rough-Env-v0 --num_envs 4096 --log_root ${PROJECT_ROOT}/unitree_rl_lab/logs/student_baseline \
#                                     --run_name baseline_sac \
#                                     --device cuda:6 \
#                                     --rl_algorithm sac \
#                                     --offpolicy_total_timesteps 1000000 \
#                                     --offpolicy_learning_rate 3e-4 \
#                                     --offpolicy_batch_size 256 \
#                                     --offpolicy_learning_starts 10000
