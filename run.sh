#!/usr/bin/env bash
# script: run_train_with_wandb.sh

#export WANDB_MODE=offline


# （可选）设置 W&B 项目／实体，如果你想指定
# export WANDB_PROJECT="your_project_name"
# export WANDB_ENTITY="your_team_or_username"

# 激活你对应的 conda 环境（如果需要）
#conda activate SEAMP


export WANDB_API_KEY="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"
export WANDB_ENTITY="yxie667-hkust"

export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v anaconda3 | paste -sd:)

cd unitree_rl_lab

echo "enter unitree_rl_lab directory"


#运行教师play命令
# python scripts/rsl_rl/play_teacher.py --headless --task Unitree-Go2-Velocity --num_envs 64 \
#                                     --checkpoint /home/rashare/yanzhexie/State-Estimation-AMP-Lab/unitree_rl_lab/logs/rsl_rl/teacher/rsl_rl/unitree_go2_velocity/2026-04-02_20-31-56_TeacherNewTerrain4/model_6200.pt \
#                                     --device cuda:5 \
#                                     --video \
#                                     --video_length 2000 \
#                                     # --track_waypoints \
#                                     # --waypoint_ratio 0.5 \



# 运行教师训练命令
# python scripts/rsl_rl/train_teacher.py --headless --task Unitree-Go2-Velocity --num_envs 4096  --log_root /home/rashare/yanzhexie/State-Estimation-AMP-Lab/unitree_rl_lab/logs/rsl_rl/teacher  \
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
    --checkpoint  /home/rashare/yanzhexie/quadruped_robot/unitree_rl_lab/logs/student_baseline/rsl_rl_MLE/unitree_go2_velocity_lab_rough_env_v0/2026-04-21_09-44-33_baseline_bc_initnoise0.1/model_2000.pt \
    --eval_duration 200.0 \
    --num_envs 2048 \
    --device cuda:4 \
    --headless \
    #--track_waypoints




#baseline play命令，使用训练好的baseline模型进行play，并录制视频
# python scripts/rsl_rl/play_baseline.py \
#     --task Unitree-Go2-Velocity-lab-Rough-Env-v0 \
#     --checkpoint /home/rashare/yanzhexie/quadruped_robot/unitree_rl_lab/logs/student_baseline/rsl_rl_MLE/unitree_go2_velocity_lab_rough_env_v0/2026-04-21_09-44-33_baseline_bc_initnoise0.1/model_2000.pt \
#     --num_envs 256 \
#     --video \
#     --video_length 2000 \
#     --headless \
#     --device cuda:4
# #    --track_waypoints \


#baseline 训练命令
# python scripts/rsl_rl/train_baseline.py --headless --task Unitree-Go2-Velocity-lab-Rough-Env-v0 --num_envs 4096  --log_root /home/rashare/yanzhexie/quadruped_robot/unitree_rl_lab/logs/student_baseline \
#                                     --resume_path /home/rashare/yanzhexie/State-Estimation-AMP-Lab/unitree_rl_lab/logs/rsl_rl/teacher/rsl_rl/unitree_go2_velocity/2026-04-02_20-31-56_TeacherNewTerrain4/model_6200.pt \
#                                     --run_name baseline_bc+rl2\
#                                     --device cuda:6 \
                                    
                                   



