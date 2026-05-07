
---

# Environment Setup Guide

## Required Versions
- IsaacSim 4.5
- IsaacLab 2.2
- unitree_rl_lab

## Installation Steps
1. Clone the project repository:
   ```sh
   git clone https://github.com/cytochronne/State-Estimation-AMP-Lab.git
   ```
2. Follow the official IsaacLab documentation to install and configure IsaacSim and IsaacLab.
   - It is recommended to create a new conda environment for isolation.
   - Complete all dependency installations as described in the IsaacLab docs.
3. Set up unitree_rl_lab:
   - Enter the unitree_rl_lab directory and follow its README to install dependencies and configure the environment.
4. Install rl_base in editable (development) mode:
   ```sh
   cd IsaacLab
   ./isaaclab.sh -p -m pip install -e /path/to/rl_base
   ```
   - Replace `/path/to/rl_base` with the actual path to your rl_base source directory.

## Additional Notes
- Using a dedicated conda environment is strongly recommended to avoid dependency conflicts.
- If you encounter installation or runtime issues, consult the official documentation for each component and check the project's issues page.

## RL / BC Switch Instructions

To switch between PPO training and BC training, modify `TerrainAwareDistillationAlgorithmCfg` in `/home/cytochrome/pan1/quadruped_robot/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rl_base_woU_baseline_cfg.py`.

If you want to run PPO, make sure the `#RL settings` section is:

```python
RL_loss_coef=1.0
entropy_coef=0.01
```

After setting this, follow the comments in `/home/cytochrome/pan1/quadruped_robot/run.sh` and run:

```bash
bash run.sh
```

If you want to enable BC, make sure the `#setting4BC` section is:

```python
bc_loss_coef=1.0
use_mse_loss = True
```
