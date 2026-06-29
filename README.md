# Reinforcement Learning Course Projects

This repository collects my reinforcement learning coursework and final project implementations. It includes tabular dynamic programming, model-free control, deep Q-learning, actor-critic methods, and a robust robotic control project using PyBullet simulation.

## Highlights

- Implemented value iteration and policy iteration for grid-based decision problems.
- Compared SARSA, Q-learning, Deep SARSA, and DQN on a custom Gymnasium gridworld.
- Built actor-critic agents for an inverted double pendulum environment.
- Developed a final ledge-climbing robot project using PPO, DDPG, URDF assets, domain randomization, observation noise, and robustness evaluation.

## Repository Structure

```text
.
├── 0309_hw1/                # Dynamic programming: value iteration and policy iteration
├── 0330_hw2/                # SARSA, Q-learning, Deep SARSA, DQN
├── 0504_hw3/                # Actor-critic experiments
├── Final_Project/
│   ├── code/
│   │   ├── URDFs/           # Robot and environment assets
│   │   ├── with_robust/     # Robust PPO implementation
│   │   └── without_robust/  # Baseline PPO implementation
│   └── Final_docs/          # Final report and project documentation
├── requirements.txt
└── README.md
```

## Project Summary

### HW1: Dynamic Programming

Implemented value iteration and policy iteration for gridworld-style environments. The experiments visualize optimal value functions and policies under different terminal-state settings.

Key files:

- `0309_hw1/main/value_iteration/value_iteration.py`
- `0309_hw1/main/value_iteration/value_iteration(non_terminal).py`
- `0309_hw1/main/policy_iteration/policy_iteration.py`
- `0309_hw1/main/policy_iteration/policy_iteration_nonterminal.py`

### HW2: Model-Free and Deep RL

Implemented and compared classic tabular RL and neural-network-based methods in a custom Gymnasium environment.

Algorithms:

- SARSA
- Q-learning
- Deep SARSA
- Deep Q-Network

Key files:

- `0330_hw2/simple_gridworld_env.py`
- `0330_hw2/sarsa_demo.py`
- `0330_hw2/q_learning_demo.py`
- `0330_hw2/deep_sarsa_demo.py`
- `0330_hw2/dqn_demo.py`

### HW3: Actor-Critic Control

Implemented actor-critic variants for an inverted double pendulum task using PyTorch and PyBullet.

Key files:

- `0504_hw3/inverted_double_pendulum_env.py`
- `0504_hw3/actor_critic_demo.py`
- `0504_hw3/actor_critic_with_baseline_demo.py`
- `0504_hw3/actor_critic_with_target_demo.py`
- `0504_hw3/actor_critic_success_demo.py`

### Final Project: Robust Ledge-Climbing Robot

The final project trains a simulated ledge-climbing robot. The policy learns residual joint-position corrections on top of a hand-designed reference gait.

Main implementation features:

- PPO training for baseline and robust settings.
- DDPG experiments for comparison.
- Multi-environment training over different ledge layouts.
- Unseen environment validation.
- Friction domain randomization.
- Sensor observation noise.
- PPO-side observation filtering.
- CSV logging, plots, and rollout videos during local experiments.

Key files:

- `Final_Project/code/with_robust/envs_robust.py`
- `Final_Project/code/with_robust/Ledge_Climb_Robot_PPO_robust.py`
- `Final_Project/code/without_robust/envs_without_robust.py`
- `Final_Project/code/without_robust/Ledge_Climb_Robot_PPO_without_robust.py`
- `Final_Project/Final_docs/robust_PPO_project_doc.md`
- `Final_Project/Final_docs/Final_Project.pdf`

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example Commands

Run tabular or deep RL gridworld experiments:

```bash
python 0330_hw2/sarsa_demo.py
python 0330_hw2/q_learning_demo.py
python 0330_hw2/dqn_demo.py
```

Run actor-critic experiments:

```bash
python 0504_hw3/actor_critic_demo.py
python 0504_hw3/actor_critic_with_baseline_demo.py
```

Run final project PPO experiments:

```bash
cd Final_Project/code/with_robust
python Ledge_Climb_Robot_PPO_robust.py
```

```bash
cd Final_Project/code/without_robust
python Ledge_Climb_Robot_PPO_without_robust.py
```

## Notes on Large Files

Large local experiment outputs are intentionally excluded from Git:

- Trained model checkpoints: `*.pt`, `*.pth`, `*.ckpt`
- Rollout videos: `*.mp4`
- Generated final-project experiment logs under `Final_Project/results/`
- Temporary Python cache files

This keeps the repository focused on source code, documentation, reusable assets, and representative reports.

## Technologies

- Python
- PyTorch
- Gymnasium
- PyBullet
- NumPy
- Matplotlib
- Reinforcement learning algorithms: value iteration, policy iteration, SARSA, Q-learning, DQN, actor-critic, PPO, DDPG
