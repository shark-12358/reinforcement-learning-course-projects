# 強化學習課程專案作品集

這個 repository 收錄我在強化學習課程中的作業與期末專案實作，內容涵蓋 Dynamic Programming、Model-Free RL、Deep RL、Actor-Critic，以及以 PyBullet/URDF 建立的機器人控制任務。

## 專案亮點

- 實作 Value Iteration 與 Policy Iteration，並視覺化最佳 value function 與 policy。
- 在自訂 Gymnasium GridWorld 中比較 SARSA、Q-learning、Deep SARSA 與 DQN。
- 使用 PyTorch 實作 Actor-Critic，應用於 inverted double pendulum 控制任務。
- 期末專案實作 ledge-climbing robot 控制，包含 PPO、DDPG、URDF 模擬環境、domain randomization、sensor noise 與 robust policy 評估。

## Repository 結構

```text
.
|-- 0309_hw1/                # Dynamic Programming: Value Iteration / Policy Iteration
|-- 0330_hw2/                # SARSA, Q-learning, Deep SARSA, DQN
|-- 0504_hw3/                # Actor-Critic experiments
|-- Final_Project/
|   |-- code/
|   |   |-- URDFs/           # Robot and environment assets
|   |   |-- with_robust/     # Robust PPO implementation
|   |   `-- without_robust/  # Baseline PPO implementation
|   `-- Final_docs/          # Final report and project documentation
|-- requirements.txt
`-- README.md
```

## 專案內容

### HW1: Dynamic Programming

實作 gridworld 類型環境中的 Value Iteration 與 Policy Iteration，並比較不同終止狀態設定下的最佳 value function 與 policy。

主要檔案：

- `0309_hw1/main/value_iteration/value_iteration.py`
- `0309_hw1/main/value_iteration/value_iteration(non_terminal).py`
- `0309_hw1/main/policy_iteration/policy_iteration.py`
- `0309_hw1/main/policy_iteration/policy_iteration_nonterminal.py`

### HW2: Model-Free RL 與 Deep RL

在自訂 Gymnasium GridWorld 環境中實作並比較傳統表格型 RL 與神經網路方法。

實作演算法：

- SARSA
- Q-learning
- Deep SARSA
- Deep Q-Network

主要檔案：

- `0330_hw2/simple_gridworld_env.py`
- `0330_hw2/sarsa_demo.py`
- `0330_hw2/q_learning_demo.py`
- `0330_hw2/deep_sarsa_demo.py`
- `0330_hw2/dqn_demo.py`

### HW3: Actor-Critic Control

使用 PyTorch 與 PyBullet 實作 Actor-Critic 相關方法，應用於 inverted double pendulum 控制任務，並比較 baseline 與 target-network 版本。

主要檔案：

- `0504_hw3/inverted_double_pendulum_env.py`
- `0504_hw3/actor_critic_demo.py`
- `0504_hw3/actor_critic_with_baseline_demo.py`
- `0504_hw3/actor_critic_with_target_demo.py`
- `0504_hw3/actor_critic_success_demo.py`

### Final Project: Robust Ledge-Climbing Robot

期末專案目標是訓練一個模擬 ledge-climbing robot。環境提供 hand-designed reference gait，policy 學習 residual joint-position correction，讓機器人能在不同 ledge 環境中完成攀爬任務。

主要特色：

- Baseline PPO 與 Robust PPO 訓練流程。
- DDPG 實驗作為比較。
- 多環境訓練與 unseen environment validation。
- Friction domain randomization。
- Sensor observation noise。
- PPO-side observation filtering。
- 本地實驗會輸出 CSV logs、plots 與 rollout videos。

主要檔案：

- `Final_Project/code/with_robust/envs_robust.py`
- `Final_Project/code/with_robust/Ledge_Climb_Robot_PPO_robust.py`
- `Final_Project/code/without_robust/envs_without_robust.py`
- `Final_Project/code/without_robust/Ledge_Climb_Robot_PPO_without_robust.py`
- `Final_Project/Final_docs/robust_PPO_project_doc.md`
- `Final_Project/Final_docs/Final_Project.pdf`

## 環境安裝

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 執行範例

執行 GridWorld 相關實驗：

```bash
python 0330_hw2/sarsa_demo.py
python 0330_hw2/q_learning_demo.py
python 0330_hw2/dqn_demo.py
```

執行 Actor-Critic 實驗：

```bash
python 0504_hw3/actor_critic_demo.py
python 0504_hw3/actor_critic_with_baseline_demo.py
```

執行期末專案 Robust PPO：

```bash
cd Final_Project/code/with_robust
python Ledge_Climb_Robot_PPO_robust.py
```

執行期末專案 Baseline PPO：

```bash
cd Final_Project/code/without_robust
python Ledge_Climb_Robot_PPO_without_robust.py
```

## 大型檔案說明

為了讓 GitHub repository 保持乾淨且適合面試展示，以下本地實驗輸出不納入 Git：

- 訓練模型權重：`*.pt`, `*.pth`, `*.ckpt`
- Rollout videos：`*.mp4`
- 壓縮檔：`*.zip`
- 期末專案大量訓練輸出：`Final_Project/results/`
- Python cache：`__pycache__/`, `*.pyc`

這個 repository 主要保留原始碼、報告、必要 URDF assets，以及代表性的結果圖表。

## 使用技術

- Python
- PyTorch
- Gymnasium
- PyBullet
- NumPy
- Matplotlib
- Reinforcement Learning: Value Iteration, Policy Iteration, SARSA, Q-learning, DQN, Actor-Critic, PPO, DDPG
