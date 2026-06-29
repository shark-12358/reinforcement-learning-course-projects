import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.patches import Rectangle

import simple_gridworld_env



# 超參數
LEARNING_RATE = 1e-3
DISCOUNT = 0.95
EPISODES = 2000
EPSILON_START = 0.6
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 1000


class QNet(nn.Module):
    # Q 網路：輸入狀態，輸出每個動作的 Q 值
    # 簡單的 Q 網路對應到Q-TABLE
    #QNet(state) -> [Q(s,up), Q(s,down), Q(s,left), Q(s,right)]
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def obs_to_tensor(obs, nrows, ncols, device):
    idx = int(obs[0]) * int(ncols) + int(obs[1])
    x = torch.zeros(int(nrows * ncols), dtype=torch.float32, device=device)
    x[idx] = 1.0
    return x.unsqueeze(0)


def format_policy_grid(policy_fn, row, col, walls, goal, start=None):
    # 文字版 greedy policy，方便直接在終端檢查學到的方向。
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    lines = []
    for i in range(row):
        row_out = []
        for j in range(col):
            if (i, j) in walls:
                row_out.append("W")
            elif (i, j) == goal:
                row_out.append("G")
            elif start is not None and (i, j) == start:
                row_out.append("S")
            else:
                row_out.append(arrows.get(int(policy_fn(i, j)), "."))
        lines.append(" ".join(row_out))
    return "\n".join(lines)


def moving_average(values, window=20):
    if not values:
        return np.array([])
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="same")


def plot_learning_curves(metrics, output_path):
    episodes = np.arange(1, len(metrics["returns"]) + 1)
    success_rate = moving_average(metrics["successes"], window=20)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(episodes, metrics["returns"], alpha=0.35, label="Episode return")
    axes[0].plot(episodes, moving_average(metrics["returns"]), linewidth=2, label="Moving avg")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Episode Return vs Training Episode")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(episodes, metrics["successes"], color="tab:green", alpha=0.25, linewidth=1, label="Raw success")
    axes[1].plot(episodes, success_rate, color="tab:green", linewidth=2, label="Moving avg")
    axes[1].set_ylabel("Success rate")
    axes[1].set_title("Success Rate vs Training Episode")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()

    axes[2].plot(episodes, metrics["lengths"], alpha=0.35, color="tab:orange", label="Episode length")
    axes[2].plot(episodes, moving_average(metrics["lengths"]), linewidth=2, color="tab:red", label="Moving avg")
    axes[2].set_xlabel("Training episode")
    axes[2].set_ylabel("Length")
    axes[2].set_title("Episode Length vs Training Episode")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_policy_blocks(policy_fn, info, output_path):
    row = info["row"]
    col = info["col"]
    walls = info["walls"]
    goal = info["goal"]
    start = info["start"]
    arrow_text = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, col)
    ax.set_ylim(row, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for i in range(row):
        for j in range(col):
            xy = (j, i)
            if (i, j) in walls:
                facecolor = "black"
                label = ""
                text_color = "white"
            elif (i, j) == goal:
                facecolor = "#39ff14"
                label = "G"
                text_color = "black"
            elif (i, j) == start:
                facecolor = "#d9d9d9"
                label = "S"
                text_color = "black"
            else:
                facecolor = "white"
                label = arrow_text[int(policy_fn(i, j))]
                text_color = "black"

            ax.add_patch(Rectangle(xy, 1, 1, facecolor=facecolor, edgecolor="gray", linewidth=1.2))
            if label:
                ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=16, color=text_color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def epsilon_greedy(q_net, obs, nrows, ncols, n_actions, epsilon, device):
    # 和 DQN 一樣先用 epsilon-greedy 選動作，但這裡 next_action 也會用同一策略選
    # => Deep SARSA 是 on-policy，target 會依「實際採樣到的下一步動作」計算
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        q_vals = q_net(obs_to_tensor(obs, nrows, ncols, device))
        return int(torch.argmax(q_vals, dim=1).item())


def deep_sarsa(epsilon_start=EPSILON_START):
    #Deepsarsa主要差別在下一步動作是按策略抽到的，不做 max
    #沒有Target_net
    #因為 target net 主要是在解決 DQN 的「max + function approximation」不穩定問題，Deep SARSA 沒有那個 max。
    env = gym.make("SimpleGridWorld-v0")
    nrows, ncols = env.observation_space.nvec
    n_actions = env.action_space.n

    
    device = torch.device("cpu")
    # 只需要一個 q_net（和 DQN 不同：這裡沒有 target_net、沒有 replay buffer）
    q_net = QNet(in_dim=int(nrows * ncols), out_dim=n_actions).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

    epsilon = float(epsilon_start)
    epsilon_decay = (float(epsilon_start) - EPSILON_END) / max(1, EPSILON_DECAY_EPISODES)

    all_episode_return = []
    episode_lengths = []
    successes = []

    for episode in range(EPISODES):
        obs, _ = env.reset()
        action = epsilon_greedy(q_net, obs, nrows, ncols, n_actions, epsilon, device)

        done = False
        episode_return = 0.0
        episode_length = 0
        success = 0

        while not done:
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if not done:
                # on-policy: next_action 也用 epsilon-greedy(q_net) 選
                next_action = epsilon_greedy(q_net, next_obs, nrows, ncols, n_actions, epsilon, device)
                # Deep SARSA target:
                # y = r + gamma * Q(next_obs, next_action)
                # 對照 DQN: DQN 用 max_a' Q_target(next_obs, a')（off-policy）
                target_q = reward + DISCOUNT * q_net(obs_to_tensor(next_obs, nrows, ncols, device))[0, next_action].detach()
            else:
                next_action = None
                # terminal 狀態沒有 bootstrap，target 就是 immediate reward
                target_q = torch.tensor(reward, dtype=torch.float32, device=device)

            # 取出 Q(obs, action) 並做 TD regression。
            current_q = q_net(obs_to_tensor(obs, nrows, ncols, device))[0, action]
            loss = nn.SmoothL1Loss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

            episode_return += reward
            episode_length += 1
            if tuple(next_obs) == env.unwrapped.goal:
                success = 1

            obs = next_obs
            # SARSA 會把「剛剛選好的 next_action」接到下一輪變 current action
            action = next_action if not done else None

        all_episode_return.append(episode_return)
        episode_lengths.append(episode_length)
        successes.append(success)

       
        if episode < EPSILON_DECAY_EPISODES:
            epsilon = max(EPSILON_END, epsilon - epsilon_decay)

    
    info = {
        "row": int(env.unwrapped.row),
        "col": int(env.unwrapped.col),
        "walls": set(env.unwrapped.walls),
        "goal": tuple(env.unwrapped.goal),
        "start": tuple(env.unwrapped.start),
        "nrows": int(nrows),
        "ncols": int(ncols),
    }
    metrics = {
        "returns": all_episode_return,
        "lengths": episode_lengths,
        "successes": successes,
    }

    env.close()
    return q_net, info, metrics


def run_and_save(epsilon_start=EPSILON_START):
    q_net, info, metrics = deep_sarsa(epsilon_start=epsilon_start)
    print("Deep SARSA training done.")
    device = torch.device("cpu")

    def policy_fn(i, j):
        obs = np.array([i, j], dtype=np.float32)
        with torch.no_grad():
            q_vals = q_net(obs_to_tensor(obs, info["nrows"], info["ncols"], device))
            return int(torch.argmax(q_vals, dim=1).item())

    out_dir = Path(__file__).resolve().parent / "pic" / "deep_sarsa"
    out_dir.mkdir(exist_ok=True)
    suffix = f"{epsilon_start:.1f}"
    curves_path = out_dir / f"deep_sarsa_curves_eps_{suffix}.png"
    policy_path = out_dir / f"deep_sarsa_policy_blocks_eps_{suffix}.png"

    plot_learning_curves(metrics, curves_path)
    plot_policy_blocks(policy_fn, info, policy_path)

    print("Greedy policy:")
    print(format_policy_grid(policy_fn, info["row"], info["col"], info["walls"], info["goal"], info["start"]))
    print(f"Saved learning curves to: {curves_path}")
    print(f"Saved policy blocks to: {policy_path}")


if __name__ == "__main__":
    # ?瑁? Deep SARSA 銝西撓??greedy policy
    run_and_save()
