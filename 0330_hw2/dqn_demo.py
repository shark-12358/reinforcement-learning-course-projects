import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.patches import Rectangle

import simple_gridworld_env  # Register SimpleGridWorld-v0
# 說明：
# 1. 使用 DQN（experience replay + target network）學習 Q(s, a)
# 2. replay buffer 取樣訓練，並定期更新 target 網路
# 3. 訓練完成後輸出 greedy policy 的文字格子圖


# 超參數
LEARNING_RATE = 1e-3
DISCOUNT = 0.95
EPISODES = 2000
BATCH_SIZE = 32 #每次訓練抽幾筆經驗
REPLAY_SIZE = 5000 #buffer 最多存多少經驗
TARGET_UPDATE_EVERY = 20 #每幾回合更新一次 target 網路
EPSILON_START = 0.6
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 1000


class QNet(nn.Module):
    # Q 網路：輸入狀態，輸出每個動作的 Q 值
    # 簡單的 Q 網路對應到Q-TABLE
    #QNet(state) -> [Q(s,up), Q(s,down), Q(s,left), Q(s,right)]

    def __init__(self, in_dim, out_dim):
        super().__init__()#初始化
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),#將輸入的狀態維度轉成32維的特徵空間例如[0.5, 0.3] -> [0.1, 0.4, ..., 0.2]可能隱含離終點進不進?是否靠近牆壁?這種特徵      
            nn.ReLU(),#加入非線性激活函數，若沒有的話多層 Linear 會等價於單層 Linear。
            nn.Linear(32, 32),#將前一層抽出的 32 維特徵再進一步組合與轉換。
            nn.ReLU(),
            nn.Linear(32, out_dim),#輸出層將 hiddenfeature 映射成 out_dim 維輸出
        )

    def forward(self, x):
        return self.net(x)

"""
def obs_to_tensor(obs, nrows, ncols, device):
    # 將 (row, col) 正規化後轉成 tensor
    # 將觀測正規化並轉成 tensor
    # normalize to [0,1]
    denom = np.array([max(1, nrows - 1), max(1, ncols - 1)], dtype=np.float32)
    #例如5*5的地圖進來就會變成[4,4]，避免除以0的情況，max則是指1或回傳值哪個大選哪個
    x = np.asarray(obs, dtype=np.float32) / denom#將資料變成0~1區間的範圍
    return torch.from_numpy(x).to(device).unsqueeze(0)
    #unsqueeze(0)是指在第0維新增一個維度，將原本的 (2,) 變成 (1, 2)，這樣就符合神經網路輸入的 batch 維度要求。
"""
#將座標改成one-hot encoding，讓網路更容易學習每個格子的位置。
def obs_to_tensor(obs, nrows, ncols, device):
    idx = int(obs[0]) * int(ncols) + int(obs[1])
    x = torch.zeros(int(nrows * ncols), dtype=torch.float32, device=device)#建立全0矩陣
    x[idx] = 1.0#將座標改成1
    return x.unsqueeze(0)


def moving_average(values, window=20):
    # 用移動平均讓訓練曲線更容易觀察趨勢。
    if not values:
        return np.array([])
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="same")


def format_policy_grid(policy_fn, info):
    # 文字版 greedy policy，方便直接在終端檢查學到的方向。
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    lines = []
    for i in range(info["row"]):
        row_out = []
        for j in range(info["col"]):
            pos = (i, j)
            if pos in info["walls"]:
                row_out.append("W")
            elif pos == info["goal"]:
                row_out.append("G")
            elif pos == info["start"]:
                row_out.append("S")
            else:
                row_out.append(arrows[int(policy_fn(i, j))])
        lines.append(" ".join(row_out))
    return "\n".join(lines)


def plot_learning_curves(metrics, output_path):
    # 畫出題目要求的三條曲線：return / success rate / episode length。
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
    # 畫出方塊版 policy 圖，外觀接近作業題目的格子圖。
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


def dqn(epsilon_start=EPSILON_START):
    # DQN 主訓練流程：
    # 1. 用 q_net 估計 Q 值
    # 2. 用 replay buffer 打散資料
    # 3. 用 target_net 產生較穩定的 target
    env = gym.make("SimpleGridWorld-v0")
    nrows, ncols = env.observation_space.nvec
    n_actions = env.action_space.n

    device = torch.device("cpu")
    # q_net 是正在學的主網路，target_net 用來提供 target Q 值。
    q_net = QNet(in_dim=int(nrows * ncols), out_dim=n_actions).to(device)
    target_net = QNet(in_dim=int(nrows * ncols), out_dim=n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    # replay buffer 會存 (state, action, reward, next_state, done)
    buffer = deque(maxlen=REPLAY_SIZE)

    epsilon = float(epsilon_start)
    epsilon_decay = (float(epsilon_start) - EPSILON_END) / max(1, EPSILON_DECAY_EPISODES)

    all_episode_return = []
    episode_lengths = []
    successes = []

    for episode in range(EPISODES):
        # 每回合重置環境
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        success = 0

        while not done:
            # epsilon-greedy：有機率隨機探索，否則選 q_net 預測值最大的動作。
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    #這裡沒有訓練!!!!!!，只是把目前的狀態丟進 q_net 看看它覺得哪個動作的 Q 值最高，就選那個動作。
                    q_vals = q_net(obs_to_tensor(obs, nrows, ncols, device))
                    action = int(torch.argmax(q_vals, dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.append((obs, action, reward, next_obs, done))

            episode_return += reward
            episode_length += 1
            if tuple(next_obs) == env.unwrapped.goal:
                success = 1

            obs = next_obs

            # 經驗夠多才開始抽 batch 訓練。
            if len(buffer) >= BATCH_SIZE:
                # 從 replay buffer 取樣並更新 Q 網路
                batch = random.sample(buffer, BATCH_SIZE)
                b_obs, b_act, b_rew, b_next, b_done = zip(*batch)
                #將抓出來的資料轉換成tenser
                obs_t = torch.cat([obs_to_tensor(o, nrows, ncols, device) for o in b_obs], dim=0)#cat是把既有tensor接起來
                next_t = torch.cat([obs_to_tensor(o, nrows, ncols, device) for o in b_next], dim=0)
                act_t = torch.tensor(b_act, dtype=torch.int64, device=device).unsqueeze(1)
                rew_t = torch.tensor(b_rew, dtype=torch.float32, device=device).unsqueeze(1)
                done_t = torch.tensor(b_done, dtype=torch.float32, device=device).unsqueeze(1)

                # 取出 q_net 對「實際採取 action」的 Q 值估計。
                q_values = q_net(obs_t).gather(1, act_t)
                # 從 q_net 輸出的所有 action Q-value 中， 取出每筆 sample 實際採取 action 的 Q(s,a)

                with torch.no_grad():
                    # target_net 負責提供下一步的 max Q，避免 target 一直跟著主網路抖動。
                    max_next_q = target_net(next_t).max(dim=1, keepdim=True)[0]
                    target = rew_t + (1.0 - done_t) * DISCOUNT * max_next_q
                    #根據 DQN 的目標計算方式，若 done_t 為 1（表示該筆經驗是終止狀態），則 target 就是 reward；若 done_t 為 0，則 target 是 reward 加上折扣後的 max_next_q。

                # 用 MSE loss 讓 q_net 的輸出逼近 Q-learning target。
                loss = nn.SmoothL1Loss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()

        all_episode_return.append(episode_return)
        episode_lengths.append(episode_length)
        successes.append(success)

        # 前期多探索，後期逐漸降低 epsilon。
        if episode < EPSILON_DECAY_EPISODES:
            epsilon = max(EPSILON_END, epsilon - epsilon_decay)

        # 每隔固定回合同步一次 target network。
        if (episode + 1) % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(q_net.state_dict())

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
    # 訓練完成後，輸出圖表與 policy 視覺化。
    q_net, info, metrics = dqn(epsilon_start=epsilon_start)
    print("DQN training done.")

    device = torch.device("cpu")

    def policy_fn(i, j):
        obs = np.array([i, j], dtype=np.float32)
        with torch.no_grad():
            q_vals = q_net(obs_to_tensor(obs, info["nrows"], info["ncols"], device))
            return int(torch.argmax(q_vals, dim=1).item())

    out_dir = Path(__file__).resolve().parent / "pic" / "dqn"
    out_dir.mkdir(exist_ok=True)
    suffix = f"{epsilon_start:.1f}"
    curves_path = out_dir / f"dqn_curves_eps_{suffix}.png"
    policy_path = out_dir / f"dqn_policy_blocks_eps_{suffix}.png"

    plot_learning_curves(metrics, curves_path)
    plot_policy_blocks(policy_fn, info, policy_path)

    print("Greedy policy:")
    print(format_policy_grid(policy_fn, info))
    print(f"Saved learning curves to: {curves_path}")
    print(f"Saved policy blocks to: {policy_path}")


if __name__ == "__main__":
    run_and_save()
