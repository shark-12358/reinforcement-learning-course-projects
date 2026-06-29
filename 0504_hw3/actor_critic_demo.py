# 連續控制演算法提示
# - 由於 action_space 是 Box([-1], [1])，policy 不能再用 Categorical，應改用 Gaussian policy。
# - actor 可以輸出 mean 與 log_std，再由 Normal(mean, std) 抽樣 action。
# - 訓練時可抽樣 action 以保留探索；測試或 render 時可直接使用 mean 當 deterministic action。
# - 抽樣後的 action 要限制在環境可接受的範圍內，例如 [-1, 1]。

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import inverted_double_pendulum_env


# 本檔示範 Actor-Critic without baseline。
# 任務使用自製雙倒單擺車環境，policy 使用 Gaussian policy 處理連續 action。
# actor 更新時直接使用 n-step return R_t，不扣除 value function baseline。
ENV_ID = inverted_double_pendulum_env.ENV_ID
EPISODES = 1000
GAMMA = 0.99
LR = 1e-3
T_MAX = 5
RENDER_EPISODES = 3
PLOT_DIR = Path(__file__).resolve().parent / "plots"
SUCCESS_WINDOW = 50
RETURN_COLOR = "tab:blue"
LENGTH_COLOR = "tab:green"
SUCCESS_COLOR = "tab:purple"
ACTOR_COLOR = "tab:cyan"
CRITIC_COLOR = "tab:orange"
TD_COLOR = "tab:red"


class ContinuousActorCritic(nn.Module):
    """連續 action 版本的 actor-critic 網路。"""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mean = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.value = nn.Linear(64, 1)#這個就是CRITIC寶貝。

    def forward(self, x):#主要是將 mean作為常態分佈中心，探索範圍為std
        h = self.shared(x)
        mean = torch.tanh(self.mean(h))
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.value(h)
        return mean, std, value


def reset_env(env):
    """處理 Gymnasium 的 reset 回傳格式。"""
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    """處理 Gymnasium 的 step 回傳格式。"""
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return next_obs, reward, done, info


def sample_action(model, obs):
    """由 Gaussian policy 抽樣連續 action，並回傳該 action 的 log probability。"""
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    mean, std, value = model(obs_t)#這裡呼叫模型計算三個值。
    dist = torch.distributions.Normal(mean, std)#在這裡創建常態分佈
    raw_action = dist.sample()
    action = torch.clamp(raw_action, -1.0, 1.0)
    log_prob = dist.log_prob(raw_action).sum(dim=-1)
    env_action = action.squeeze(0).detach().cpu().numpy()
    return env_action, log_prob.squeeze(0), value.squeeze(0).squeeze(-1)


def compute_n_step_returns(rewards, next_value, done):
    """由 rollout reward 倒推 n-step return。"""
    returns = []
    R = torch.zeros_like(next_value) if done else next_value#如果已經DONE就是0如果不是就是下一格這樣
    for reward in reversed(rewards):
        R = reward + GAMMA * R
        returns.insert(0, R)
    return torch.stack(returns)


def moving_average(values, window=SUCCESS_WINDOW):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    output = np.zeros_like(values)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        output[i] = values[start : i + 1].mean()
    return output


def plot_training_metrics(metrics, filename):
    import os

    os.makedirs(PLOT_DIR, exist_ok=True)
    episodes = np.arange(1, len(metrics["returns"]) + 1)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    axes[0, 0].plot(episodes, metrics["returns"], color=RETURN_COLOR, alpha=0.25, label="Episode")
    axes[0, 0].plot(episodes, moving_average(metrics["returns"]), color=RETURN_COLOR, label="Average")
    axes[0, 0].set_title("Episode Return")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].legend()

    axes[0, 1].plot(episodes, metrics["lengths"], color=LENGTH_COLOR, alpha=0.25, label="Episode")
    axes[0, 1].plot(episodes, moving_average(metrics["lengths"]), color=LENGTH_COLOR, label="Average")
    axes[0, 1].set_title("Episode Length")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].legend()

    axes[1, 0].plot(episodes, metrics["success_rates"], color=SUCCESS_COLOR, alpha=0.25, label="Episode")
    axes[1, 0].plot(episodes, moving_average(metrics["success_rates"]), color=SUCCESS_COLOR, label="Average")
    axes[1, 0].set_title("Success Rate")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel(f"Rate ({SUCCESS_WINDOW}-episode window)")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend()

    axes[1, 1].plot(episodes, metrics["actor_losses"], color=ACTOR_COLOR, alpha=0.25, label="Actor")
    axes[1, 1].plot(episodes, moving_average(metrics["actor_losses"]), color=ACTOR_COLOR, label="Actor avg")
    axes[1, 1].plot(episodes, metrics["critic_losses"], color=CRITIC_COLOR, alpha=0.25, label="Critic")
    axes[1, 1].plot(episodes, moving_average(metrics["critic_losses"]), color=CRITIC_COLOR, label="Critic avg")
    axes[1, 1].set_title("Actor Loss and Critic Loss")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()

    axes[2, 0].plot(episodes, metrics["td_errors"], color=TD_COLOR, alpha=0.25, label="Episode")
    axes[2, 0].plot(episodes, moving_average(metrics["td_errors"]), color=TD_COLOR, label="Average")
    axes[2, 0].set_title("TD Error")
    axes[2, 0].set_xlabel("Episode")
    axes[2, 0].set_ylabel("Mean absolute TD error")
    axes[2, 0].legend()

    axes[2, 1].axis("off")

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path}")


def actor_critic():
    """
    Actor-Critic without baseline。

    actor 直接使用 n-step return 當作 policy gradient 權重：
        actor_loss = -log pi(a|s) * R_t

    critic 仍然學習 V(s)，但 V(s) 不會從 actor 權重中扣掉。
    """
    env = gym.make(ENV_ID)
    obs = reset_env(env)

    obs_dim = int(np.prod(np.array(obs).shape))
    act_dim = int(np.prod(env.action_space.shape))

    model = ContinuousActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    metrics = {
        "returns": [],
        "lengths": [],
        "success_rates": [],
        "actor_losses": [],
        "critic_losses": [],
        "td_errors": [],
    }
    successes = []

    for ep in range(EPISODES):
        obs = reset_env(env)
        done = False
        ep_return = 0.0
        ep_length = 0
        ep_actor_losses = []
        ep_critic_losses = []
        ep_td_errors = []

        while not done:
            log_probs = []
            values = []
            rewards = []

            # 收集一段最多 T_MAX 步的 rollout。
            for _ in range(T_MAX):
                action, log_prob, value = sample_action(model, obs)
                next_obs, reward, done, _ = step_env(env, action)
                ep_return += reward
                ep_length += 1

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor(reward, dtype=torch.float32))

                obs = next_obs
                if done:
                    break

            with torch.no_grad():#如果已經DONE就是0如果不是就用 critic 估計最後狀態的 value
                if done:
                    next_value = torch.zeros(())
                else:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    _, _, next_value = model(obs_t)
                    next_value = next_value.squeeze()

            returns = compute_n_step_returns(rewards, next_value, done)
            values_t = torch.stack(values)
            log_probs_t = torch.stack(log_probs)

            # without baseline：actor 直接用 n-step return 更新。
            actor_loss = -(log_probs_t * returns.detach()).mean()
            critic_loss = (returns - values_t).pow(2).mean()
            td_error = returns - values_t
            loss = actor_loss + 0.5 * critic_loss
            ep_actor_losses.append(float(actor_loss.detach().cpu()))
            ep_critic_losses.append(float(critic_loss.detach().cpu()))
            ep_td_errors.append(float(td_error.detach().abs().mean().cpu()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        success = float(ep_length >= env.unwrapped.max_steps)
        successes.append(success)
        metrics["returns"].append(ep_return)
        metrics["lengths"].append(ep_length)
        metrics["success_rates"].append(float(np.mean(successes[-SUCCESS_WINDOW:])))
        metrics["actor_losses"].append(float(np.mean(ep_actor_losses)))
        metrics["critic_losses"].append(float(np.mean(ep_critic_losses)))
        metrics["td_errors"].append(float(np.mean(ep_td_errors)))

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}, return: {ep_return:.1f}")

    plot_training_metrics(metrics, "actor_critic_without_baseline.png")
    env.close()
    return model


def render_trained_policy(model, episodes=RENDER_EPISODES):
    """訓練完成後，用 policy mean 進行渲染。"""
    render_env = gym.make(ENV_ID, render_mode="human")
    model.eval()

    for ep in range(episodes):
        obs = reset_env(render_env)
        done = False
        ep_return = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, _, _ = model(obs_t)
            action = mean.squeeze(0).cpu().numpy()

            obs, reward, done, _ = step_env(render_env, action)
            ep_return += reward
            render_env.render()

        print(f"[Render] Episode {ep + 1}, return: {ep_return:.1f}")

    render_env.close()


if __name__ == "__main__":
    model = actor_critic()
    render_trained_policy(model, episodes=RENDER_EPISODES)
