from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import simple_gridworld_env  # Register SimpleGridWorld-v0

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 500

START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2


def state_to_idx(state, ncols):
    return int(state[0]) * ncols + int(state[1])


def epsilon_greedy(q_table, state_idx, n_actions, epsilon):
    if np.random.random() > epsilon:
        return int(np.argmax(q_table[state_idx]))
    return np.random.randint(0, n_actions)


def moving_average(values, window=20):
    if not values:
        return np.array([])
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="same")


def format_policy_grid(policy_fn, info):
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


def sarsa(epsilon_start=0.5):
    env = gym.make("SimpleGridWorld-v0")
    nrows, ncols = env.observation_space.nvec
    n_states = int(nrows * ncols)
    n_actions = env.action_space.n

    q_table = np.random.uniform(low=-0.1, high=0.1, size=(n_states, n_actions))
    epsilon = float(epsilon_start)
    epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    all_episode_return = []
    episode_lengths = []
    successes = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        s = state_to_idx(state, ncols)
        action = epsilon_greedy(q_table, s, n_actions, epsilon)
        done = False
        episode_return = 0.0
        episode_length = 0
        success = 0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ns = state_to_idx(next_state, ncols)

            if not done:
                next_action = epsilon_greedy(q_table, ns, n_actions, epsilon)
                current_q = q_table[s, action]
                next_q = q_table[ns, next_action]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * next_q)
                q_table[s, action] = new_q
            else:
                q_table[s, action] = reward
                next_action = None

            episode_return += reward
            episode_length += 1
            if tuple(next_state) == env.unwrapped.goal:
                success = 1

            state = next_state
            s = ns
            action = next_action if not done else None

        all_episode_return.append(episode_return)
        episode_lengths.append(episode_length)
        successes.append(success)

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

    info = {
        "row": int(env.unwrapped.row),
        "col": int(env.unwrapped.col),
        "walls": set(env.unwrapped.walls),
        "goal": tuple(env.unwrapped.goal),
        "start": tuple(env.unwrapped.start),
    }
    metrics = {
        "returns": all_episode_return,
        "lengths": episode_lengths,
        "successes": successes,
    }

    env.close()
    return q_table, info, metrics


def run_and_save(epsilon_start=0.5):
    q_table, info, metrics = sarsa(epsilon_start=epsilon_start)
    print("Q-table shape:", q_table.shape)

    def policy_fn(i, j):
        s = i * info["col"] + j
        return int(np.argmax(q_table[s]))

    out_dir = Path(__file__).resolve().parent / "pic" / "sarsa"
    out_dir.mkdir(exist_ok=True)
    suffix = f"{epsilon_start:.1f}"
    curves_path = out_dir / f"sarsa_curves_eps_{suffix}.png"
    policy_path = out_dir / f"sarsa_policy_blocks_eps_{suffix}.png"

    plot_learning_curves(metrics, curves_path)
    plot_policy_blocks(policy_fn, info, policy_path)

    print("Greedy policy:")
    print(format_policy_grid(policy_fn, info))
    print(f"Saved learning curves to: {curves_path}")
    print(f"Saved policy blocks to: {policy_path}")


if __name__ == "__main__":
    run_and_save()
