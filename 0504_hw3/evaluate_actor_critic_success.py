from pathlib import Path
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gymnasium as gym

import inverted_double_pendulum_env
from actor_critic_success_demo import ContinuousActorCritic, reset_env, step_env


ENV_ID = inverted_double_pendulum_env.ENV_ID
ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
PLOT_DIR = ROOT_DIR / "plots"
DEFAULT_MODEL_PATH = MODEL_DIR / "actor_critic_success_3000_3.pt"


def load_model(model_path, obs_dim, act_dim):
    model = ContinuousActorCritic(obs_dim, act_dim)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def choose_action(model, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean, _, _ = model(obs_t)
    action = torch.clamp(mean, -1.0, 1.0)
    return action.squeeze(0).cpu().numpy()


def moving_average(values, window=50):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    output = np.zeros_like(values)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        output[i] = values[start : i + 1].mean()
    return output


def plot_evaluation_results(results, model_path, filename):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(results["returns"]) + 1)
    path = PLOT_DIR / filename

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(episodes, results["returns"], color="tab:blue", alpha=0.25, label="Episode")
    axes[0, 0].plot(episodes, moving_average(results["returns"]), color="tab:blue", label="Average")
    axes[0, 0].set_title("Evaluation Return")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].legend()

    axes[0, 1].plot(episodes, results["lengths"], color="tab:green", alpha=0.25, label="Episode")
    axes[0, 1].plot(episodes, moving_average(results["lengths"]), color="tab:green", label="Average")
    axes[0, 1].set_title("Evaluation Length")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].legend()

    axes[1, 0].plot(episodes, results["success_rates"], color="tab:purple", label="Average")
    axes[1, 0].set_title("Success Rate")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend()

    axes[1, 1].hist(results["returns"], bins=30, color="tab:orange", alpha=0.8)
    axes[1, 1].axvline(np.mean(results["returns"]), color="black", linestyle="--", linewidth=1, label="Mean")
    axes[1, 1].set_title("Return Distribution")
    axes[1, 1].set_xlabel("Return")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()

    fig.suptitle(f"Model: {model_path.name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved evaluation plot: {path}")


def evaluate(model_path, episodes, render, plot=False, plot_name=None):
    render_mode = "human" if render else None
    env = gym.make(ENV_ID, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model = load_model(model_path, obs_dim, act_dim)

    returns = []
    lengths = []
    successes = []
    success_rates = []

    for ep in range(episodes):
        obs = reset_env(env)
        done = False
        ep_return = 0.0
        ep_length = 0

        while not done:
            action = choose_action(model, obs)
            obs, reward, done, _ = step_env(env, action)
            ep_return += reward
            ep_length += 1
            if render:
                env.render()

        success = float(ep_length >= env.unwrapped.max_steps)
        returns.append(ep_return)
        lengths.append(ep_length)
        successes.append(success)
        success_rates.append(float(np.mean(successes[-50:])))

        if render or (ep + 1) % 50 == 0 or ep == 0 or ep + 1 == episodes:
            print(f"Episode {ep + 1}: return={ep_return:.1f}, length={ep_length}, success={bool(success)}")

    env.close()

    print()
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Average return: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"Average length: {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    print(f"Success rate: {np.mean(successes):.2f}")

    results = {
        "returns": returns,
        "lengths": lengths,
        "successes": successes,
        "success_rates": success_rates,
    }

    if plot:
        if plot_name is None:
            plot_name = f"{model_path.stem}_evaluation.png"
        plot_evaluation_results(results, model_path, plot_name)

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--preview-episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--skip-preview", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.render:
        evaluate(args.model, args.episodes, True, plot=args.plot, plot_name=args.plot_name)
    else:
        if not args.skip_preview and args.preview_episodes > 0:
            print(f"Preview with render: {args.preview_episodes} episodes")
            evaluate(args.model, args.preview_episodes, True)

        print(f"Fast evaluation without render: {args.episodes} episodes")
        evaluate(args.model, args.episodes, False, plot=True, plot_name=args.plot_name)
