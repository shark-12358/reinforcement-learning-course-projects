from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import matplotlib.pyplot as plt

from envs_without_robust import LedgeClimbEnv, MAX_EPISODE_SECONDS



RUN_DIR = Path(__file__).resolve().parent
CODE_ROOT = RUN_DIR.parent
PROJECT_ROOT = CODE_ROOT.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "PPO" / "without_robust"

TRAIN_ENVS = ["1"]                 # Choose envs to train, e.g. ["1"], ["2"], or ["1", "2", "3", "4"].
SEED = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_TIMESTEPS = 400_000
ROLLOUT_LENGTH = 2048
MINIBATCH_SIZE = 256
PPO_EPOCHS = 10

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.20
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.50
MAX_GRAD_NORM = 0.50
TARGET_KL: Optional[float] = None   # Keep None for no PPO KL early stopping.

HIDDEN_SIZE = 256
LOG_STD_INIT = -0.5
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
OBS_CLIP = 5.0

EVAL_INTERVAL = 10_000
EVAL_EPISODES = 5
MAX_EPISODE_SECONDS_TRAIN = MAX_EPISODE_SECONDS

ROLLING_WINDOW = 20
SAVE_FINAL_VIDEO = True
ONLY_RECORD_VIDEO = False
RECORD_MODEL_ENV_ID = "1"          # Load results/PPO/without_robust/env{ID}/best_model/best.pt when ONLY_RECORD_VIDEO is True.
RECORD_TEST_ENVS = ["1"]           # Rollout envs when ONLY_RECORD_VIDEO is True, e.g. ["1", "2", "3", "4"].
RECORD_MODEL_FILENAME = "best.pt"
VIDEO_MAX_EPISODES = 1
PRINT_EVERY_EPISODES = 1
PRINT_EVERY_UPDATES = 1


@dataclass
class PPOConfig:
    total_timesteps: int = TOTAL_TIMESTEPS
    rollout_length: int = ROLLOUT_LENGTH
    minibatch_size: int = MINIBATCH_SIZE
    ppo_epochs: int = PPO_EPOCHS
    gamma: float = GAMMA
    gae_lambda: float = GAE_LAMBDA
    clip_ratio: float = CLIP_RATIO
    learning_rate: float = LEARNING_RATE
    entropy_coef: float = ENTROPY_COEF
    value_coef: float = VALUE_COEF
    max_grad_norm: float = MAX_GRAD_NORM
    target_kl: Optional[float] = TARGET_KL
    hidden_size: int = HIDDEN_SIZE
    log_std_init: float = LOG_STD_INIT
    log_std_min: float = LOG_STD_MIN
    log_std_max: float = LOG_STD_MAX
    obs_clip: float = OBS_CLIP
    eval_interval: int = EVAL_INTERVAL
    eval_episodes: int = EVAL_EPISODES
    max_episode_seconds: float = MAX_EPISODE_SECONDS_TRAIN
    seed: int = SEED
    device: str = DEVICE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_result_dirs(env_id: str) -> Dict[str, Path]:
    env_root = RESULTS_ROOT / f"env{env_id}"
    dirs = {
        "root": env_root,
        "best_model": env_root / "best_model",
        "csv": env_root / "csv",
        "plot": env_root / "plot",
        "video": env_root / "video",
    }
    return dirs


def get_cross_eval_dirs(model_env_id: str, test_env_id: str) -> Dict[str, Path]:
    eval_root = RESULTS_ROOT / "cross_eval" / f"model_env{model_env_id}" / f"test_env{test_env_id}"
    dirs = {
        "root": eval_root,
        "best_model": eval_root / "best_model",
        "csv": eval_root / "csv",
        "plot": eval_root / "plot",
        "video": eval_root / "video",
    }
    return dirs


def ensure_dirs(dirs: Dict[str, Path]) -> Dict[str, Path]:
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def ensure_result_dirs(env_id: str) -> Dict[str, Path]:
    return ensure_dirs(get_result_dirs(env_id))


def ensure_cross_eval_dirs(model_env_id: str, test_env_id: str) -> Dict[str, Path]:
    return ensure_dirs(get_cross_eval_dirs(model_env_id, test_env_id))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    window = max(1, int(window))
    output = np.empty_like(values, dtype=np.float64)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        output[i] = np.nanmean(values[start : i + 1])
    return output


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y < 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


class CSVLogger:
    def __init__(self, path: Path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def write(self, row: Dict[str, object]) -> None:
        clean_row = {name: row.get(name, "") for name in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean_row)


class RunningMeanStd:
    """Online observation normalization with finite-value protection."""

    def __init__(self, shape, epsilon: float = 1e-4, clip: float = OBS_CLIP):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)
        self.clip = float(clip)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = float(total_count)

    def normalize(self, obs: np.ndarray, update: bool) -> Tuple[np.ndarray, int]:
        obs = np.asarray(obs, dtype=np.float32)
        invalid_count = int(np.sum(~np.isfinite(obs)))
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        if update:
            self.update(obs)
        norm = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        norm = np.clip(norm, -self.clip, self.clip)
        norm = np.nan_to_num(norm, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return norm.astype(np.float32), invalid_count

    def state_dict(self) -> Dict[str, object]:
        return {
            "mean": self.mean.astype(np.float64).tolist(),
            "var": self.var.astype(np.float64).tolist(),
            "count": float(self.count),
            "clip": float(self.clip),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
        self.clip = float(state.get("clip", self.clip))


class RolloutBuffer:
    def __init__(self, obs_dim: int, act_dim: int):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clear()

    def clear(self) -> None:
        self.observations: List[np.ndarray] = []
        self.raw_actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.next_values: List[float] = []
        self.rewards: List[float] = []
        self.terminated: List[bool] = []
        self.done: List[bool] = []
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(
        self,
        obs: np.ndarray,
        raw_action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        terminated: bool,
        done: bool,
        next_value: float,
    ) -> None:
        self.observations.append(np.asarray(obs, dtype=np.float32))
        self.raw_actions.append(np.asarray(raw_action, dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.done.append(bool(done))

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_gae(self, gamma: float, gae_lambda: float) -> None:
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_advantage = 0.0

        for t in reversed(range(n)):
            # Terminated episodes should not bootstrap. Time-limit truncations bootstrap
            # from the final observation but do not propagate GAE across the reset.
            bootstrap_mask = 0.0 if self.terminated[t] else 1.0
            continuation_mask = 0.0 if self.done[t] else 1.0
            delta = self.rewards[t] + gamma * bootstrap_mask * self.next_values[t] - self.values[t]
            last_advantage = delta + gamma * gae_lambda * continuation_mask * last_advantage
            advantages[t] = last_advantage

        values = np.asarray(self.values, dtype=np.float32)
        self.advantages = advantages
        self.returns = advantages + values

    def tensors(self, device: torch.device):
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_gae() before requesting tensors.")
        return {
            "obs": torch.as_tensor(np.asarray(self.observations), dtype=torch.float32, device=device),
            "raw_actions": torch.as_tensor(np.asarray(self.raw_actions), dtype=torch.float32, device=device),
            "old_log_probs": torch.as_tensor(np.asarray(self.log_probs), dtype=torch.float32, device=device),
            "values": torch.as_tensor(np.asarray(self.values), dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(self.advantages, dtype=torch.float32, device=device),
            "returns": torch.as_tensor(self.returns, dtype=torch.float32, device=device),
        }


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
                nn.init.constant_(layer.bias, 0.0)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, config: PPOConfig):
        super().__init__()
        self.actor = MLP(obs_dim, act_dim, config.hidden_size)
        self.critic = MLP(obs_dim, 1, config.hidden_size)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(config.log_std_init)))
        self.log_std_min = float(config.log_std_min)
        self.log_std_max = float(config.log_std_max)

    def get_dist_params(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        log_std = log_std.expand_as(mean)
        return mean, log_std

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, action_limit: np.ndarray, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = ActorCritic(obs_dim, act_dim, config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.action_limit = torch.as_tensor(action_limit, dtype=torch.float32, device=self.device)
        self.eps = 1e-6

    def _log_prob_from_raw_action(
        self,
        raw_action: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        base_log_prob = normal.log_prob(raw_action).sum(dim=-1)
        tanh_action = torch.tanh(raw_action)
        squash_correction = torch.log(1.0 - tanh_action.pow(2) + self.eps).sum(dim=-1)
        scale_correction = torch.log(self.action_limit + self.eps).sum()
        return base_log_prob - squash_correction - scale_correction

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.model.get_dist_params(obs_t)
            value = self.model.get_value(obs_t)
            if deterministic:
                raw_action = mean
            else:
                std = torch.exp(log_std)
                raw_action = Normal(mean, std).sample()
            log_prob = self._log_prob_from_raw_action(raw_action, mean, log_std)
            action = torch.tanh(raw_action) * self.action_limit
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            raw_action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    def value(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.model.get_value(obs_t).item())

    def evaluate_raw_actions(self, obs: torch.Tensor, raw_actions: torch.Tensor):
        mean, log_std = self.model.get_dist_params(obs)
        log_probs = self._log_prob_from_raw_action(raw_actions, mean, log_std)
        values = self.model.get_value(obs)
        entropy_estimate = -log_probs.mean()
        return log_probs, entropy_estimate, values

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        buffer.compute_gae(self.config.gamma, self.config.gae_lambda)
        data = buffer.tensors(self.device)

        advantages = data["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        returns = data["returns"]
        old_values_np = data["values"].detach().cpu().numpy()
        returns_np = returns.detach().cpu().numpy()

        n_samples = len(buffer)
        batch_size = min(self.config.minibatch_size, n_samples)

        metric_sums = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        num_updates = 0
        kl_early_stop = False

        for _ in range(self.config.ppo_epochs):
            indices = torch.randperm(n_samples, device=self.device)
            early_stop = False
            for start in range(0, n_samples, batch_size):
                mb_idx = indices[start : start + batch_size]
                mb_obs = data["obs"][mb_idx]
                mb_raw_actions = data["raw_actions"][mb_idx]
                mb_old_log_probs = data["old_log_probs"][mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_log_probs, entropy, new_values = self.evaluate_raw_actions(mb_obs, mb_raw_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                unclipped = ratio * mb_advantages
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * mb_advantages
                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = 0.5 * (mb_returns - new_values).pow(2).mean()
                entropy_loss = -self.config.entropy_coef * entropy
                total_loss = actor_loss + self.config.value_coef * critic_loss + entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()

                metric_sums["actor_loss"] += float(actor_loss.item())
                metric_sums["critic_loss"] += float(critic_loss.item())
                metric_sums["entropy"] += float(entropy.item())
                metric_sums["total_loss"] += float(total_loss.item())
                metric_sums["approx_kl"] += float(approx_kl.item())
                metric_sums["clip_fraction"] += float(clip_fraction.item())
                num_updates += 1

                if self.config.target_kl is not None and approx_kl.item() > 1.5 * self.config.target_kl:
                    early_stop = True
                    kl_early_stop = True
                    break
            if early_stop:
                break

        metrics = {k: v / max(1, num_updates) for k, v in metric_sums.items()}
        metrics["explained_variance"] = explained_variance(old_values_np, returns_np)
        metrics["num_minibatch_updates"] = num_updates
        metrics["target_kl"] = float(self.config.target_kl) if self.config.target_kl is not None else float("nan")
        metrics["kl_early_stop"] = float(kl_early_stop)
        return metrics

    def save(self, path: Path, obs_rms: RunningMeanStd, extra: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "obs_rms": obs_rms.state_dict(),
                "config": asdict(self.config),
                "extra": extra,
            },
            path,
        )

    def load(self, path: Path, obs_rms: Optional[RunningMeanStd] = None) -> Dict[str, object]:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if obs_rms is not None and "obs_rms" in checkpoint:
            obs_rms.load_state_dict(checkpoint["obs_rms"])
        return checkpoint.get("extra", {})


def make_episode_logger(env_id: str, csv_dir: Path) -> CSVLogger:
    fields = [
        "episode",
        "global_step",
        "episode_return",
        "episode_length",
        "success",
        "fall",
        "timeout",
        "final_forward_progress",
        "final_base_forward_progress",
        "max_forward_progress",
        "final_front_x",
        "target_x",
        "completed_cycle_count",
        "fractional_cycle_count",
        "reference_cycle_steps",
        "episode_abs_work_j",
        "episode_signed_work_j",
        "episode_cot_abs",
        "cycle_mean_forward_distance_m",
        "cycle_mean_forward_speed_mps",
        "base_z",
        "base_y",
        "left_gripper_y",
        "right_gripper_y",
        "base_tilt_abs",
        "base_roll",
        "base_pitch",
        "base_yaw",
        "max_gripper_abs_y",
        "gripper_center_y",
        "gripper_y_span",
        "gripper_yaw_abs",
        "gripper_yaw_diff_abs",
        "max_gripper_tilt_abs",
        "left_gripper_tilt_abs",
        "right_gripper_tilt_abs",
        "mean_abs_joint_velocity",
        "max_abs_joint_velocity",
        "num_contacts",
        "total_normal_force",
        "mean_residual_abs",
        "max_residual_abs",
        "mean_joint_limit_clip_fraction",
        "progress_fraction",
        "pace_error",
        "reward_progress",
        "reward_time",
        "reward_pace",
        "reward_success",
        "reward_success_speed",
        "reward_action",
        "reward_smoothness",
        "reward_joint_velocity",
        "reward_base_lateral",
        "reward_base_tilt",
        "reward_base_yaw",
        "reward_gripper_lateral",
        "reward_gripper_center_y",
        "reward_gripper_y_span",
        "reward_gripper_tilt",
        "reward_gripper_yaw",
        "invalid_obs_count",
    ]
    return CSVLogger(csv_dir / "episode_log.csv", fields)


def make_update_logger(env_id: str, csv_dir: Path) -> CSVLogger:
    fields = [
        "update",
        "episode",
        "global_step",
        "actor_loss",
        "critic_loss",
        "entropy",
        "total_loss",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "num_minibatch_updates",
        "target_kl",
        "kl_early_stop",
    ]
    return CSVLogger(csv_dir / "update_log.csv", fields)


def make_eval_logger(env_id: str, csv_dir: Path) -> CSVLogger:
    fields = [
        "episode",
        "global_step",
        "eval_mean_return",
        "eval_success_rate",
        "eval_mean_length",
        "eval_mean_forward_progress",
        "eval_mean_front_x",
        "eval_mean_max_gripper_abs_y",
        "eval_mean_max_gripper_tilt_abs",
        "eval_mean_base_tilt_abs",
        "eval_mean_base_abs_y",
        "eval_mean_gripper_center_abs_y",
        "eval_mean_gripper_yaw_abs",
        "eval_mean_fractional_cycle_count",
        "eval_mean_completed_cycle_count",
        "eval_mean_episode_cot_abs",
        "eval_mean_episode_abs_work_j",
        "eval_mean_cycle_speed_mps",
        "best_model_updated",
    ]
    return CSVLogger(csv_dir / "eval_log.csv", fields)


def evaluate_policy(env_id: str, agent: PPOAgent, obs_rms: RunningMeanStd, config: PPOConfig) -> Dict[str, float]:
    env = LedgeClimbEnv(env_id=env_id, render_mode=None, max_episode_seconds=config.max_episode_seconds)
    returns = []
    lengths = []
    successes = []
    progresses = []
    front_positions = []
    gripper_abs_y_values = []
    gripper_tilt_values = []
    base_tilt_values = []
    base_y_values = []
    gripper_center_y_values = []
    gripper_yaw_values = []
    fractional_cycle_values = []
    completed_cycle_values = []
    episode_cot_values = []
    episode_abs_work_values = []
    cycle_speed_values = []

    try:
        for _ in range(config.eval_episodes):
            raw_obs, _ = env.reset()
            obs, _ = obs_rms.normalize(raw_obs, update=False)
            done = False
            ep_return = 0.0
            ep_len = 0
            final_info = {}

            while not done:
                action, _, _, _ = agent.select_action(obs, deterministic=True)
                raw_obs, reward, terminated, truncated, info = env.step(action)
                obs, _ = obs_rms.normalize(raw_obs, update=False)
                done = terminated or truncated
                ep_return += reward
                ep_len += 1
                final_info = info

            metrics = final_info.get("task_metrics", {})
            energy_cycle_metrics = final_info.get("energy_cycle_metrics", {})
            returns.append(ep_return)
            lengths.append(ep_len)
            successes.append(float(final_info.get("success", False)))
            progresses.append(float(metrics.get("forward_progress", 0.0)))
            front_positions.append(float(metrics.get("front_x", 0.0)))
            gripper_abs_y_values.append(float(metrics.get("max_gripper_abs_y", 0.0)))
            gripper_tilt_values.append(float(metrics.get("max_gripper_tilt_abs", 0.0)))
            base_tilt_values.append(float(metrics.get("base_tilt_abs", 0.0)))
            base_y_values.append(abs(float(metrics.get("base_y", 0.0))))
            gripper_center_y_values.append(abs(float(metrics.get("gripper_center_y", 0.0))))
            gripper_yaw_values.append(float(metrics.get("gripper_yaw_abs", 0.0)))
            fractional_cycle_values.append(float(energy_cycle_metrics.get("fractional_cycle_count", np.nan)))
            completed_cycle_values.append(float(energy_cycle_metrics.get("completed_cycle_count", np.nan)))
            episode_cot_values.append(float(energy_cycle_metrics.get("episode_cot_abs", np.nan)))
            episode_abs_work_values.append(float(energy_cycle_metrics.get("episode_abs_work_j", np.nan)))
            cycle_speed_values.append(float(energy_cycle_metrics.get("cycle_mean_forward_speed_mps", np.nan)))
    finally:
        env.close()

    return {
        "eval_mean_return": float(np.mean(returns)) if returns else 0.0,
        "eval_success_rate": float(np.mean(successes)) if successes else 0.0,
        "eval_mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "eval_mean_forward_progress": float(np.mean(progresses)) if progresses else 0.0,
        "eval_mean_front_x": float(np.mean(front_positions)) if front_positions else 0.0,
        "eval_mean_max_gripper_abs_y": float(np.mean(gripper_abs_y_values)) if gripper_abs_y_values else 0.0,
        "eval_mean_max_gripper_tilt_abs": float(np.mean(gripper_tilt_values)) if gripper_tilt_values else 0.0,
        "eval_mean_base_tilt_abs": float(np.mean(base_tilt_values)) if base_tilt_values else 0.0,
        "eval_mean_base_abs_y": float(np.mean(base_y_values)) if base_y_values else 0.0,
        "eval_mean_gripper_center_abs_y": float(np.mean(gripper_center_y_values)) if gripper_center_y_values else 0.0,
        "eval_mean_gripper_yaw_abs": float(np.mean(gripper_yaw_values)) if gripper_yaw_values else 0.0,
        "eval_mean_fractional_cycle_count": float(np.nanmean(fractional_cycle_values)) if fractional_cycle_values else 0.0,
        "eval_mean_completed_cycle_count": float(np.nanmean(completed_cycle_values)) if completed_cycle_values else 0.0,
        "eval_mean_episode_cot_abs": float(np.nanmean(episode_cot_values)) if episode_cot_values else 0.0,
        "eval_mean_episode_abs_work_j": float(np.nanmean(episode_abs_work_values)) if episode_abs_work_values else 0.0,
        "eval_mean_cycle_speed_mps": float(np.nanmean(cycle_speed_values)) if cycle_speed_values else 0.0,
    }

def is_better_model(new_eval: Dict[str, float], best_eval: Optional[Dict[str, float]]) -> bool:
    if best_eval is None:
        return True

    def score(metrics: Dict[str, float]) -> Tuple[float, ...]:
        success_rate = metrics["eval_success_rate"]
        if success_rate > 0.0:
            return (
                success_rate,
                -metrics["eval_mean_length"],
                metrics["eval_mean_return"],
                metrics["eval_mean_forward_progress"],
                -metrics.get("eval_mean_max_gripper_tilt_abs", 0.0),
                -metrics.get("eval_mean_gripper_center_abs_y", 0.0),
                -metrics.get("eval_mean_base_abs_y", 0.0),
                -metrics.get("eval_mean_gripper_yaw_abs", 0.0),
                -metrics.get("eval_mean_max_gripper_abs_y", 0.0),
            )
        return (
            success_rate,
            metrics["eval_mean_return"],
            metrics["eval_mean_forward_progress"],
            -metrics["eval_mean_length"],
            -metrics.get("eval_mean_max_gripper_tilt_abs", 0.0),
            -metrics.get("eval_mean_gripper_center_abs_y", 0.0),
            -metrics.get("eval_mean_base_abs_y", 0.0),
            -metrics.get("eval_mean_gripper_yaw_abs", 0.0),
            -metrics.get("eval_mean_max_gripper_abs_y", 0.0),
        )

    return score(new_eval) > score(best_eval)


def train_one_env(env_id: str, config: PPOConfig) -> None:
    print(f"\nTraining PPO on env_{env_id}")
    dirs = ensure_result_dirs(env_id)
    env = LedgeClimbEnv(env_id=env_id, render_mode=None, max_episode_seconds=config.max_episode_seconds)
    residual_limit_deg = float(np.rad2deg(np.max(np.abs(env.action_space.high))))
    print(
        f"env_{env_id} config | control_hz {1.0 / env.control_dt:.1f} | "
        f"max_episode_seconds {env.max_episode_seconds:.1f} | "
        f"max_episode_steps {env.max_episode_steps} | "
        f"residual_limit_deg {residual_limit_deg:.1f}"
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_limit = np.asarray(env.action_space.high, dtype=np.float32)

    agent = PPOAgent(obs_dim, act_dim, action_limit, config)
    obs_rms = RunningMeanStd(shape=(obs_dim,), clip=config.obs_clip)
    buffer = RolloutBuffer(obs_dim, act_dim)

    episode_logger = make_episode_logger(env_id, dirs["csv"])
    update_logger = make_update_logger(env_id, dirs["csv"])
    eval_logger = make_eval_logger(env_id, dirs["csv"])

    raw_obs, _ = env.reset()
    obs, invalid_count = obs_rms.normalize(raw_obs, update=True)

    global_step = 0
    episode = 0
    update_count = 0
    next_eval_step = config.eval_interval
    best_eval: Optional[Dict[str, float]] = None
    best_model_path = dirs["best_model"] / "best.pt"
    final_model_path = dirs["best_model"] / "final.pt"

    ep_return = 0.0
    ep_length = 0
    ep_invalid_obs = invalid_count
    ep_residual_abs_sum = 0.0
    ep_residual_abs_max = 0.0
    ep_joint_clip_sum = 0.0
    ep_reward_terms: Dict[str, float] = {}
    final_info = {}

    try:
        while global_step < config.total_timesteps:
            buffer.clear()

            for _ in range(config.rollout_length):
                action, raw_action, log_prob, value = agent.select_action(obs, deterministic=False)
                next_raw_obs, reward, terminated, truncated, info = env.step(action)
                next_obs, invalid_count = obs_rms.normalize(next_raw_obs, update=True)
                done = bool(terminated or truncated)
                next_value = 0.0 if terminated else agent.value(next_obs)

                buffer.add(
                    obs=obs,
                    raw_action=raw_action,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    terminated=terminated,
                    done=done,
                    next_value=next_value,
                )

                residual_abs = np.abs(np.asarray(info.get("residual_action", action), dtype=np.float32))
                ep_residual_abs_sum += float(np.mean(residual_abs))
                ep_residual_abs_max = max(ep_residual_abs_max, float(np.max(residual_abs)))
                ep_joint_clip_sum += float(info.get("joint_limit_clip_fraction", 0.0))
                for term_name, term_value in info.get("reward_terms", {}).items():
                    if isinstance(term_value, (int, float, np.integer, np.floating)) and np.isfinite(term_value):
                        ep_reward_terms[term_name] = ep_reward_terms.get(term_name, 0.0) + float(term_value)
                ep_invalid_obs += invalid_count
                ep_return += float(reward)
                ep_length += 1
                global_step += 1
                final_info = info
                obs = next_obs

                if done:
                    episode += 1
                    metrics = final_info.get("task_metrics", {})
                    energy_cycle_metrics = final_info.get("energy_cycle_metrics", {})
                    episode_row = {
                        "episode": episode,
                        "global_step": global_step,
                        "episode_return": ep_return,
                        "episode_length": ep_length,
                        "success": int(final_info.get("success", False)),
                        "fall": int(final_info.get("fall", False)),
                        "timeout": int(final_info.get("timeout", False)),
                        "final_forward_progress": metrics.get("forward_progress", 0.0),
                        "final_base_forward_progress": metrics.get("base_forward_progress", 0.0),
                        "max_forward_progress": metrics.get("max_forward_progress", 0.0),
                        "final_front_x": metrics.get("front_x", 0.0),
                        "target_x": metrics.get("target_x", 0.0),
                        "completed_cycle_count": energy_cycle_metrics.get("completed_cycle_count", 0.0),
                        "fractional_cycle_count": energy_cycle_metrics.get("fractional_cycle_count", 0.0),
                        "reference_cycle_steps": energy_cycle_metrics.get("reference_cycle_steps", 0.0),
                        "episode_abs_work_j": energy_cycle_metrics.get("episode_abs_work_j", 0.0),
                        "episode_signed_work_j": energy_cycle_metrics.get("episode_signed_work_j", 0.0),
                        "episode_cot_abs": energy_cycle_metrics.get("episode_cot_abs", 0.0),
                        "cycle_mean_forward_distance_m": energy_cycle_metrics.get("cycle_mean_forward_distance_m", 0.0),
                        "cycle_mean_forward_speed_mps": energy_cycle_metrics.get("cycle_mean_forward_speed_mps", 0.0),
                        "base_z": metrics.get("base_z", 0.0),
                        "base_y": metrics.get("base_y", 0.0),
                        "left_gripper_y": metrics.get("left_gripper_y", 0.0),
                        "right_gripper_y": metrics.get("right_gripper_y", 0.0),
                        "base_tilt_abs": metrics.get("base_tilt_abs", 0.0),
                        "base_roll": metrics.get("base_roll", 0.0),
                        "base_pitch": metrics.get("base_pitch", 0.0),
                        "base_yaw": metrics.get("base_yaw", 0.0),
                        "max_gripper_abs_y": metrics.get("max_gripper_abs_y", 0.0),
                        "gripper_center_y": metrics.get("gripper_center_y", 0.0),
                        "gripper_y_span": metrics.get("gripper_y_span", 0.0),
                        "gripper_yaw_abs": metrics.get("gripper_yaw_abs", 0.0),
                        "gripper_yaw_diff_abs": metrics.get("gripper_yaw_diff_abs", 0.0),
                        "max_gripper_tilt_abs": metrics.get("max_gripper_tilt_abs", 0.0),
                        "left_gripper_tilt_abs": metrics.get("left_gripper_tilt_abs", 0.0),
                        "right_gripper_tilt_abs": metrics.get("right_gripper_tilt_abs", 0.0),
                        "mean_abs_joint_velocity": metrics.get("mean_abs_joint_velocity", 0.0),
                        "max_abs_joint_velocity": metrics.get("max_abs_joint_velocity", 0.0),
                        "num_contacts": metrics.get("num_contacts", 0.0),
                        "total_normal_force": metrics.get("total_normal_force", 0.0),
                        "mean_residual_abs": ep_residual_abs_sum / max(1, ep_length),
                        "max_residual_abs": ep_residual_abs_max,
                        "mean_joint_limit_clip_fraction": ep_joint_clip_sum / max(1, ep_length),
                        "progress_fraction": float(final_info.get("reward_terms", {}).get("progress_fraction", 0.0)),
                        "pace_error": float(final_info.get("reward_terms", {}).get("pace_error", 0.0)),
                        "reward_progress": ep_reward_terms.get("progress_reward", 0.0),
                        "reward_time": ep_reward_terms.get("time_penalty", 0.0),
                        "reward_pace": ep_reward_terms.get("pace_cost", 0.0),
                        "reward_success": ep_reward_terms.get("success_bonus", 0.0),
                        "reward_success_speed": ep_reward_terms.get("success_speed_bonus", 0.0),
                        "reward_action": ep_reward_terms.get("action_cost", 0.0),
                        "reward_smoothness": ep_reward_terms.get("smoothness_cost", 0.0),
                        "reward_joint_velocity": ep_reward_terms.get("joint_velocity_cost", 0.0),
                        "reward_base_lateral": ep_reward_terms.get("base_lateral_cost", 0.0),
                        "reward_base_tilt": ep_reward_terms.get("base_tilt_cost", 0.0),
                        "reward_base_yaw": ep_reward_terms.get("base_yaw_cost", 0.0),
                        "reward_gripper_lateral": ep_reward_terms.get("gripper_lateral_cost", 0.0),
                        "reward_gripper_center_y": ep_reward_terms.get("gripper_center_y_cost", 0.0),
                        "reward_gripper_y_span": ep_reward_terms.get("gripper_y_span_cost", 0.0),
                        "reward_gripper_tilt": ep_reward_terms.get("gripper_tilt_cost", 0.0),
                        "reward_gripper_yaw": ep_reward_terms.get("gripper_yaw_cost", 0.0),
                        "invalid_obs_count": ep_invalid_obs,
                    }
                    episode_logger.write(episode_row)

                    if PRINT_EVERY_EPISODES > 0 and episode % PRINT_EVERY_EPISODES == 0:
                        if episode_row["success"]:
                            outcome = "success"
                        elif episode_row["fall"]:
                            outcome = "fall"
                        elif episode_row["timeout"]:
                            outcome = "timeout"
                        else:
                            outcome = "done"
                        print(
                            f"env_{env_id} | ep {episode:>5} | "
                            f"step {global_step:>8}/{config.total_timesteps:<8} | "
                            f"return {ep_return:>9.2f} | len {ep_length:>5} | "
                            f"{outcome} | progress {metrics.get('forward_progress', 0.0):.3f} m | "
                            f"front_x {metrics.get('front_x', 0.0):.3f}/{metrics.get('target_x', 0.0):.3f}"
                        )

                    raw_obs, _ = env.reset()
                    obs, invalid_count = obs_rms.normalize(raw_obs, update=True)
                    ep_return = 0.0
                    ep_length = 0
                    ep_invalid_obs = invalid_count
                    ep_residual_abs_sum = 0.0
                    ep_residual_abs_max = 0.0
                    ep_joint_clip_sum = 0.0
                    ep_reward_terms = {}
                    final_info = {}

                if global_step >= config.total_timesteps:
                    break

            if len(buffer) == 0:
                continue

            update_count += 1
            update_metrics = agent.update(buffer)
            update_logger.write(
                {
                    "update": update_count,
                    "episode": episode,
                    "global_step": global_step,
                    **update_metrics,
                }
            )

            if PRINT_EVERY_UPDATES > 0 and update_count % PRINT_EVERY_UPDATES == 0:
                print(
                    f"env_{env_id} | update {update_count:>4} | "
                    f"episode {episode:>5} | step {global_step:>8}/{config.total_timesteps:<8} | "
                    f"actor {update_metrics['actor_loss']:.6f} | "
                    f"critic {update_metrics['critic_loss']:.4f} | "
                    f"entropy {update_metrics['entropy']:.4f}"
                )

            if global_step >= next_eval_step or global_step >= config.total_timesteps:
                eval_metrics = evaluate_policy(env_id, agent, obs_rms, config)
                best_updated = is_better_model(eval_metrics, best_eval)
                if best_updated:
                    best_eval = dict(eval_metrics)
                    agent.save(
                        best_model_path,
                        obs_rms,
                        extra={
                            "env_id": env_id,
                            "global_step": global_step,
                            "best_eval": best_eval,
                        },
                    )
                eval_logger.write(
                    {
                        "episode": episode,
                        "global_step": global_step,
                        **eval_metrics,
                        "best_model_updated": int(best_updated),
                    }
                )
                print(
                    f"env_{env_id} | step {global_step:>8} | "
                    f"success {eval_metrics['eval_success_rate']:.2f} | "
                    f"progress {eval_metrics['eval_mean_forward_progress']:.3f} m | "
                    f"front_x {eval_metrics['eval_mean_front_x']:.3f} m | "
                    f"return {eval_metrics['eval_mean_return']:.2f} | "
                    f"length {eval_metrics['eval_mean_length']:.1f}"
                )
                while next_eval_step <= global_step:
                    next_eval_step += config.eval_interval

        agent.save(
            final_model_path,
            obs_rms,
            extra={"env_id": env_id, "global_step": global_step, "final": True},
        )

    finally:
        env.close()

    plot_training_curves(env_id, dirs["csv"], dirs["plot"])

    if SAVE_FINAL_VIDEO and best_model_path.exists():
        try:
            record_best_model_video(env_id, best_model_path, config, dirs)
        except Exception as exc:
            print(f"Video recording skipped for env_{env_id}: {exc}")

    print(f"Finished env_{env_id}. Best model: {best_model_path}")


def write_best_rollout_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_rollout_joint_row(rows: List[Dict[str, float]], env: LedgeClimbEnv, info: Dict[str, object], episode: int, step: int) -> None:
    joint_names = list(info.get("joint_names", []))
    joint_angles = np.asarray(info.get("joint_angles", []), dtype=np.float64)
    joint_velocities = np.asarray(info.get("joint_velocities", []), dtype=np.float64)
    reference_action = np.asarray(info.get("reference_action", []), dtype=np.float64)
    residual_action = np.asarray(info.get("residual_action", []), dtype=np.float64)
    applied_action = np.asarray(info.get("applied_action", []), dtype=np.float64)
    metrics = info.get("task_metrics", {})
    energy_cycle_metrics = info.get("energy_cycle_metrics", {})

    row: Dict[str, float] = {
        "episode": float(episode),
        "step": float(step),
        "time_s": float(step * env.control_dt),
        "success": float(bool(info.get("success", False))),
        "fall": float(bool(info.get("fall", False))),
        "timeout": float(bool(info.get("timeout", False))),
    }
    if isinstance(metrics, dict):
        for key in [
            "front_x", "target_x", "forward_progress", "base_z",
            "left_gripper_x", "left_gripper_y", "left_gripper_z",
            "left_gripper_yaw", "left_gripper_pitch", "left_gripper_roll", "left_gripper_tilt_abs",
            "right_gripper_x", "right_gripper_y", "right_gripper_z",
            "right_gripper_yaw", "right_gripper_pitch", "right_gripper_roll", "right_gripper_tilt_abs",
            "max_gripper_abs_y", "max_gripper_tilt_abs",
        ]:
            row[key] = float(metrics.get(key, np.nan))
    if isinstance(energy_cycle_metrics, dict):
        for key in [
            "completed_cycle_count", "fractional_cycle_count", "reference_cycle_steps",
            "episode_abs_work_j", "episode_signed_work_j", "episode_cot_abs",
            "cycle_mean_forward_distance_m", "cycle_mean_forward_speed_mps",
        ]:
            row[key] = float(energy_cycle_metrics.get(key, np.nan))

    for i, name in enumerate(joint_names):
        if i < len(reference_action):
            row[f"{name}_demo_ref_deg"] = float(np.rad2deg(reference_action[i]))
        if i < len(residual_action):
            row[f"{name}_residual_deg"] = float(np.rad2deg(residual_action[i]))
        if i < len(applied_action):
            row[f"{name}_ref_deg"] = float(np.rad2deg(applied_action[i]))
        if i < len(joint_angles):
            row[f"{name}_feedback_deg"] = float(np.rad2deg(joint_angles[i]))
        if i < len(joint_velocities):
            row[f"{name}_velocity_deg_s"] = float(np.rad2deg(joint_velocities[i]))
    rows.append(row)


def plot_joint_angle_group(
    rollout_csv: Path,
    plot_dir: Path,
    env_id: str,
    joint_names: List[str],
    title: str,
    filename: str,
    layout: Tuple[int, int],
) -> None:
    data = read_csv_columns(rollout_csv)
    if not data:
        return
    x = data.get("time_s", np.array([]))
    if len(x) == 0:
        return

    rows, cols = layout
    fig_width = max(10.0, 5.0 * cols)
    fig_height = 3.2 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    for ax, joint_name in zip(axes.ravel(), joint_names):
        ref_key = f"{joint_name}_ref_deg"
        fb_key = f"{joint_name}_feedback_deg"
        if ref_key in data:
            ax.plot(x, data[ref_key], linewidth=1.6, label="reference")
        if fb_key in data:
            ax.plot(x, data[fb_key], linewidth=1.1, alpha=0.75, label="feedback")
        ax.set_title(joint_name)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("angle [deg]")
        ax.grid(True, alpha=0.3)
        ax.legend()
    for ax in axes.ravel()[len(joint_names):]:
        ax.axis("off")
    fig.suptitle(f"env_{env_id} {title}")
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / filename, dpi=200)
    plt.close(fig)


def plot_best_rollout_joint_angles(rollout_csv: Path, plot_dir: Path, env_id: str) -> None:
    plot_joint_angle_group(
        rollout_csv,
        plot_dir,
        env_id,
        ["left_upper_claw", "left_lower_claw", "right_upper_claw", "right_lower_claw"],
        "claw joint angles",
        "best_model_claw_joint_angles.png",
        (2, 2),
    )
    plot_joint_angle_group(
        rollout_csv,
        plot_dir,
        env_id,
        ["left_wrist", "right_wrist"],
        "wrist joint angles",
        "best_model_wrist_joint_angles.png",
        (2, 1),
    )
    plot_joint_angle_group(
        rollout_csv,
        plot_dir,
        env_id,
        ["left_elbow", "shoulder", "right_elbow"],
        "body joint angles",
        "best_model_body_joint_angles.png",
        (3, 1),
    )


def record_best_model_video(
    env_id: str,
    model_path: Path,
    config: PPOConfig,
    dirs: Dict[str, Path],
    model_env_id: Optional[str] = None,
) -> Dict[str, float]:
    video_dir = dirs["video"]
    csv_dir = dirs["csv"]
    plot_dir = dirs["plot"]
    model_label = f"model_env{model_env_id}" if model_env_id is not None else "best model"
    print(f"Recording {model_label} on test_env{env_id}...")
    video_path = video_dir / "best_model_run.mp4"
    rollout_csv = csv_dir / "best_model_rollout_joint_log.csv"
    cycle_csv = csv_dir / "best_model_rollout_cycle_log.csv"
    env = LedgeClimbEnv(
        env_id=env_id,
        render_mode="human",
        record_video=True,
        video_path=video_path,
        max_episode_seconds=config.max_episode_seconds,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_limit = np.asarray(env.action_space.high, dtype=np.float32)
    agent = PPOAgent(obs_dim, act_dim, action_limit, config)
    obs_rms = RunningMeanStd(shape=(obs_dim,), clip=config.obs_clip)
    agent.load(model_path, obs_rms)

    rollout_rows: List[Dict[str, float]] = []
    summary: Dict[str, float] = {
        "success": 0.0,
        "fall": 0.0,
        "timeout": 0.0,
        "episode_length": 0.0,
        "episode_time_s": 0.0,
        "final_forward_progress": 0.0,
        "final_front_x": 0.0,
        "target_x": 0.0,
        "fractional_cycle_count": 0.0,
        "completed_cycle_count": 0.0,
        "episode_abs_work_j": 0.0,
        "episode_cot_abs": 0.0,
        "cycle_mean_forward_speed_mps": 0.0,
    }
    try:
        for episode_idx in range(1, VIDEO_MAX_EPISODES + 1):
            raw_obs, _ = env.reset()
            obs, _ = obs_rms.normalize(raw_obs, update=False)
            done = False
            step = 0
            final_info: Dict[str, object] = {}
            while not done:
                action, _, _, _ = agent.select_action(obs, deterministic=True)
                raw_obs, _, terminated, truncated, info = env.step(action)
                step += 1
                append_rollout_joint_row(rollout_rows, env, info, episode_idx, step)
                obs, _ = obs_rms.normalize(raw_obs, update=False)
                done = terminated or truncated
                final_info = info

            metrics = final_info.get("task_metrics", {}) if isinstance(final_info, dict) else {}
            energy_cycle_metrics = final_info.get("energy_cycle_metrics", {}) if isinstance(final_info, dict) else {}
            summary = {
                "success": float(bool(final_info.get("success", False))) if isinstance(final_info, dict) else 0.0,
                "fall": float(bool(final_info.get("fall", False))) if isinstance(final_info, dict) else 0.0,
                "timeout": float(bool(final_info.get("timeout", False))) if isinstance(final_info, dict) else 0.0,
                "episode_length": float(step),
                "episode_time_s": float(step * env.control_dt),
                "final_forward_progress": float(metrics.get("forward_progress", 0.0)) if isinstance(metrics, dict) else 0.0,
                "final_front_x": float(metrics.get("front_x", 0.0)) if isinstance(metrics, dict) else 0.0,
                "target_x": float(metrics.get("target_x", 0.0)) if isinstance(metrics, dict) else 0.0,
                "fractional_cycle_count": float(energy_cycle_metrics.get("fractional_cycle_count", 0.0)) if isinstance(energy_cycle_metrics, dict) else 0.0,
                "completed_cycle_count": float(energy_cycle_metrics.get("completed_cycle_count", 0.0)) if isinstance(energy_cycle_metrics, dict) else 0.0,
                "episode_abs_work_j": float(energy_cycle_metrics.get("episode_abs_work_j", 0.0)) if isinstance(energy_cycle_metrics, dict) else 0.0,
                "episode_cot_abs": float(energy_cycle_metrics.get("episode_cot_abs", 0.0)) if isinstance(energy_cycle_metrics, dict) else 0.0,
                "cycle_mean_forward_speed_mps": float(energy_cycle_metrics.get("cycle_mean_forward_speed_mps", 0.0)) if isinstance(energy_cycle_metrics, dict) else 0.0,
            }
    finally:
        env.close()

    write_best_rollout_csv(rollout_rows, rollout_csv)
    write_best_rollout_csv(env.get_cycle_log(), cycle_csv)
    plot_best_rollout_joint_angles(rollout_csv, plot_dir, env_id)

    if video_path.exists() and video_path.stat().st_size > 0:
        print(f"Saved video: {video_path}")
    else:
        print(f"Video file was not created: {video_path}")
    if rollout_csv.exists():
        print(f"Saved best rollout joint log: {rollout_csv}")
    if cycle_csv.exists():
        print(f"Saved best rollout cycle log: {cycle_csv}")
    print(
        f"Result | success {summary['success']:.0f} | fall {summary['fall']:.0f} | "
        f"timeout {summary['timeout']:.0f} | time {summary['episode_time_s']:.2f} s | "
        f"cycles {summary['fractional_cycle_count']:.2f} | COT {summary['episode_cot_abs']:.3f} | "
        f"front_x {summary['final_front_x']:.3f} / target {summary['target_x']:.3f}"
    )
    return summary


def write_cross_eval_summary(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "model_env",
        "test_env",
        "model_path",
        "success",
        "fall",
        "timeout",
        "episode_length",
        "episode_time_s",
        "final_forward_progress",
        "final_front_x",
        "target_x",
        "fractional_cycle_count",
        "completed_cycle_count",
        "episode_abs_work_j",
        "episode_cot_abs",
        "cycle_mean_forward_speed_mps",
        "error",
    ]
    all_keys = {key for row in rows for key in row.keys()}
    fieldnames = [key for key in preferred if key in all_keys] + sorted(all_keys - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def record_existing_best_model_videos(config: PPOConfig) -> None:
    model_env_id = str(RECORD_MODEL_ENV_ID)
    model_dirs = get_result_dirs(model_env_id)
    model_path = model_dirs["best_model"] / RECORD_MODEL_FILENAME
    if not model_path.exists():
        print(f"No model found: {model_path}")
        return

    test_envs = [str(env_id) for env_id in RECORD_TEST_ENVS]
    print(f"ONLY_RECORD_VIDEO mode | model_env{model_env_id} | test_envs {test_envs}")
    summary_rows: List[Dict[str, object]] = []
    for test_env_id in test_envs:
        dirs = ensure_cross_eval_dirs(model_env_id, test_env_id)
        try:
            summary = record_best_model_video(test_env_id, model_path, config, dirs, model_env_id=model_env_id)
            summary_rows.append({
                "model_env": model_env_id,
                "test_env": test_env_id,
                "model_path": str(model_path),
                **summary,
            })
        except Exception as exc:
            print(f"Video recording skipped for model_env{model_env_id} on test_env{test_env_id}: {exc}")
            summary_rows.append({
                "model_env": model_env_id,
                "test_env": test_env_id,
                "model_path": str(model_path),
                "error": str(exc),
            })

    summary_path = RESULTS_ROOT / "cross_eval" / f"model_env{model_env_id}" / "cross_eval_summary.csv"
    write_cross_eval_summary(summary_rows, summary_path)
    print(f"Saved cross-evaluation summary: {summary_path}")


def read_csv_columns(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    columns: Dict[str, List[float]] = {name: [] for name in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            try:
                columns[key].append(float(value))
            except (TypeError, ValueError):
                columns[key].append(np.nan)
    return {key: np.asarray(value, dtype=np.float64) for key, value in columns.items()}


def save_line_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    rolling_window: Optional[int] = None,
    raw_label: str = "raw",
    average_label: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0 or len(y) == 0:
        return

    plt.figure(figsize=(8, 5))
    raw_line = plt.plot(x, y, linewidth=0.8, alpha=0.25, label=raw_label)[0]
    if rolling_window is not None:
        y_ma = moving_average(y, rolling_window)
        label = average_label if average_label is not None else f"moving average-{rolling_window}"
        plt.plot(x, y_ma, linewidth=2.0, color=raw_line.get_color(), label=label)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def save_losses_plot(update_data: Dict[str, np.ndarray], env_id: str, plot_dir: Path) -> None:
    if not update_data:
        return
    x = update_data.get("episode", np.array([]))
    xlabel = "episode"
    if len(x) == 0:
        x = update_data.get("global_step", np.array([]))
        xlabel = "global step"
    if len(x) == 0:
        return

    keys = ["actor_loss", "critic_loss", "total_loss"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        if key not in update_data:
            ax.axis("off")
            continue
        y = np.asarray(update_data[key], dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            ax.axis("off")
            continue
        xv = np.asarray(x, dtype=np.float64)[valid]
        yv = y[valid]
        raw_line = ax.plot(xv, yv, linewidth=0.8, alpha=0.25, label="raw")[0]
        ax.plot(xv, moving_average(yv, ROLLING_WINDOW), linewidth=2.0, color=raw_line.get_color(), label=f"moving average-{ROLLING_WINDOW}")
        ax.set_title(key)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"env_{env_id} PPO losses")
    fig.tight_layout()
    fig.savefig(plot_dir / "losses.png", dpi=200)
    plt.close(fig)


def save_optimizer_metrics_plot(update_data: Dict[str, np.ndarray], env_id: str, plot_dir: Path) -> None:
    if not update_data:
        return
    x = update_data.get("episode", np.array([]))
    xlabel = "episode"
    if len(x) == 0:
        x = update_data.get("global_step", np.array([]))
        xlabel = "global step"
    if len(x) == 0:
        return

    metric_items = [
        ("entropy", "policy entropy"),
        ("approx_kl", "approximate KL"),
        ("clip_fraction", "clip fraction"),
        ("explained_variance", "explained variance"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), squeeze=False)
    for ax, (key, title) in zip(axes.ravel(), metric_items):
        if key not in update_data:
            ax.axis("off")
            continue
        y = np.asarray(update_data[key], dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            ax.axis("off")
            continue
        xv = np.asarray(x, dtype=np.float64)[valid]
        yv = y[valid]
        raw_line = ax.plot(xv, yv, linewidth=0.8, alpha=0.25, label="raw")[0]
        ax.plot(xv, moving_average(yv, ROLLING_WINDOW), linewidth=2.0, color=raw_line.get_color(), label=f"moving average-{ROLLING_WINDOW}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"env_{env_id} PPO optimizer diagnostics")
    fig.tight_layout()
    fig.savefig(plot_dir / "optimizer_diagnostics.png", dpi=200)
    plt.close(fig)


def plot_training_curves(env_id: str, csv_dir: Path, plot_dir: Path) -> None:
    episode_data = read_csv_columns(csv_dir / "episode_log.csv")
    update_data = read_csv_columns(csv_dir / "update_log.csv")
    eval_data = read_csv_columns(csv_dir / "eval_log.csv")

    if episode_data:
        x_ep = episode_data.get("episode", np.array([]))
        save_line_plot(
            x_ep,
            episode_data.get("episode_return", np.array([])),
            f"env_{env_id} episode return",
            "episode",
            "return",
            plot_dir / "episode_return.png",
            ROLLING_WINDOW,
        )
        save_line_plot(
            x_ep,
            episode_data.get("episode_length", np.array([])),
            f"env_{env_id} episode length",
            "episode",
            "steps",
            plot_dir / "episode_length.png",
            ROLLING_WINDOW,
        )
        save_line_plot(
            x_ep,
            episode_data.get("success", np.array([])),
            f"env_{env_id} success rate",
            "episode",
            "success / success rate",
            plot_dir / "success_rate.png",
            ROLLING_WINDOW,
            raw_label="raw success",
            average_label="moving average",
            ylim=(-0.05, 1.05),
        )
        save_line_plot(
            x_ep,
            episode_data.get("final_forward_progress", np.array([])),
            f"env_{env_id} forward progress",
            "episode",
            "forward progress [m]",
            plot_dir / "forward_progress.png",
            ROLLING_WINDOW,
        )
        if "fractional_cycle_count" in episode_data:
            save_line_plot(
                x_ep,
                episode_data.get("fractional_cycle_count", np.array([])),
                f"env_{env_id} fractional cycle count",
                "episode",
                "cycles",
                plot_dir / "fractional_cycle_count.png",
                ROLLING_WINDOW,
            )
        if "cycle_mean_forward_speed_mps" in episode_data:
            save_line_plot(
                x_ep,
                episode_data.get("cycle_mean_forward_speed_mps", np.array([])),
                f"env_{env_id} cycle mean forward speed",
                "episode",
                "speed [m/s]",
                plot_dir / "cycle_mean_forward_speed.png",
                ROLLING_WINDOW,
            )
        if "episode_cot_abs" in episode_data:
            save_line_plot(
                x_ep,
                episode_data.get("episode_cot_abs", np.array([])),
                f"env_{env_id} absolute cost of transport",
                "episode",
                "COT abs",
                plot_dir / "episode_cot_abs.png",
                ROLLING_WINDOW,
            )
        if "mean_residual_abs" in episode_data:
            save_line_plot(
                x_ep,
                np.rad2deg(episode_data.get("mean_residual_abs", np.array([]))),
                f"env_{env_id} residual action magnitude",
                "episode",
                "mean |residual action| [deg]",
                plot_dir / "residual_action_magnitude.png",
                ROLLING_WINDOW,
            )
        save_line_plot(
            x_ep,
            episode_data.get("max_gripper_abs_y", np.array([])),
            f"env_{env_id} gripper lateral offset",
            "episode",
            "max |gripper y| [m]",
            plot_dir / "gripper_lateral_offset.png",
            ROLLING_WINDOW,
        )
        if "gripper_center_y" in episode_data:
            save_line_plot(
                x_ep,
                episode_data.get("gripper_center_y", np.array([])),
                f"env_{env_id} gripper center lateral drift",
                "episode",
                "gripper center y [m]",
                plot_dir / "gripper_center_y.png",
                ROLLING_WINDOW,
            )
        if "max_gripper_tilt_abs" in episode_data:
            save_line_plot(
                x_ep,
                np.rad2deg(episode_data.get("max_gripper_tilt_abs", np.array([]))),
                f"env_{env_id} gripper tilt",
                "episode",
                "max gripper tilt [deg]",
                plot_dir / "gripper_tilt.png",
                ROLLING_WINDOW,
            )
        if "base_tilt_abs" in episode_data:
            save_line_plot(
                x_ep,
                np.rad2deg(episode_data.get("base_tilt_abs", np.array([]))),
                f"env_{env_id} base tilt",
                "episode",
                "base tilt [deg]",
                plot_dir / "base_tilt.png",
                ROLLING_WINDOW,
            )
        if "base_y" in episode_data:
            save_line_plot(
                x_ep,
                episode_data.get("base_y", np.array([])),
                f"env_{env_id} base lateral drift",
                "episode",
                "base y [m]",
                plot_dir / "base_y.png",
                ROLLING_WINDOW,
            )
        if "base_yaw" in episode_data:
            save_line_plot(
                x_ep,
                np.rad2deg(episode_data.get("base_yaw", np.array([]))),
                f"env_{env_id} base yaw",
                "episode",
                "base yaw [deg]",
                plot_dir / "base_yaw.png",
                ROLLING_WINDOW,
            )
        if "gripper_y_span" in episode_data:
            save_line_plot(
                x_ep,
                episode_data.get("gripper_y_span", np.array([])),
                f"env_{env_id} gripper y span",
                "episode",
                "gripper y span [m]",
                plot_dir / "gripper_y_span.png",
                ROLLING_WINDOW,
            )
        if "gripper_yaw_abs" in episode_data:
            save_line_plot(
                x_ep,
                np.rad2deg(episode_data.get("gripper_yaw_abs", np.array([]))),
                f"env_{env_id} gripper yaw",
                "episode",
                "max gripper yaw [deg]",
                plot_dir / "gripper_yaw.png",
                ROLLING_WINDOW,
            )

    save_losses_plot(update_data, env_id, plot_dir)
    save_optimizer_metrics_plot(update_data, env_id, plot_dir)


def main() -> None:
    config = PPOConfig()
    set_seed(config.seed)

    print(f"script path: {Path(__file__).resolve()}")
    print(f"results root: {RESULTS_ROOT}")
    print("PPO config:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    print(f"Training environments: {TRAIN_ENVS}")

    if ONLY_RECORD_VIDEO:
        record_existing_best_model_videos(config)
        return

    for env_id in TRAIN_ENVS:
        train_one_env(env_id, config)


if __name__ == "__main__":
    main()