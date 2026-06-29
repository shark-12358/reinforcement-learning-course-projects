from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - only used when video recording is enabled
    imageio = None



RUN_DIR = Path(__file__).resolve().parent
CODE_ROOT = RUN_DIR.parent
PROJECT_ROOT = CODE_ROOT.parent
URDF_ROOT = CODE_ROOT / "URDFs"
ROBOT_URDF_PATH = URDF_ROOT / "robot" / "robot.urdf"
ENV_URDF_DIR = URDF_ROOT / "env"

ENV_CONFIGS: Dict[str, Dict[str, object]] = {
    "1": {
        "name": "env_1",
        "env_urdf": ENV_URDF_DIR / "env_1.urdf",
        "robot_base_pos": [0.05, -0.06, 1.0225],
        "env_base_pos": [0.0, 0.0, 1.0],
        "target_x": None,          # None: use environment AABB max x
        "success_margin": 0.05,
    },
    "2": {
        "name": "env_2",
        "env_urdf": ENV_URDF_DIR / "env_2.urdf",
        "robot_base_pos": [0.05, -0.06, 1.0225],
        "env_base_pos": [0.0, 0.0, 1.0],
        "target_x": None,
        "success_margin": 0.05,
    },
    "3": {
        "name": "env_3",
        "env_urdf": ENV_URDF_DIR / "env_3.urdf",
        "robot_base_pos": [0.05, -0.06, 1.0225],
        "env_base_pos": [0.0, 0.0, 1.0],
        "target_x": None,
        "success_margin": 0.05,
    },
    "4": {
        "name": "env_4",
        "env_urdf": ENV_URDF_DIR / "env_4.urdf",
        "robot_base_pos": [0.05, -0.06, 1.0225],
        "env_base_pos": [0.0, 0.0, 1.0],
        "target_x": None,
        "success_margin": 0.05,
    },
}

PHYSICS_DT = 1.0 / 240.0
FRAME_SKIP = 8                         # PPO control frequency = 240 / 8 = 30 Hz
CONTROL_DT = PHYSICS_DT * FRAME_SKIP
MAX_EPISODE_SECONDS = 120.0
FALL_MARGIN = 0.25
RESIDUAL_ACTION_LIMIT_DEG = 5.0
RESIDUAL_ACTION_LIMITS_DEG = {
    "left_upper_claw": 5.0,
    "left_lower_claw": 5.0,
    "right_upper_claw": 5.0,
    "right_lower_claw": 5.0,
    "left_wrist": 10.0,
    "right_wrist": 10.0,
    "left_elbow": 15.0,
    "shoulder": 15.0,
    "right_elbow": 15.0,
}
REFERENCE_DURATION_SCALE = 0.50
# Pace and success-speed targets are split for logging/analysis,
# but both are set to 15 s here to recover the original reward behavior.
PACE_TARGET_SECONDS = 15.0
SUCCESS_SPEED_TARGET_SECONDS = 15.0
# Backward-compatible alias for older scripts that may import SPEED_TARGET_SECONDS.
SPEED_TARGET_SECONDS = PACE_TARGET_SECONDS

REWARD_WEIGHTS = {
    "progress": 10.0,
    "success": 150.0,
    "success_speed": 450.0,
    "fall": -150.0,
    "timeout": -25.0,
    "time": -0.02,
    "pace": -0.08,
    "action": -0.01,
    "smoothness": -0.03,
    "joint_velocity": -0.012,
    "base_lateral": -0.025,
    "base_tilt": -0.020,
    "base_yaw": -0.012,
    "gripper_lateral": -0.028,
    "gripper_center_y": -0.024,
    "gripper_y_span": -0.010,
    "gripper_tilt": -0.020,
    "gripper_yaw": -0.010,
}

RESIDUAL_FILTER_ALPHA = 0.35
RESIDUAL_RATE_LIMITS_DEG_PER_STEP = {
    "left_upper_claw": 3.0,
    "left_lower_claw": 3.0,
    "right_upper_claw": 3.0,
    "right_lower_claw": 3.0,
    "left_wrist": 3.0,
    "right_wrist": 3.0,
    "left_elbow": 4.5,
    "shoulder": 4.5,
    "right_elbow": 4.5,
}
JOINT_VELOCITY_NORMALIZER = 20.0
BASE_Y_NORMALIZER = 0.06
BASE_TILT_NORMALIZER = np.deg2rad(10.0)
BASE_YAW_NORMALIZER = np.deg2rad(18.0)
GRIPPER_Y_NORMALIZER = 0.05
GRIPPER_CENTER_Y_NORMALIZER = 0.04
GRIPPER_Y_SPAN_NORMALIZER = 0.10
GRIPPER_TILT_NORMALIZER = np.deg2rad(10.0)
GRIPPER_YAW_NORMALIZER = np.deg2rad(22.0)
UNSTABLE_GRIPPER_Y_LIMIT = 0.20
UNSTABLE_GRIPPER_TILT_LIMIT = np.deg2rad(50.0)
SUCCESS_BASE_Y_LIMIT = 0.10
SUCCESS_BASE_TILT_LIMIT = np.deg2rad(32.0)
SUCCESS_GRIPPER_CENTER_Y_LIMIT = 0.09
SUCCESS_GRIPPER_TILT_LIMIT = np.deg2rad(38.0)

# Settling time before each episode.
RESET_SETTLE_SECONDS = 0.25

# Video camera settings.
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = int(round(1.0 / CONTROL_DT))
CAMERA_DISTANCE = 1.6
CAMERA_YAW = -50.0
CAMERA_PITCH = -25.0
CAMERA_TARGET = [0.9, -0.03, 1.0]
CAMERA_FOV = 55.0

# Demo reference durations are interpreted as 240 Hz demo steps.
DEMO_REFERENCE_DT = 1.0 / 240.0


class LedgeClimbEnv(gym.Env):
    """Gymnasium environment for one selected ledge configuration."""

    metadata = {"render_modes": [None, "human"]}

    def __init__(
        self,
        env_id: str = "1",
        render_mode: Optional[str] = None,
        record_video: bool = False,
        video_path: Optional[str | Path] = None,
        max_episode_seconds: float = MAX_EPISODE_SECONDS,
        residual_action_limit_deg: float = RESIDUAL_ACTION_LIMIT_DEG,
        residual_limit_deg_by_joint: Optional[Dict[str, float]] = None,
        physics_dt: float = PHYSICS_DT,
        frame_skip: int = FRAME_SKIP,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        if env_id not in ENV_CONFIGS:
            raise ValueError(f"Unknown env_id={env_id!r}. Available ids: {list(ENV_CONFIGS)}")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'.")

        self.env_id = str(env_id)
        self.env_config = ENV_CONFIGS[self.env_id]
        self.render_mode = render_mode
        self.record_video = bool(record_video)
        self.video_path = Path(video_path) if video_path is not None else None
        self.physics_dt = float(physics_dt)
        self.frame_skip = int(frame_skip)
        self.control_dt = self.physics_dt * self.frame_skip
        self.max_episode_seconds = float(max_episode_seconds)
        self.max_episode_steps = int(round(self.max_episode_seconds / self.control_dt))
        self.fall_margin = FALL_MARGIN
        self.reward_weights = dict(REWARD_WEIGHTS if reward_weights is None else reward_weights)

        self.robot_path = Path(ROBOT_URDF_PATH)
        self.robot_mesh_dir = self.robot_path.parent / "meshes"
        self.env_path = Path(self.env_config["env_urdf"])
        self.robot_base_pos = list(self.env_config["robot_base_pos"])
        self.env_base_pos = list(self.env_config["env_base_pos"])

        self._check_required_files()
        self.client_id = self._connect_pybullet()

        self.robot_id: Optional[int] = None
        self.env_body_id: Optional[int] = None
        self.video_writer = None
        self.video_disabled_reason: Optional[str] = None

        (
            self.joint_indices,
            self.joint_low,
            self.joint_high,
            self.joint_name_to_index,
            self.joint_index_to_name,
        ) = self._init_joint_info()
        self.joint_index_to_action_pos = {j: i for i, j in enumerate(self.joint_indices)}
        self.action_pos_to_joint_index = {i: j for i, j in enumerate(self.joint_indices)}
        self.link_name_to_index = self._init_link_info()

        self.n_joints = len(self.joint_indices)

        
        #角度限制
        self.residual_action_limit_deg = float(residual_action_limit_deg)
        joint_limit_map = dict(RESIDUAL_ACTION_LIMITS_DEG)
        if residual_limit_deg_by_joint is not None:
            joint_limit_map.update({name: float(value) for name, value in residual_limit_deg_by_joint.items()})
        self.residual_limits_deg = np.asarray(
            [joint_limit_map.get(self.joint_index_to_name[joint_index], self.residual_action_limit_deg) for joint_index in self.joint_indices],
            dtype=np.float32,
        )
        self.residual_limit_rad = np.deg2rad(self.residual_limits_deg).astype(np.float32)
        self.residual_rate_limits_rad = np.deg2rad(
            [
                RESIDUAL_RATE_LIMITS_DEG_PER_STEP.get(self.joint_index_to_name[joint_index], 3.0)
                for joint_index in self.joint_indices
            ]
        ).astype(np.float32)
        
        
        self.residual_low = -self.residual_limit_rad
        self.residual_high = self.residual_limit_rad
        self.action_space = spaces.Box(
            low=self.residual_low,
            high=self.residual_high,
            shape=(self.n_joints,),
            dtype=np.float32,
        )

        # Observation blocks include joint states, base motion, gripper pose, previous residual, phase, and reference.
        self.obs_dim = (2 * self.n_joints) + 12 + 12 + 6 + self.n_joints + 1 + self.n_joints
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.fixed_left_claw_targets: Dict[int, float] = {}
        self.initial_body_targets: Dict[str, float] = {}
        self.initial_right_claw_targets: Dict[int, float] = {}
        self._init_default_joint_targets()

        self.step_counter = 0
        self.reference_step = 0
        self.reference_trajectory: Optional[np.ndarray] = None
        self.current_reference_action = np.zeros(self.n_joints, dtype=np.float32)
        self.prev_applied_action = np.zeros(self.n_joints, dtype=np.float32)
        self.prev_residual_action = np.zeros(self.n_joints, dtype=np.float32)
        self.start_base_x = 0.0
        self.prev_base_x = 0.0
        self.start_front_x = 0.0
        self.prev_front_x = 0.0
        self.max_forward_progress = 0.0
        self.target_x: Optional[float] = None
        self.ledge_height: Optional[float] = None
        self.robot_mass_kg = 1.0


    def _check_required_files(self) -> None:
        missing = []
        if not self.robot_path.exists():
            missing.append(str(self.robot_path))
        if not self.robot_mesh_dir.exists():
            missing.append(str(self.robot_mesh_dir))
        if not self.env_path.exists():
            missing.append(str(self.env_path))
        if missing:
            raise FileNotFoundError("Missing required file or directory:\n" + "\n".join(missing))
        self._validate_robot_mesh_paths()

    def _validate_robot_mesh_paths(self) -> None:
        text = self.robot_path.read_text(encoding="utf-8")
        mesh_filenames = re.findall(r'filename=["\']([^"\']+)["\']', text)
        unresolved = []
        for mesh_name in mesh_filenames:
            mesh_path = Path(mesh_name)
            candidates = []
            if mesh_path.is_absolute():
                candidates.append(mesh_path)
            else:
                candidates.append(self.robot_path.parent / mesh_path)
                if len(mesh_path.parts) == 1:
                    candidates.append(self.robot_mesh_dir / mesh_path.name)
            if not any(candidate.exists() for candidate in candidates):
                unresolved.append(mesh_name)
        if unresolved:
            msg = [
                "Robot URDF contains mesh paths that PyBullet is unlikely to resolve.",
                f"Robot URDF: {self.robot_path}",
                "Recommended form inside robot.urdf: meshes/<mesh_file>.obj",
                "Unresolved mesh path(s):",
            ]
            msg.extend(f"  {name}" for name in sorted(set(unresolved)))
            raise FileNotFoundError("\n".join(msg))

    def _set_search_paths(self, cid: Optional[int] = None) -> None:
        cid = self.client_id if cid is None else cid
        for path in [CODE_ROOT, URDF_ROOT, self.robot_path.parent, self.robot_mesh_dir, self.env_path.parent]:
            p.setAdditionalSearchPath(str(path), physicsClientId=cid)

    def _connect_pybullet(self) -> int:
        cid = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        self._set_search_paths(cid)
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=cid)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=cid)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=cid)
            self._reset_debug_camera(cid)
        return cid

    def _reset_debug_camera(self, cid: Optional[int] = None) -> None:
        cid = self.client_id if cid is None else cid
        if self.render_mode != "human":
            return
        p.resetDebugVisualizerCamera(
            cameraDistance=CAMERA_DISTANCE,
            cameraYaw=CAMERA_YAW,
            cameraPitch=CAMERA_PITCH,
            cameraTargetPosition=CAMERA_TARGET,
            physicsClientId=cid,
        )

    def _reset_simulation(self) -> None:
        p.resetSimulation(physicsClientId=self.client_id)
        self._set_search_paths()
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.physics_dt, physicsClientId=self.client_id)
        p.setRealTimeSimulation(0, physicsClientId=self.client_id)

    def _create_floor(self) -> int:
        floor_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[5.0, 5.0, 0.01],
            physicsClientId=self.client_id,
        )
        floor_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[5.0, 5.0, 0.01],
            rgbaColor=[0.75, 0.75, 0.75, 1.0],
            physicsClientId=self.client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=floor_collision,
            baseVisualShapeIndex=floor_visual,
            basePosition=[0.0, 0.0, -0.01],
            physicsClientId=self.client_id,
        )

    def _compute_robot_mass(self) -> float:
        if self.robot_id is None:
            return 1.0
        mass = float(p.getDynamicsInfo(self.robot_id, -1, physicsClientId=self.client_id)[0])
        for joint_idx in self.joint_indices:
            mass += float(p.getDynamicsInfo(self.robot_id, joint_idx, physicsClientId=self.client_id)[0])
        return max(mass, 1.0e-6)

    def _reset_cycle_and_energy_log(self) -> None:
        self.episode_abs_work_j = 0.0
        self.episode_signed_work_j = 0.0
        self.reference_cycle_steps = max(1, len(self.reference_trajectory))
        self.completed_cycle_count = 0
        self.cycle_start_step = 0
        self.cycle_start_reference_step = 0
        self.cycle_start_abs_work_j = 0.0
        self.cycle_start_signed_work_j = 0.0
        self.cycle_start_metrics = dict(self.initial_metrics)
        self.cycle_log: list[Dict[str, float]] = []
        self.partial_cycle_recorded = False

    def _accumulate_energy_step(self) -> None:
        if self.robot_id is None or not hasattr(self, "episode_abs_work_j"):
            return
        states = p.getJointStates(self.robot_id, self.joint_indices, physicsClientId=self.client_id)
        joint_vel = np.asarray([state[1] for state in states], dtype=np.float64)
        joint_torque = np.asarray([state[3] for state in states], dtype=np.float64)
        signed_power = float(np.sum(joint_torque * joint_vel))
        abs_power = float(np.sum(np.abs(joint_torque * joint_vel)))
        self.episode_signed_work_j += signed_power * self.physics_dt
        self.episode_abs_work_j += abs_power * self.physics_dt

    def _make_cycle_record(
        self,
        cycle_index: int,
        start_step: int,
        end_step: int,
        start_reference_step: int,
        end_reference_step: int,
        start_metrics: Dict[str, float],
        end_metrics: Dict[str, float],
        start_abs_work_j: float,
        end_abs_work_j: float,
        start_signed_work_j: float,
        end_signed_work_j: float,
        partial: bool,
    ) -> Dict[str, float]:
        duration_s = max((end_step - start_step) * self.control_dt, 1.0e-12)
        forward_distance_m = float(end_metrics["front_x"] - start_metrics["front_x"])
        forward_speed_mps = forward_distance_m / duration_s
        abs_work_j = float(end_abs_work_j - start_abs_work_j)
        signed_work_j = float(end_signed_work_j - start_signed_work_j)
        cot_den = self.robot_mass_kg * 9.81 * max(forward_distance_m, 1.0e-6)
        cot_abs = abs_work_j / cot_den if forward_distance_m > 1.0e-6 else float("nan")
        record: Dict[str, float] = {
            "cycle_index": float(cycle_index),
            "partial_cycle": float(bool(partial)),
            "start_step": float(start_step),
            "end_step": float(end_step),
            "start_time_s": float(start_step * self.control_dt),
            "end_time_s": float(end_step * self.control_dt),
            "cycle_duration_s": float(duration_s),
            "start_reference_step": float(start_reference_step),
            "end_reference_step": float(end_reference_step),
            "reference_cycle_fraction": float((end_reference_step - start_reference_step) / self.reference_cycle_steps),
            "start_front_x_m": float(start_metrics["front_x"]),
            "end_front_x_m": float(end_metrics["front_x"]),
            "forward_distance_m": float(forward_distance_m),
            "forward_speed_mps": float(forward_speed_mps),
            "abs_work_j": abs_work_j,
            "signed_work_j": signed_work_j,
            "cot_abs": float(cot_abs),
            "robot_mass_kg": float(self.robot_mass_kg),
        }
        for side in ("left", "right"):
            for suffix in ("x", "y", "z", "yaw", "pitch", "roll", "tilt_abs"):
                key = f"{side}_gripper_{suffix}"
                record[f"start_{key}"] = float(start_metrics.get(key, np.nan))
                record[f"end_{key}"] = float(end_metrics.get(key, np.nan))
        return record

    def _finalize_cycle_record(self, partial: bool = False) -> None:
        if self.robot_id is None:
            return
        end_metrics = self._get_task_metrics()
        cycle_index = self.completed_cycle_count + 1
        record = self._make_cycle_record(
            cycle_index=cycle_index,
            start_step=self.cycle_start_step,
            end_step=self.step_counter,
            start_reference_step=self.cycle_start_reference_step,
            end_reference_step=self.reference_step,
            start_metrics=self.cycle_start_metrics,
            end_metrics=end_metrics,
            start_abs_work_j=self.cycle_start_abs_work_j,
            end_abs_work_j=self.episode_abs_work_j,
            start_signed_work_j=self.cycle_start_signed_work_j,
            end_signed_work_j=self.episode_signed_work_j,
            partial=partial,
        )
        self.cycle_log.append(record)
        if not partial:
            self.completed_cycle_count += 1
            self.cycle_start_step = self.step_counter
            self.cycle_start_reference_step = self.reference_step
            self.cycle_start_abs_work_j = self.episode_abs_work_j
            self.cycle_start_signed_work_j = self.episode_signed_work_j
            self.cycle_start_metrics = end_metrics

    def _update_cycle_log(self, terminal: bool = False) -> None:
        if not hasattr(self, "reference_cycle_steps"):
            return
        while self.reference_step >= (self.completed_cycle_count + 1) * self.reference_cycle_steps:
            self._finalize_cycle_record(partial=False)
        if terminal and not self.partial_cycle_recorded and self.reference_step > self.cycle_start_reference_step:
            self._finalize_cycle_record(partial=True)
            self.partial_cycle_recorded = True

    def get_cycle_log(self) -> list[Dict[str, float]]:
        return [dict(row) for row in getattr(self, "cycle_log", [])]

    def _get_energy_and_cycle_metrics(self) -> Dict[str, float]:
        if not hasattr(self, "reference_cycle_steps"):
            return {
                "completed_cycle_count": 0.0,
                "fractional_cycle_count": 0.0,
                "reference_cycle_steps": float(len(self.reference_trajectory)),
                "episode_abs_work_j": 0.0,
                "episode_signed_work_j": 0.0,
                "episode_cot_abs": float("nan"),
                "cycle_mean_forward_distance_m": float("nan"),
                "cycle_mean_forward_speed_mps": float("nan"),
            }
        forward_progress = max(float(self._get_task_metrics()["forward_progress"]), 1.0e-6)
        episode_cot_abs = self.episode_abs_work_j / (self.robot_mass_kg * 9.81 * forward_progress)
        cycle_rows = [row for row in self.cycle_log if row.get("forward_distance_m", 0.0) > 0.0]
        if cycle_rows:
            mean_dist = float(np.mean([row["forward_distance_m"] for row in cycle_rows]))
            mean_speed = float(np.mean([row["forward_speed_mps"] for row in cycle_rows]))
        else:
            mean_dist = float("nan")
            mean_speed = float("nan")
        return {
            "completed_cycle_count": float(self.completed_cycle_count),
            "fractional_cycle_count": float(self.reference_step / self.reference_cycle_steps),
            "reference_cycle_steps": float(self.reference_cycle_steps),
            "episode_abs_work_j": float(self.episode_abs_work_j),
            "episode_signed_work_j": float(self.episode_signed_work_j),
            "episode_cot_abs": float(episode_cot_abs),
            "cycle_mean_forward_distance_m": mean_dist,
            "cycle_mean_forward_speed_mps": mean_speed,
        }

    def _load_scene(self) -> None:
        self._reset_simulation()
        self._create_floor()
        self.env_body_id = p.loadURDF(
            str(self.env_path),
            self.env_base_pos,
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        self.robot_id = p.loadURDF(
            str(self.robot_path),
            self.robot_base_pos,
            useFixedBase=False,
            physicsClientId=self.client_id,
        )
        self.robot_mass_kg = self._compute_robot_mass()
        self._reset_debug_camera()
        self._update_ledge_geometry()

    def _init_joint_info(self):
        self._reset_simulation()
        robot_id = p.loadURDF(
            str(self.robot_path),
            self.robot_base_pos,
            useFixedBase=False,
            physicsClientId=self.client_id,
        )

        joint_indices = []
        lows = []
        highs = []
        name_to_index = {}
        index_to_name = {}

        for i in range(p.getNumJoints(robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(robot_id, i, physicsClientId=self.client_id)
            joint_type = info[2]
            joint_name = info[1].decode("utf-8")
            name_to_index[joint_name] = i
            index_to_name[i] = joint_name
            if joint_type == p.JOINT_FIXED:
                continue
            joint_indices.append(i)
            lower, upper = float(info[8]), float(info[9])
            if joint_type == p.JOINT_REVOLUTE and (upper <= lower or upper - lower < 1e-6):
                lower, upper = -np.pi, np.pi
            lows.append(lower)
            highs.append(upper)

        if not joint_indices:
            raise RuntimeError("No movable joints found in robot URDF.")
        return (
            joint_indices,
            np.array(lows, dtype=np.float32),
            np.array(highs, dtype=np.float32),
            name_to_index,
            index_to_name,
        )

    def _init_link_info(self) -> Dict[str, int]:
        self._reset_simulation()
        robot_id = p.loadURDF(
            str(self.robot_path),
            self.robot_base_pos,
            useFixedBase=False,
            physicsClientId=self.client_id,
        )
        link_name_to_index = {"left_gripper": -1}
        for i in range(p.getNumJoints(robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(robot_id, i, physicsClientId=self.client_id)
            link_name = info[12].decode("utf-8")
            link_name_to_index[link_name] = i
        return link_name_to_index

    def _init_default_joint_targets(self) -> None:
        required_joints = [
            "left_upper_claw",
            "left_lower_claw",
            "right_upper_claw",
            "right_lower_claw",
            "left_elbow",
            "shoulder",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        ]
        missing = [name for name in required_joints if name not in self.joint_name_to_index]
        if missing:
            raise KeyError(
                "Robot URDF joint names do not match the expected demo names. "
                f"Missing: {missing}\nAvailable: {list(self.joint_name_to_index)}"
            )

        self.fixed_left_claw_targets = {
            self.joint_name_to_index["left_upper_claw"]: np.deg2rad(100.0),
            self.joint_name_to_index["left_lower_claw"]: np.deg2rad(-100.0),
        }
        self.initial_right_claw_targets = {
            self.joint_name_to_index["right_upper_claw"]: np.deg2rad(90.0),
            self.joint_name_to_index["right_lower_claw"]: np.deg2rad(-90.0),
        }
        self.initial_body_targets = {
            "left_elbow": np.deg2rad(45.0),
            "shoulder": np.deg2rad(90.0),
            "right_elbow": np.deg2rad(45.0),
        }


    def _demo_steps_to_control_steps(self, demo_steps: int) -> int:
        duration_seconds = float(demo_steps) * DEMO_REFERENCE_DT
        return max(1, int(round(duration_seconds / self.control_dt)))

    def _build_reference_trajectory(self, initial_targets: np.ndarray) -> np.ndarray:
        # Durations use the original 240 Hz demo time base and are converted to control steps.
        def scaled_duration(value: int) -> int:
            return max(1, int(round(value * REFERENCE_DURATION_SCALE)))

        keyframes = [
            {"duration": scaled_duration(120), "targets": {"right_upper_claw": 0.0, "right_lower_claw": 0.0}},
            {
                "duration": scaled_duration(120),
                "targets": {
                    "left_elbow": np.deg2rad(90.0),
                    "shoulder": np.deg2rad(5.0),
                    "right_elbow": np.deg2rad(90.0),
                },
            },
            {
                "duration": scaled_duration(120),
                "targets": {
                    "right_upper_claw": np.deg2rad(100.0),
                    "right_lower_claw": np.deg2rad(-100.0),
                },
            },
            {"duration": scaled_duration(60), "targets": {}},
            {"duration": scaled_duration(120), "targets": {"left_upper_claw": 0.0, "left_lower_claw": 0.0}},
            {
                "duration": scaled_duration(120),
                "targets": {
                    "left_elbow": np.deg2rad(45.0),
                    "shoulder": np.deg2rad(90.0),
                    "right_elbow": np.deg2rad(45.0),
                },
            },
            {
                "duration": scaled_duration(120),
                "targets": {
                    "left_upper_claw": np.deg2rad(100.0),
                    "left_lower_claw": np.deg2rad(-100.0),
                },
            },
        ]

        trajectory = []
        current = np.asarray(initial_targets, dtype=np.float32).copy()

        for frame in keyframes:
            duration = self._demo_steps_to_control_steps(int(frame["duration"]))
            start = current.copy()
            end = current.copy()
            for name, value in frame["targets"].items():
                joint_idx = self.joint_name_to_index[name]
                action_pos = self.joint_index_to_action_pos[joint_idx]
                end[action_pos] = float(value)

            for k in range(duration):
                # smooth_s = (k + 1) / duration
                # alpha = smooth_s * smooth_s * smooth_s * (10.0 - 15.0 * smooth_s + 6.0 * smooth_s * smooth_s)
                # target = (1.0 - alpha) * start + alpha * end
                
                alpha = (k + 1) / duration
                target = (1.0 - alpha) * start + alpha * end

                target = np.clip(target, self.joint_low, self.joint_high)
                trajectory.append(target.astype(np.float32))
            current = np.clip(end, self.joint_low, self.joint_high)

        return np.asarray(trajectory, dtype=np.float32)

    def _get_reference_action(self) -> Tuple[np.ndarray, int, float]:
        if self.reference_trajectory is None or len(self.reference_trajectory) == 0:
            return np.zeros(self.n_joints, dtype=np.float32), 0, 0.0
        ref_idx = self.reference_step % len(self.reference_trajectory)
        phase = ref_idx / max(1, len(self.reference_trajectory) - 1)
        return self.reference_trajectory[ref_idx].copy(), ref_idx, float(phase)


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_counter = 0
        self.reference_step = 0
        self.prev_residual_action = np.zeros(self.n_joints, dtype=np.float32)

        self._load_scene()
        assert self.robot_id is not None

        for joint_idx in self.joint_indices:
            p.resetJointState(self.robot_id, joint_idx, 0.0, 0.0, physicsClientId=self.client_id)

        target_positions = np.zeros(self.n_joints, dtype=np.float32)
        for joint_idx, joint_val in self.fixed_left_claw_targets.items():
            p.resetJointState(self.robot_id, joint_idx, joint_val, 0.0, physicsClientId=self.client_id)
            target_positions[self.joint_index_to_action_pos[joint_idx]] = joint_val

        for name, joint_val in self.initial_body_targets.items():
            joint_idx = self.joint_name_to_index[name]
            p.resetJointState(self.robot_id, joint_idx, joint_val, 0.0, physicsClientId=self.client_id)
            target_positions[self.joint_index_to_action_pos[joint_idx]] = joint_val

        target_positions = np.clip(target_positions, self.joint_low, self.joint_high)
        self._apply_position_control(target_positions)
        self._settle_robot()

        for joint_idx, joint_val in self.initial_right_claw_targets.items():
            p.resetJointState(self.robot_id, joint_idx, joint_val, 0.0, physicsClientId=self.client_id)
            target_positions[self.joint_index_to_action_pos[joint_idx]] = joint_val

        target_positions = np.clip(target_positions, self.joint_low, self.joint_high)
        self.prev_applied_action = target_positions.astype(np.float32).copy()
        self.reference_trajectory = self._build_reference_trajectory(target_positions)
        self.current_reference_action, _, _ = self._get_reference_action()

        self._apply_position_control(target_positions)
        self._settle_robot()

        initial_metrics = self._get_task_metrics()
        self.start_base_x = float(initial_metrics["base_x"])
        self.start_front_x = float(initial_metrics["front_x"])
        self.prev_base_x = self.start_base_x
        self.prev_front_x = self.start_front_x
        self.max_forward_progress = 0.0
        self.initial_metrics = dict(initial_metrics)
        self._reset_cycle_and_energy_log()

        obs = self._get_obs()
        info = self._get_info(
            residual_action=np.zeros(self.n_joints, dtype=np.float32),
            applied_action=target_positions,
            reward_terms={},
            joint_limit_clip_fraction=0.0,
            reference_index=0,
        )
        return obs, info

    def step(self, action: np.ndarray):
        assert self.robot_id is not None
        self.step_counter += 1

        self.current_reference_action, reference_index, _ = self._get_reference_action()
        self.reference_step += 1

        commanded_residual_action = np.asarray(action, dtype=np.float32)
        if commanded_residual_action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {commanded_residual_action.shape}")
        commanded_residual_action = np.nan_to_num(commanded_residual_action, nan=0.0, posinf=0.0, neginf=0.0)
        commanded_residual_action = np.clip(commanded_residual_action, self.residual_low, self.residual_high)
        alpha = float(np.clip(RESIDUAL_FILTER_ALPHA, 0.0, 1.0))
        proposed_residual_action = alpha * commanded_residual_action + (1.0 - alpha) * self.prev_residual_action
        residual_action = self._limit_residual_rate(proposed_residual_action, self.prev_residual_action)

        unclipped_applied = self.current_reference_action + residual_action
        applied_action = np.clip(unclipped_applied, self.joint_low, self.joint_high).astype(np.float32)
        joint_limit_clip_fraction = float(np.mean(np.abs(unclipped_applied - applied_action) > 1e-6))

        self._apply_position_control(applied_action)
        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.client_id)
            self._accumulate_energy_step()
            if self.render_mode == "human":
                time.sleep(self.physics_dt)

        metrics_before_reward = self._get_task_metrics()
        success, fall, timeout = self._check_termination(metrics_before_reward)
        reward, reward_terms = self._compute_reward(
            residual_action=residual_action,
            commanded_residual_action=commanded_residual_action,
            success=success,
            fall=fall,
            timeout=timeout,
        )

        self.prev_applied_action = applied_action.copy()
        self.prev_residual_action = residual_action.copy()
        obs = self._get_obs()

        terminated = bool(success or fall)
        truncated = bool(timeout and not terminated)
        self._update_cycle_log(terminal=bool(terminated or truncated))
        info = self._get_info(
            residual_action=residual_action,
            applied_action=applied_action,
            reward_terms=reward_terms,
            joint_limit_clip_fraction=joint_limit_clip_fraction,
            reference_index=reference_index,
            commanded_residual_action=commanded_residual_action,
        )

        if self.record_video:
            self._capture_video_frame()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._reset_debug_camera()
        return None

    def close(self) -> None:
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if getattr(self, "client_id", None) is not None and p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    # Control, observations, reward, termination

    def _limit_residual_rate(self, proposed: np.ndarray, previous: np.ndarray) -> np.ndarray:
        delta = np.clip(proposed - previous, -self.residual_rate_limits_rad, self.residual_rate_limits_rad)
        limited = previous + delta
        return np.clip(limited, self.residual_low, self.residual_high).astype(np.float32)

    def _apply_position_control(self, target_positions: np.ndarray) -> None:
        assert self.robot_id is not None
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions.tolist(),
            physicsClientId=self.client_id,
        )

    def _settle_robot(self) -> None:
        settle_steps = max(1, int(round(RESET_SETTLE_SECONDS / self.physics_dt)))
        for _ in range(settle_steps):
            p.stepSimulation(physicsClientId=self.client_id)
            self._accumulate_energy_step()
            if self.render_mode == "human":
                time.sleep(self.physics_dt)

    def _get_obs(self) -> np.ndarray:
        assert self.robot_id is not None
        joint_states = [p.getJointState(self.robot_id, i, physicsClientId=self.client_id) for i in self.joint_indices]
        joint_angles = np.array([s[0] for s in joint_states], dtype=np.float32)
        joint_vels = np.array([s[1] for s in joint_states], dtype=np.float32)

        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        base_linear_vel, base_angular_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)
        base_euler = p.getEulerFromQuaternion(base_orn)
        base_state = np.concatenate(
            [
                np.asarray(base_pos, dtype=np.float32),
                np.asarray(base_euler, dtype=np.float32),
                np.asarray(base_linear_vel, dtype=np.float32),
                np.asarray(base_angular_vel, dtype=np.float32),
            ]
        ).astype(np.float32)

        left_pos, left_euler = self._get_link_pose_euler("left_gripper")
        right_pos, right_euler = self._get_link_pose_euler("right_gripper")
        gripper_pose = np.concatenate([left_pos, left_euler, right_pos, right_euler]).astype(np.float32)

        _, left_vel = self._get_link_position_velocity("left_gripper")
        _, right_vel = self._get_link_position_velocity("right_gripper")
        gripper_velocities = np.concatenate([left_vel, right_vel]).astype(np.float32)


        _, _, reference_phase = self._get_reference_action()
        reference_phase_array = np.array([reference_phase], dtype=np.float32)

        obs = np.concatenate(
            [
                joint_angles,
                joint_vels,
                base_state,
                gripper_pose,
                gripper_velocities,
                self.prev_residual_action.astype(np.float32),
                reference_phase_array,
                self.current_reference_action.astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(f"Observation dimension mismatch: expected {self.obs_dim}, got {obs.shape[0]}")
        return obs

    def _get_link_position_velocity(self, link_name: str) -> Tuple[np.ndarray, np.ndarray]:
        assert self.robot_id is not None
        if link_name not in self.link_name_to_index:
            raise KeyError(f"Missing link name {link_name!r}. Available: {list(self.link_name_to_index)}")
        link_index = self.link_name_to_index[link_name]
        if link_index == -1:
            pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            linear_vel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)
        else:
            state = p.getLinkState(
                self.robot_id,
                link_index,
                computeLinkVelocity=1,
                physicsClientId=self.client_id,
            )
            pos = state[0]
            linear_vel = state[6]
        return np.array(pos, dtype=np.float32), np.array(linear_vel, dtype=np.float32)

    def _compute_reward(
        self,
        residual_action: np.ndarray,
        commanded_residual_action: np.ndarray,
        success: bool,
        fall: bool,
        timeout: bool,
    ) -> Tuple[float, Dict[str, float]]:
        metrics = self._get_task_metrics()
        delta_x = metrics["front_x"] - self.prev_front_x
        self.prev_base_x = metrics["base_x"]
        self.prev_front_x = metrics["front_x"]
        self.max_forward_progress = max(self.max_forward_progress, metrics["forward_progress"])

        residual_den = np.maximum(self.residual_limit_rad, 1e-8)
        action_norm = commanded_residual_action / residual_den
        smoothness_norm = (residual_action - self.prev_residual_action) / residual_den
        joint_velocity_norm = min(metrics["mean_abs_joint_velocity"] / JOINT_VELOCITY_NORMALIZER, 5.0)
        base_lateral_norm = min(abs(metrics["base_y"]) / BASE_Y_NORMALIZER, 5.0)
        base_tilt_norm = min(metrics["base_tilt_abs"] / BASE_TILT_NORMALIZER, 5.0)
        base_yaw_norm = min(abs(metrics["base_yaw"]) / BASE_YAW_NORMALIZER, 5.0)
        gripper_lateral_norm = min(metrics["max_gripper_abs_y"] / GRIPPER_Y_NORMALIZER, 5.0)
        gripper_center_y_norm = min(abs(metrics["gripper_center_y"]) / GRIPPER_CENTER_Y_NORMALIZER, 5.0)
        gripper_y_span_norm = min(metrics["gripper_y_span"] / GRIPPER_Y_SPAN_NORMALIZER, 5.0)
        gripper_tilt_norm = min(metrics["max_gripper_tilt_abs"] / GRIPPER_TILT_NORMALIZER, 5.0)
        gripper_yaw_norm = min(metrics["gripper_yaw_abs"] / GRIPPER_YAW_NORMALIZER, 5.0)

        progress_reward = self.reward_weights["progress"] * delta_x
        time_penalty = self.reward_weights["time"]
        action_cost = self.reward_weights["action"] * float(np.mean(action_norm ** 2))
        smoothness_cost = self.reward_weights["smoothness"] * float(np.mean(smoothness_norm ** 2))
        joint_velocity_cost = self.reward_weights["joint_velocity"] * float(joint_velocity_norm ** 2)
        base_lateral_cost = self.reward_weights["base_lateral"] * float(base_lateral_norm ** 2)
        base_tilt_cost = self.reward_weights["base_tilt"] * float(base_tilt_norm ** 2)
        base_yaw_cost = self.reward_weights["base_yaw"] * float(base_yaw_norm ** 2)
        gripper_lateral_cost = self.reward_weights["gripper_lateral"] * float(gripper_lateral_norm ** 2)
        gripper_center_y_cost = self.reward_weights["gripper_center_y"] * float(gripper_center_y_norm ** 2)
        gripper_y_span_cost = self.reward_weights["gripper_y_span"] * float(gripper_y_span_norm ** 2)
        gripper_tilt_cost = self.reward_weights["gripper_tilt"] * float(gripper_tilt_norm ** 2)
        gripper_yaw_cost = self.reward_weights["gripper_yaw"] * float(gripper_yaw_norm ** 2)
        success_bonus = self.reward_weights["success"] if success else 0.0
        elapsed_seconds = self.step_counter * self.control_dt
        success_speed_remaining_fraction = max(
            0.0,
            1.0 - elapsed_seconds / max(1e-6, SUCCESS_SPEED_TARGET_SECONDS),
        )
        success_speed_bonus = (
            self.reward_weights.get("success_speed", 0.0) * success_speed_remaining_fraction
            if success
            else 0.0
        )
        target_distance = max(metrics["target_x"] - self.start_front_x, 1e-6)
        progress_fraction = float(np.clip(metrics["forward_progress"] / target_distance, 0.0, 1.0))
        target_step_count = max(1, int(round(PACE_TARGET_SECONDS / self.control_dt)))
        target_elapsed_fraction = float(np.clip(self.step_counter / target_step_count, 0.0, 1.0))
        pace_error = max(0.0, target_elapsed_fraction - progress_fraction)
        pace_cost = self.reward_weights.get("pace", 0.0) * float(pace_error ** 2)
        fall_penalty = self.reward_weights["fall"] if fall else 0.0
        timeout_penalty = self.reward_weights["timeout"] if timeout and not success and not fall else 0.0

        reward = (
            progress_reward
            + time_penalty
            + action_cost
            + smoothness_cost
            + joint_velocity_cost
            + base_lateral_cost
            + base_tilt_cost
            + base_yaw_cost
            + gripper_lateral_cost
            + gripper_center_y_cost
            + gripper_y_span_cost
            + gripper_tilt_cost
            + gripper_yaw_cost
            + pace_cost
            + success_bonus
            + success_speed_bonus
            + fall_penalty
            + timeout_penalty
        )
        terms = {
            "progress_reward": float(progress_reward),
            "time_penalty": float(time_penalty),
            "action_cost": float(action_cost),
            "smoothness_cost": float(smoothness_cost),
            "joint_velocity_cost": float(joint_velocity_cost),
            "base_lateral_cost": float(base_lateral_cost),
            "base_tilt_cost": float(base_tilt_cost),
            "base_yaw_cost": float(base_yaw_cost),
            "gripper_lateral_cost": float(gripper_lateral_cost),
            "gripper_center_y_cost": float(gripper_center_y_cost),
            "gripper_y_span_cost": float(gripper_y_span_cost),
            "gripper_tilt_cost": float(gripper_tilt_cost),
            "gripper_yaw_cost": float(gripper_yaw_cost),
            "pace_cost": float(pace_cost),
            "progress_fraction": float(progress_fraction),
            "target_elapsed_fraction": float(target_elapsed_fraction),
            "pace_error": float(pace_error),
            "elapsed_seconds": float(elapsed_seconds),
            "pace_target_seconds": float(PACE_TARGET_SECONDS),
            "success_speed_target_seconds": float(SUCCESS_SPEED_TARGET_SECONDS),
            "success_speed_remaining_fraction": float(success_speed_remaining_fraction),
            "success_bonus": float(success_bonus),
            "success_speed_bonus": float(success_speed_bonus),
            "fall_penalty": float(fall_penalty),
            "timeout_penalty": float(timeout_penalty),
        }
        return float(reward), terms

    def _update_ledge_geometry(self) -> None:
        if self.env_body_id is None:
            return
        aabb_min, aabb_max = p.getAABB(self.env_body_id, -1, physicsClientId=self.client_id)
        self.ledge_height = float(aabb_max[2])
        configured_target_x = self.env_config.get("target_x", None)
        success_margin = float(self.env_config.get("success_margin", 0.05))
        if configured_target_x is None:
            self.target_x = float(aabb_max[0] - success_margin)
        else:
            self.target_x = float(configured_target_x)

    def _get_link_pose_euler(self, link_name: str) -> Tuple[np.ndarray, np.ndarray]:
        assert self.robot_id is not None
        if link_name not in self.link_name_to_index:
            raise KeyError(f"Missing link name {link_name!r}. Available: {list(self.link_name_to_index)}")
        link_index = self.link_name_to_index[link_name]
        if link_index == -1:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        else:
            state = p.getLinkState(self.robot_id, link_index, physicsClientId=self.client_id)
            pos, orn = state[0], state[1]
        euler = p.getEulerFromQuaternion(orn)
        return np.asarray(pos, dtype=np.float32), np.asarray(euler, dtype=np.float32)

    @staticmethod
    def _angle_abs(angle: float) -> float:
        return float(abs((angle + np.pi) % (2.0 * np.pi) - np.pi))

    @staticmethod
    def _angle_diff_abs(angle_a: float, angle_b: float) -> float:
        return float(abs((angle_a - angle_b + np.pi) % (2.0 * np.pi) - np.pi))

    def _get_task_metrics(self) -> Dict[str, float]:
        assert self.robot_id is not None
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        base_euler = p.getEulerFromQuaternion(base_orn)
        base_x = float(base_pos[0])
        base_y = float(base_pos[1])
        base_z = float(base_pos[2])
        base_roll = float(base_euler[0])
        base_pitch = float(base_euler[1])
        base_yaw = float(base_euler[2])
        base_tilt_abs = float(max(self._angle_abs(base_roll), self._angle_abs(base_pitch)))

        left_pos, left_euler = self._get_link_pose_euler("left_gripper")
        right_pos, right_euler = self._get_link_pose_euler("right_gripper")
        front_x = float(max(base_x, float(left_pos[0]), float(right_pos[0])))
        base_forward_progress = base_x - self.start_base_x
        forward_progress = front_x - self.start_front_x
        target_x = self.target_x if self.target_x is not None else 2.0
        ledge_height = self.ledge_height if self.ledge_height is not None else 1.0

        joint_states = [p.getJointState(self.robot_id, i, physicsClientId=self.client_id) for i in self.joint_indices]
        joint_vels = np.asarray([s[1] for s in joint_states], dtype=np.float32)
        mean_abs_joint_velocity = float(np.mean(np.abs(joint_vels))) if len(joint_vels) else 0.0
        max_abs_joint_velocity = float(np.max(np.abs(joint_vels))) if len(joint_vels) else 0.0

        contact_points = []
        if self.env_body_id is not None:
            contact_points = p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.env_body_id,
                physicsClientId=self.client_id,
            )
        total_normal_force = float(sum(point[9] for point in contact_points)) if contact_points else 0.0

        left_roll_abs = self._angle_abs(float(left_euler[0]))
        left_pitch_abs = self._angle_abs(float(left_euler[1]))
        right_roll_abs = self._angle_abs(float(right_euler[0]))
        right_pitch_abs = self._angle_abs(float(right_euler[1]))
        left_tilt = float(max(left_roll_abs, left_pitch_abs))
        right_tilt = float(max(right_roll_abs, right_pitch_abs))
        gripper_center_y = 0.5 * (float(left_pos[1]) + float(right_pos[1]))
        gripper_y_span = abs(float(left_pos[1]) - float(right_pos[1]))
        gripper_yaw_abs = max(self._angle_abs(float(left_euler[2])), self._angle_abs(float(right_euler[2])))
        gripper_yaw_diff_abs = self._angle_diff_abs(float(left_euler[2]), float(right_euler[2]))

        return {
            "base_x": base_x,
            "base_y": base_y,
            "base_z": base_z,
            "base_roll": base_roll,
            "base_pitch": base_pitch,
            "base_yaw": base_yaw,
            "base_tilt_abs": base_tilt_abs,
            "front_x": front_x,
            "forward_progress": float(forward_progress),
            "base_forward_progress": float(base_forward_progress),
            "target_x": float(target_x),
            "ledge_height": float(ledge_height),
            "max_forward_progress": float(self.max_forward_progress),
            "left_gripper_x": float(left_pos[0]),
            "left_gripper_y": float(left_pos[1]),
            "left_gripper_z": float(left_pos[2]),
            "right_gripper_x": float(right_pos[0]),
            "right_gripper_y": float(right_pos[1]),
            "right_gripper_z": float(right_pos[2]),
            "left_gripper_roll": float(left_euler[0]),
            "left_gripper_pitch": float(left_euler[1]),
            "left_gripper_yaw": float(left_euler[2]),
            "right_gripper_roll": float(right_euler[0]),
            "right_gripper_pitch": float(right_euler[1]),
            "right_gripper_yaw": float(right_euler[2]),
            "left_gripper_tilt_abs": float(left_tilt),
            "right_gripper_tilt_abs": float(right_tilt),
            "gripper_center_y": float(gripper_center_y),
            "gripper_y_span": float(gripper_y_span),
            "gripper_yaw_abs": float(gripper_yaw_abs),
            "gripper_yaw_diff_abs": float(gripper_yaw_diff_abs),
            "max_gripper_abs_y": float(max(abs(float(left_pos[1])), abs(float(right_pos[1])))),
            "max_gripper_tilt_abs": float(max(left_tilt, right_tilt)),
            "mean_abs_joint_velocity": mean_abs_joint_velocity,
            "max_abs_joint_velocity": max_abs_joint_velocity,
            "num_contacts": float(len(contact_points)),
            "total_normal_force": total_normal_force,
        }

    def _check_termination(self, metrics: Dict[str, float]) -> Tuple[bool, bool, bool]:
        fall_threshold = metrics["ledge_height"] - self.fall_margin
        raw_fall = metrics["base_z"] < fall_threshold
        raw_unstable = (
            metrics["max_gripper_abs_y"] > UNSTABLE_GRIPPER_Y_LIMIT
            or metrics["max_gripper_tilt_abs"] > UNSTABLE_GRIPPER_TILT_LIMIT
        )
        posture_ok_for_success = (
            abs(metrics["base_y"]) <= SUCCESS_BASE_Y_LIMIT
            and metrics["base_tilt_abs"] <= SUCCESS_BASE_TILT_LIMIT
            and abs(metrics["gripper_center_y"]) <= SUCCESS_GRIPPER_CENTER_Y_LIMIT
            and metrics["max_gripper_tilt_abs"] <= SUCCESS_GRIPPER_TILT_LIMIT
        )
        raw_success = metrics["front_x"] >= metrics["target_x"] and posture_ok_for_success
        raw_timeout = self.step_counter >= self.max_episode_steps

        if raw_success and not raw_fall and not raw_unstable:
            return True, False, False
        if raw_fall or raw_unstable:
            return False, True, False
        if raw_timeout:
            return False, False, True
        return False, False, False

    def _get_joint_debug_data(self) -> Tuple[list[str], np.ndarray, np.ndarray]:
        assert self.robot_id is not None
        joint_names = [self.joint_index_to_name[joint_idx] for joint_idx in self.joint_indices]
        states = [p.getJointState(self.robot_id, joint_idx, physicsClientId=self.client_id) for joint_idx in self.joint_indices]
        joint_angles = np.asarray([state[0] for state in states], dtype=np.float32)
        joint_velocities = np.asarray([state[1] for state in states], dtype=np.float32)
        return joint_names, joint_angles, joint_velocities

    def _get_info(
        self,
        residual_action: np.ndarray,
        applied_action: np.ndarray,
        reward_terms: Dict[str, float],
        joint_limit_clip_fraction: float,
        reference_index: int,
        commanded_residual_action: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        metrics = self._get_task_metrics()
        success, fall, timeout = self._check_termination(metrics)
        joint_names, joint_angles, joint_velocities = self._get_joint_debug_data()
        energy_cycle_metrics = self._get_energy_and_cycle_metrics()
        return {
            "env_id": self.env_id,
            "reference_index": int(reference_index),
            "reference_action": self.current_reference_action.copy(),
            "raw_residual_action": (
                commanded_residual_action.copy() if commanded_residual_action is not None else residual_action.copy()
            ),
            "residual_action": residual_action.copy(),
            "applied_action": applied_action.copy(),
            "joint_names": joint_names,
            "joint_angles": joint_angles.copy(),
            "joint_velocities": joint_velocities.copy(),
            "joint_limit_clip_fraction": float(joint_limit_clip_fraction),
            "task_metrics": metrics,
            "energy_cycle_metrics": energy_cycle_metrics,
            "success": bool(success),
            "fall": bool(fall),
            "timeout": bool(timeout),
            "reward_terms": dict(reward_terms),
        }

    def _prev_joint_action(self, joint_name: str) -> float:
        joint_index = self.joint_name_to_index[joint_name]
        action_pos = self.joint_index_to_action_pos[joint_index]
        return float(self.prev_applied_action[action_pos])


    def _ensure_video_writer(self) -> bool:
        if not self.record_video or self.video_path is None:
            return False
        if self.video_writer is not None:
            return True
        if imageio is None:
            self.video_disabled_reason = "imageio is not installed; MP4 recording is skipped."
            print(f"Video recording skipped: {self.video_disabled_reason}")
            self.record_video = False
            return False
        try:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            video_fps = max(1, int(round(1.0 / self.control_dt)))
            self.video_writer = imageio.get_writer(str(self.video_path), fps=video_fps)
        except Exception as exc:
            self.video_disabled_reason = f"failed to create MP4 writer: {exc}"
            print(f"Video recording skipped: {self.video_disabled_reason}")
            self.video_writer = None
            self.record_video = False
            return False
        return True

    def _capture_video_frame(self) -> None:
        if not self.record_video:
            return
        if not self._ensure_video_writer():
            return
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=CAMERA_TARGET,
            distance=CAMERA_DISTANCE,
            yaw=CAMERA_YAW,
            pitch=CAMERA_PITCH,
            roll=0.0,
            upAxisIndex=2,
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=CAMERA_FOV,
            aspect=VIDEO_WIDTH / VIDEO_HEIGHT,
            nearVal=0.01,
            farVal=10.0,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id,
        )
        frame = np.reshape(np.asarray(rgba, dtype=np.uint8), (VIDEO_HEIGHT, VIDEO_WIDTH, 4))[:, :, :3]
        self.video_writer.append_data(frame)


if __name__ == "__main__":
    print("envs_without_robust.py path:", Path(__file__).resolve())
    print("robot urdf:", ROBOT_URDF_PATH)
    print("env urdf:", ENV_CONFIGS["1"]["env_urdf"])
    env = LedgeClimbEnv(env_id="1", render_mode="human")
    obs, info = env.reset()
    print("obs_dim:", obs.shape, "act_dim:", env.action_space.shape)
    for _ in range(1000):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode ended:", info)
            break
    env.close()