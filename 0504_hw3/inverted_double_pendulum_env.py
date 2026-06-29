# Tips (作業方向提示)
# - 本題使用 gymnasium 風格包裝自製 PyBullet 雙倒單擺小車環境，方便演算法與 black-box 環境交互。
# - 本環境是連續控制任務：action 是 1 維連續推力，不是離散方向選擇。
# - observation 是 9 維狀態，包含小車位置、兩節桿角度的 sin/cos、速度與約束力資訊。
# - episode 從接近平衡的初始狀態開始。
# - reward 目標是讓桿端維持高處，同時懲罰小車偏移、角度偏移、速度過大與控制力過大。
# - terminated 條件包含小車出界、桿子倒太多、桿端高度過低。
# - 作業目標可以比較不同演算法或不同 baseline 設計，並畫出每個 episode return 曲線。

# 用 gymnasium 做交互環境要注意的地方
# - 一定要有 __init__()、reset()、step()，以及 _get_obs() / _get_info() 這類回傳資料的函式。
# - 在 __init__() 內要定義 action_space 與 observation_space。
# - reset() 回傳 (obs, info)；step() 回傳 (obs, reward, terminated, truncated, info)。
# - 如果有 render，記得在 metadata["render_modes"] 宣告可用模式。
# - PyBullet 環境要記得在 close() 中 disconnect，避免重複開啟 physics client。

from pathlib import Path
import time
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
import numpy as np
import pybullet as p
import pybullet_data


ENV_ID = "CustomInvertedDoublePendulum-v0"


class InvertedDoublePendulumEnv(gym.Env):
    """
    自製雙倒單擺小車環境。

    observation 採用接近 pybulletgym / MuJoCo 倒雙擺的 9 維格式：
        [cart_x,小車位置
         sin(theta_1), 第一節桿角度
         sin(theta_2),第二節桿相對第一節的角度
         cos(theta_1),
         cos(theta_2),
         cart_x_dot,小車速度
         theta_1_dot,第一節角速度
         theta_2_dot,第二節角速度
         cart_constraint_force_x]小車關節反作用力 x 方向

    action 是 1 維連續值，範圍為 [-1, 1]。
    環境會把 action 映射成小車水平方向的推力。
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        render_mode=None,
        gui_backend=False,
        max_steps=1000,
        time_step=1.0 / 240.0,
        frame_skip=4,
        force_mag=60.0,
    ):
        self.render_mode = render_mode if render_mode is not None else "none"
        self.gui_backend = bool(gui_backend or self.render_mode == "human")
        self.max_steps = int(max_steps)
        self.time_step = float(time_step)
        self.frame_skip = int(frame_skip)
        self.force_mag = float(force_mag)

        # episode 終止條件的門檻。
        self.max_cart_pos = 2.2
        self.max_joint_angle = np.deg2rad(50.0)
        self.min_tip_height = 0.45

        self.project_root = Path(__file__).resolve().parent
        self.asset_path = self.project_root / "inverted_double_pendulum_student_01.urdf"
        if not self.asset_path.exists():
            raise FileNotFoundError(f"URDF file not found: {self.asset_path}")
        #顯示設定，這裡是決定要不要開啟 PyBullet 的 GUI 模式。
        self.physics_client = p.connect(p.GUI if self.gui_backend else p.DIRECT)
        if self.gui_backend:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)
            p.configureDebugVisualizer(
                p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.physics_client
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.physics_client
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.physics_client
            )
            p.resetDebugVisualizerCamera(
                cameraDistance=2.4,
                cameraYaw=0,
                cameraPitch=-10,
                cameraTargetPosition=[0.0, 0.0, 0.7],
                physicsClientId=self.physics_client,
            )
        #這裡是代表動作是-1~1從最打向右推到最大向左的連續數值，而不是向右向左的離散選擇。
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_low = np.array(
            [-5.0, -1.0, -1.0, -1.0, -1.0, -20.0, -25.0, -25.0, -1.0e4],
            dtype=np.float32,
        )
        obs_high = np.array(
            [5.0, 1.0, 1.0, 1.0, 1.0, 20.0, 25.0, 25.0, 1.0e4],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.step_counter = 0
        self.plane_id = None
        self.robot_id = None
        self.cart_joint = 0
        self.pole_1_joint = 1
        self.pole_2_joint = 2
        self.tip_link_index = 2
        self.last_force = 0.0

    def reset(self, seed=None, options=None):
        """重設 PyBullet 世界，並把小車與兩節桿放回接近平衡的初始狀態。"""
        super().reset(seed=seed)
        self.step_counter = 0
        self.last_force = 0.0

        #清除pybullet裡面的資料
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        self.robot_id = p.loadURDF(
            str(self.asset_path),
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.physics_client,
        )

        # 關閉 URDF 預設馬達，讓小車完全由 torque control 驅動。
        for joint_index in (self.cart_joint, self.pole_1_joint, self.pole_2_joint):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self.physics_client,
            )
        #初始狀態的設定，讓小車位置、桿角度與速度都在接近平衡的範圍內隨機分布，增加訓練的多樣性。
        cart_pos = float(self.np_random.uniform(-0.05, 0.05))
        pole_1_angle = float(self.np_random.uniform(-0.08, 0.08))
        pole_2_angle = float(self.np_random.uniform(-0.08, 0.08))
        cart_vel = float(self.np_random.uniform(-0.02, 0.02))
        pole_1_vel = float(self.np_random.uniform(-0.02, 0.02))
        pole_2_vel = float(self.np_random.uniform(-0.02, 0.02))
        #將上面的隨機值寫入 PyBullet 的關節狀態，完成環境的重置。
        p.resetJointState(
            self.robot_id,
            self.cart_joint,
            targetValue=cart_pos,
            targetVelocity=cart_vel,
            physicsClientId=self.physics_client,
        )
        p.resetJointState(
            self.robot_id,
            self.pole_1_joint,
            targetValue=pole_1_angle,
            targetVelocity=pole_1_vel,
            physicsClientId=self.physics_client,
        )
        p.resetJointState(
            self.robot_id,
            self.pole_2_joint,
            targetValue=pole_2_angle,
            targetVelocity=pole_2_vel,
            physicsClientId=self.physics_client,
        )

        physics = self._get_physics_state()
        obs = self._build_obs(physics)
        info = self._get_info(physics)
        return obs, info

    def step(self, action):
        """把連續 action 轉成小車推力，推進模擬，回傳 Gymnasium 格式結果。
        action -> clip 到 [-1, 1]
       -> 乘 force_mag 變成實際推力
       -> 對 cart joint 施力
       -> PyBullet 模擬 frame_skip 次
       -> 讀取新的物理狀態
       -> 建 observation
       -> 算 reward
       -> 判斷 terminated / truncated
       -> 回傳結果

        """
        self.step_counter += 1

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        clipped = np.clip(action, self.action_space.low, self.action_space.high)#限制輸出範圍
        force = float(clipped[0] * self.force_mag)
        self.last_force = force

        for _ in range(self.frame_skip):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=self.cart_joint,
                controlMode=p.TORQUE_CONTROL,
                force=force,
                physicsClientId=self.physics_client,
            )
            p.stepSimulation(physicsClientId=self.physics_client)
            if self.render_mode == "human":
                time.sleep(self.time_step)

        physics = self._get_physics_state()
        obs = self._build_obs(physics)
        reward = self._compute_reward(physics, force)#回傳REWARD
        terminated = self._is_terminated(physics)
        truncated = self.step_counter >= self.max_steps
        info = self._get_info(physics)
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except p.error:
                pass
            self.physics_client = None

    def _get_physics_state(self):
        """集中讀取 PyBullet 的關節狀態與桿端狀態，避免各函式重複查詢。"""
        cart_state = p.getJointState(
            self.robot_id, self.cart_joint, physicsClientId=self.physics_client
        )
        pole_1_state = p.getJointState(
            self.robot_id, self.pole_1_joint, physicsClientId=self.physics_client
        )
        pole_2_state = p.getJointState(
            self.robot_id, self.pole_2_joint, physicsClientId=self.physics_client
        )
        tip_state = p.getLinkState(
            self.robot_id,
            self.tip_link_index,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
            physicsClientId=self.physics_client,
        )

        return {
            "cart_pos": float(cart_state[0]),
            "cart_vel": float(cart_state[1]),
            "cart_reaction": np.array(cart_state[2], dtype=np.float32),
            "theta_1": float(pole_1_state[0]),
            "theta_1_dot": float(pole_1_state[1]),
            "theta_2": float(pole_2_state[0]),
            "theta_2_dot": float(pole_2_state[1]),
            "tip_pos": np.array(tip_state[4], dtype=np.float32),
            "tip_vel": np.array(tip_state[6], dtype=np.float32),
        }

    def _build_obs(self, physics):
        """把物理狀態轉成 9 維 observation。"""
        return np.array(
            [
                physics["cart_pos"],
                np.sin(physics["theta_1"]),
                np.sin(physics["theta_2"]),
                np.cos(physics["theta_1"]),
                np.cos(physics["theta_2"]),
                physics["cart_vel"],
                physics["theta_1_dot"],
                physics["theta_2_dot"],
                float(physics["cart_reaction"][0]),
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, physics, force):
        """鼓勵桿端保持高處，並懲罰小車偏移、角度偏移、速度與過大控制力。"""
        cart_x = physics["cart_pos"]
        cart_x_dot = physics["cart_vel"]
        theta_1 = physics["theta_1"]
        theta_1_dot = physics["theta_1_dot"]
        theta_2 = physics["theta_2"]
        theta_2_dot = physics["theta_2_dot"]
        tip_z = float(physics["tip_pos"][2])
        tip_x_dot = float(physics["tip_vel"][0])
        absolute_tip_angle = theta_1 + theta_2

        return float(
            2.0
            + 2.0 * tip_z
            - 0.6 * (cart_x**2)
            - 0.03 * (cart_x_dot**2)
            - 3.0 * (theta_1**2)
            - 1.5 * (absolute_tip_angle**2)
            - 0.02 * (theta_1_dot**2)
            - 0.01 * (theta_2_dot**2)
            - 0.01 * (tip_x_dot**2)
            - 0.0005 * (force**2)
        )

    def _is_terminated(self, physics):
        """判斷 episode 是否因倒下或小車出界而結束。"""
        cart_x = physics["cart_pos"]
        theta_1 = physics["theta_1"]
        theta_2 = physics["theta_2"]
        tip_z = float(physics["tip_pos"][2])
        absolute_tip_angle = theta_1 + theta_2

        return bool(
            abs(cart_x) > self.max_cart_pos
            or abs(theta_1) > self.max_joint_angle
            or abs(absolute_tip_angle) > self.max_joint_angle
            or tip_z < self.min_tip_height
        )

    def _get_info(self, physics):
        """提供除 observation 之外，方便除錯與畫圖的可讀資訊。"""
        return {
            "cart_position": physics["cart_pos"],
            "cart_velocity": physics["cart_vel"],
            "pole_1_angle": physics["theta_1"],
            "pole_2_angle": physics["theta_2"],
            "tip_height": float(physics["tip_pos"][2]),
            "constraint_force_x": float(physics["cart_reaction"][0]),
            "force": float(self.last_force),
        }


if ENV_ID not in registry:
    register(id=ENV_ID, entry_point=InvertedDoublePendulumEnv)


if __name__ == "__main__":
    env = InvertedDoublePendulumEnv(render_mode="human", gui_backend=True, max_steps=600)
    obs, info = env.reset(seed=0)
    print("reset:", obs, info)
    try:
        for step in range(300):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if (step + 1) % 30 == 0 or terminated or truncated:
                print(
                    f"step={step + 1:03d} reward={reward: .3f} "
                    f"terminated={terminated} truncated={truncated} info={info}"
                )
            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()
