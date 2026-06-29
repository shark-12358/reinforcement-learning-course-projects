# Tips (作業方向提示)
# - 用gymnasium的方式做成環境，方便演算法與black-box交互
# - 需要依照題目改變環境定義
# - episode 要從「隨機合法、非終止的白色格子」開始
# - 白格 reward = -1 且非終止，黑格 reward = -100 且終止，綠格 reward = +100 且終止
# - 作業目標要畫出圖，可以用matplotlib、也可以把data用csv存起來後在matlab出圖

# 用gymnasium做交互環境要注意的地方
# - 一定要有 __init__()、reset()、step()，以及 _get_obs() / _get_info() 這類回傳資料的函式
# - 在 __init__() 內要定義 action_space 與 observation_space
# - reset() 回傳 (obs, info)；step() 回傳 (obs, reward, terminated, truncated, info)
# - 如果有 render，記得在 metadata["render_modes"] 宣告可用模式

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry


# 簡易 GridWorld 環境定義
class SimpleGridWorldEnv(gym.Env):
    """
    State  : (row, col)
    Action : 0=up, 1=down, 2=left, 3=right
    Reward : goal +100, wall -100, else -1
    Usage:
        import gymnasium as gym
        env = gym.make("SimpleGridWorld-v0")
    """

    metadata = {"render_modes": ["human", "ansi"]}
    
    

    def __init__(self, render_mode=None, max_steps=None):#self用來隔開兩個環境
        # 環境與格子設定
        self.row = 11
        self.col = 11
        self.start = (1, 1)#隨機給一個合法起點，後面會再隨機抽。
        self.goal = (0, 5)
        self.walls = {
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10),
            (1, 0), (1, 10),
            (2, 0), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (2, 10),
            (3, 0), (3, 8), (3, 10),
            (4, 0), (4, 1), (4, 2), (4, 4), (4, 5), (4, 6), (4, 8), (4, 9), (4, 10),
            (6, 0), (6, 1), (6, 2), (6, 3), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10),
            (7, 0), (7, 10),
            (8, 0), (8, 1), (8, 2), (8, 4), (8, 5), (8, 6), (8, 8), (8, 9), (8, 10),
            (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
        }
        

        # 定義 action / observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([self.row, self.col])

        # 渲染與步數限制設定
        self.render_mode = render_mode
        self.render_mode = render_mode
        self.max_steps = max_steps if max_steps is not None else self.row * self.col * 4
        self._step_count = 0
        self._agent_pos = np.array(self.start, dtype=np.int32)

        # 動作對應的位移 (上、下、左、右)
        self._actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def randomstart   (self):
        #起點設定
        valid_starts = []
        for i in range(self.row):
            for j in range(self.col):
                if (i, j) not in self.walls and (i, j) != self.goal:
                    valid_starts.append((i, j))

        idx = self.np_random.integers(len(valid_starts))
        start = valid_starts[idx]

                    

        return (start)
    

    def reset(self, seed=None, options=None):
        # 重置環境狀態
        super().reset(seed=seed)#這行是遵守 Gymnasium 的規範。如果環境之後有隨機性，seed 管理會靠這邊。
        self._step_count = 0
        super().reset(seed=seed)
        self._step_count = 0
        self.start = self.randomstart()
        self._agent_pos = np.array(self.start, dtype=np.int32)
        return self._get_obs(), self._get_info()

    def step(self, action):
        # 執行一步互動
        if not self.action_space.contains(action):#檢查輸入動作是否屬於合法，若不合法則拋出錯誤。
            raise ValueError(f"Invalid action: {action}")

        # 根據動作計算下一位置
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        di, dj = self._actions[int(action)]
        ni = int(self._agent_pos[0] + di)
        nj = int(self._agent_pos[1] + dj)

        #確定再格子裡面才更新位置，否則保持原地不動
        if 0 <= ni < self.row and 0 <= nj < self.col :
            self._agent_pos = np.array([ni, nj], dtype=np.int32)

        self._step_count += 1

        # 計算 reward 與終止條件
        terminated = False
        truncated = False
        reward = -1

        pos = (int(self._agent_pos[0]), int(self._agent_pos[1]))
        if pos == self.goal:
            reward = 100
            terminated = True
        elif pos in self.walls:
            reward = -100
            terminated = True

        if self._step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        # 文字方式渲染格子內容
        if self.render_mode is None:
            return None

        grid = [["." for _ in range(self.col)] for _ in range(self.row)]
        for (wi, wj) in self.walls:
            grid[wi][wj] = "W"
        gi, gj = self.goal
        grid[gi][gj] = "G"
        si, sj = self.start
        if 0 <= si < self.row and 0 <= sj < self.col:
            grid[si][sj] = "S"
        ai, aj = int(self._agent_pos[0]), int(self._agent_pos[1])
        grid[ai][aj] = "A"

        lines = [" ".join(row) for row in grid]
        text = "\n".join(lines)

        if self.render_mode == "human":
            print(text)
            return None
        if self.render_mode == "ansi":
            return text
        return None

    def _get_obs(self):
        # 回傳觀測 (agent 位置)
        return self._agent_pos.copy()

    def _get_info(self):
        # 回傳附加資訊
        return {"position": (int(self._agent_pos[0]), int(self._agent_pos[1]))}


# 註冊環境，方便用 gym.make 建立
ENV_ID = "SimpleGridWorld-v0"
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="simple_gridworld_env:SimpleGridWorldEnv")


if __name__ == "__main__":
    # 簡單測試環境互動
    env = SimpleGridWorldEnv(render_mode="human")
    obs, info = env.reset()
    print("start:", obs, info)
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("step:", obs, reward, terminated, truncated, info)
        env.render()
        if terminated or truncated:
            break

# 說明：
# 若程式上遇到問題，可查閱相關資料（如 GitHub、YouTube、paper、AI）或尋求助教幫忙；
# 但最終成果的影片與報告中，需說明你已對程式有完整理解。
