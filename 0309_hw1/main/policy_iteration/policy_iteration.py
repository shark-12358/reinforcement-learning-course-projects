# Tips (作業方向提示)
# - 需要依照題目改變環境定義（grid size、牆、獎勵、起點等）
# - 記得加入轉移模型（ 0.8/0.1/0.1 的隨機轉移）
# - non-terminal rewards 的 case要另外考慮
# - 作業目標要畫出圖，可以用matplotlib、也可以把data用csv存起來後在matlab出圖

import matplotlib.pyplot as plt
import numpy as np

# Simple 4x3 GridWorld (traditional: goal / hole / wall)
# GridWorld 環境尺寸與折扣因子
row = 6
col = 6
gamma = 0.99

# 動作定義：上、下、左、右
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 動作對應的隨機轉移（0.8/0.1/0.1）
up=[(-1, 0),(0, -1),(0, 1)]
down=[(1, 0),(0, -1),(0, 1)]
left=[(0, -1),(-1, 0),(1, 0)]
right=[(0, 1),(-1, 0),(1, 0)]

# 以動作為索引，對應到轉移的方向列表  
transition_dirs = [up, down, left, right]
probs = [0.8, 0.1, 0.1]


# Traditional 6*6 settings (row, col)
# 環境元素位置：起點、終點、陷阱、牆
START = (2, 4)
GOAL = (0, 5)  # terminal +1
BIGGOAL = (3, 0)  # terminal +3
HOLE = {(0, 0), (0, 1), (0, 4), (4, 4), (5, 0)}  # terminal -1
WALLS = {(0, 2), (2, 2), (3, 2), (5, 2)}

# 畫曲線用的追蹤狀態
STARTSTATE = START
GOALSTATE = (1, 5)
BIGGOALSTATES = ((2, 0), (3, 1), (4, 0))
HOLESTATES = ((0, 3), (1, 0), (1, 1), (1, 4), (3, 4), (4, 1), (4, 5), (5, 1), (5, 4))

# 每一步的預設獎勵
STEP_REWARD = -0.04

# 狀態索引轉換與合法性檢查
def state_to_idx(i: int, j: int) -> int:
    # 將 (row, col) 轉成單一索引
    return i * col + j


def idx_to_state(idx: int) -> tuple[int, int]:
    # 將單一索引轉回 (row, col)
    return idx // col, idx % col


def valid_state(i: int, j: int) -> bool:
    # 檢查是否在格子範圍內
    return 0 <= i < row and 0 <= j < col


def policy_iteration(theta: float = 1e-4):
    # 策略迭代：交替進行策略評估與策略改進
    # 狀態數與動作數
    nS = row * col
    nA = len(actions)

    # 初始策略：四個動作等機率
    # 初始化均勻策略與狀態價值
    #建立一個矩陣，裡面所有值都是 1/nA，表示每個動作的機率相同
    policy = np.full((nS, nA), 1.0 / nA, dtype=np.float32)
    V = np.zeros(nS, dtype=np.float32)

    # 存儲每次策略評估迭代後的狀態價值，方便畫圖
    START_HISTORY = []
    GOAL_HISTORY = []
    BIGGOAL_HISTORY = {state: [] for state in BIGGOALSTATES}
    HOLE_HISTORY = {state: [] for state in HOLESTATES}

    # 終止狀態索引集合
    terminal_states = {state_to_idx(*GOAL),state_to_idx(*BIGGOAL)}
    for a in HOLE:
        terminal_states.add(state_to_idx(*a))

    # 牆壁狀態索引集合
    wall_states = {state_to_idx(*w) for w in WALLS}

    # 反覆評估與改進直到策略穩定
    while True:
        # Policy Evaluation：在固定策略下估計 V(s)
        while True:
            # 單次策略評估迭代
            # 追蹤最大變化量以判斷收斂
            delta = 0.0
            new_V = np.zeros_like(V)
            for s in range(nS):
                # 終點/洞：視為 terminal
                # 終止狀態不更新
                if s in terminal_states or s in wall_states:
                    new_V[s] = 0.0
                    continue

                i, j = idx_to_state(s)
                q = np.zeros(nA, dtype=np.float32)
                v = 0.0
                # 依策略計算各動作的期望價值
                for a, (di, dj) in enumerate(actions):
                    dirs = transition_dirs[a]
                    q[a] = 0

                    for (ddi, ddj), p in zip(dirs, probs):
                        ni, nj = i + ddi, j + ddj
                        if valid_state(ni, nj) and (ni, nj) not in WALLS:
                            ns = state_to_idx(ni, nj)
                        else:
                            ns = s  # 撞牆或出界：留在原地

                        # 依下一位置決定即時獎勵
                        if (ni, nj) == GOAL:
                            reward = 1.0

                        elif (ni, nj) == BIGGOAL:
                            reward = 3.0
                            
                        elif (ni, nj) in HOLE:
                            reward = -1.0
                        else:
                            reward = STEP_REWARD
                        # 累積期望回報
                        #policy[s, a]的意思是從狀態 s 採取動作 a 的機率，例如第一個的往上，然後累加四個動作。
                        q[a] += p * (reward + gamma * V[ns])
                    v += policy[s, a] * q[a]


                new_V[s] = v
                # 更新最大變化量
                delta = max(delta, float(abs(new_V[s] - V[s])))

            V = new_V

            START_HISTORY.append(V[state_to_idx(*STARTSTATE)])
            GOAL_HISTORY.append(V[state_to_idx(*GOALSTATE)])
            for biggoal in BIGGOALSTATES:
                BIGGOAL_HISTORY[biggoal].append(V[state_to_idx(*biggoal)])
            for hole in HOLESTATES:
                HOLE_HISTORY[hole].append(V[state_to_idx(*hole)])

            # 變化量小於門標則完成評估
            if delta < theta:
                break

        # Policy Improvement：依照 V(s) 改成 greedy policy
        # 策略改進：對每個狀態取 greedy 動作
        policy_stable = True
        for s in range(nS):
            # 記錄改進前的動作以判斷策略是否改變
            old_action = int(np.argmax(policy[s]))

            if s in terminal_states or s in wall_states:
                continue

            i, j = idx_to_state(s)
            # 計算各動作的 Q 值
            q = np.zeros(nA, dtype=np.float32)
            for a, (di, dj) in enumerate(actions):

                dirs = transition_dirs[a]
                q[a] = 0 
                for (ddi, ddj), p in zip(dirs, probs):
                    ni, nj = i + ddi, j + ddj
                    if valid_state(ni, nj) and (ni, nj) not in WALLS:
                        ns = state_to_idx(ni, nj)
                    else:
                        ns = s

                    # 計算即時獎勵
                    if (ni, nj) == GOAL:
                        reward = 1.0
                    elif (ni, nj) == BIGGOAL:
                        reward = 3.0
                    elif (ni, nj) in HOLE:
                        reward = -1.0
                    else:
                        reward = STEP_REWARD
                    q[a] += p * (reward + gamma * V[ns]) 

            # 依最大 Q 值更新策略
            best_action = int(np.argmax(q))
            policy[s] = 0.0
            policy[s, best_action] = 1.0
            if best_action != old_action:
                policy_stable = False

        # 若策略不再改變則停止
        if policy_stable:
            break

    # V：最終最優 state value
    # policy：最優策略（每格只有一個動作為 1）
    return V, policy, START_HISTORY, GOAL_HISTORY, BIGGOAL_HISTORY, HOLE_HISTORY


def show_state_values(V: np.ndarray):
    # 以 4x3 的格式印出 V(s)
    for i in range(row):
        for j in range(col):
            print(f"{V[state_to_idx(i, j)]:6.2f}", end=" ")
        print()
    print()


def show_policy(policy: np.ndarray):
    # 以箭頭印出 greedy policy（牆、終點會顯示符號）
    arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    for i in range(row):
        row_out = []
        for j in range(col):
            if (i, j) in WALLS:
                row_out.append("W")
                continue
            if (i, j) == GOAL:
                row_out.append("G")
                continue
            if (i, j) == BIGGOAL:
                row_out.append("B")
                continue
            if (i, j) in HOLE:
                row_out.append("H")
                continue
            if (i, j) == START:
                row_out.append("S")
                continue
            s = state_to_idx(i, j)
            a = int(np.argmax(policy[s]))
            row_out.append(arrows.get(a, "."))
        print(" ".join(row_out))
    print()


def plot_histories(
    start_history: list[float],
    goal_history: list[float],
    biggoal_history: dict[tuple[int, int], list[float]],
    hole_history: dict[tuple[int, int], list[float]],
):
    plt.figure()
    plt.plot(start_history, label="Start")
    plt.plot(goal_history, label="Adj to +1")
    plt.xlabel("Iteration")
    plt.ylabel("Utility")
    plt.title("Utility vs Iteration")
    plt.legend()
    plt.show()

    plt.figure()
    for state in biggoal_history:
        plt.plot(biggoal_history[state], label="BIGGOAL " + str(state))
    plt.xlabel("Iteration")
    plt.ylabel("Utility")
    plt.title("Utility vs Iteration near +3")
    plt.legend()
    plt.show()

    plt.figure()
    for state in hole_history:
        plt.plot(hole_history[state], label="HOLE " + str(state))
    plt.xlabel("Iteration")
    plt.ylabel("Utility")
    plt.title("Utility vs Iteration near -1")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 執行策略迭代並顯示結果
    V, policy, START_HISTORY, GOAL_HISTORY, BIGGOAL_HISTORY, HOLE_HISTORY = policy_iteration(theta=1e-4)
    print("State Values:")
    show_state_values(V)
    print("Policy:")
    show_policy(policy)
    plot_histories(START_HISTORY, GOAL_HISTORY, BIGGOAL_HISTORY, HOLE_HISTORY)


# 說明：
# 若程式上遇到問題，可查閱相關資料（如 GitHub、YouTube、paper、AI）或尋求助教幫忙；
# 但最終成果的影片與報告中，需說明你已對程式有完整理解。
