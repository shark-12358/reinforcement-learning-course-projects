# Tips (作業方向提示)
# - 需要依照題目改變環境定義（grid size、牆、獎勵、起點等）
# - 記得加入轉移模型（ 0.8/0.1/0.1 的隨機轉移）
# - non-terminal rewards 的 case要另外考慮
# - 作業目標要畫出圖，可以用matplotlib、也可以把data用csv存起來後在matlab出圖

import matplotlib.pyplot as plt
import numpy as np

# GridWorld 環境尺寸與折扣因子
row = 6
col = 6
gamma = 0.99

# 動作定義：上、下、左、右
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 動作對應的隨機轉移（0.8/0.1/0.1）
up = [(-1, 0), (0, -1), (0, 1)]
down = [(1, 0), (0, -1), (0, 1)]
left = [(0, -1), (-1, 0), (1, 0)]
right = [(0, 1), (-1, 0), (1, 0)]

# 以動作為索引，對應到轉移的方向列表
transition_dirs = [up, down, left, right]
probs = [0.8, 0.1, 0.1]

# 6x6 地圖設定
START = (2, 4)
GOAL = (0, 5)  # non-terminal +1
BIGGOAL = (3, 0)  # non-terminal +3
HOLE = {(0, 0), (0, 1), (0, 4), (4, 4), (5, 0)}  # non-terminal -1
WALLS = {(0, 2), (2, 2), (3, 2), (5, 2)}

# 畫曲線用的追蹤狀態
STARTSTATE = START
GOALSTATE = (1, 5)
BIGGOALSTATES = ((2, 0), (3, 1), (4, 0))
HOLESTATES = ((0, 3), (1, 0), (1, 1), (1, 4), (3, 4), (4, 1), (4, 5), (5, 1), (5, 4))

# 每一步的預設獎勵
STEP_REWARD = -0.04


def state_to_idx(i: int, j: int) -> int:
    return i * col + j


def idx_to_state(idx: int) -> tuple[int, int]:
    return idx // col, idx % col


def valid_state(i: int, j: int) -> bool:
    return 0 <= i < row and 0 <= j < col


def policy_iteration(theta: float = 1e-4):
    nS = row * col
    nA = len(actions)

    policy = np.full((nS, nA), 1.0 / nA, dtype=np.float32)
    V = np.zeros(nS, dtype=np.float32)

    START_HISTORY = []
    GOAL_HISTORY = []
    BIGGOAL_HISTORY = {state: [] for state in BIGGOALSTATES}
    HOLE_HISTORY = {state: [] for state in HOLESTATES}

    wall_states = {state_to_idx(*w) for w in WALLS}

    while True:
        while True:
            delta = 0.0
            new_V = np.zeros_like(V)
            for s in range(nS):
                if s in wall_states:
                    new_V[s] = 0.0
                    continue

                i, j = idx_to_state(s)
                q = np.zeros(nA, dtype=np.float32)
                v = 0.0
                for a, (di, dj) in enumerate(actions):
                    dirs = transition_dirs[a]
                    q[a] = 0.0

                    for (ddi, ddj), p in zip(dirs, probs):
                        ni, nj = i + ddi, j + ddj
                        if valid_state(ni, nj) and (ni, nj) not in WALLS:
                            ns = state_to_idx(ni, nj)
                        else:
                            ns = s

                        if (ni, nj) == GOAL:
                            reward = 1.0
                        elif (ni, nj) == BIGGOAL:
                            reward = 3.0
                        elif (ni, nj) in HOLE:
                            reward = -1.0
                        else:
                            reward = STEP_REWARD
                        q[a] += p * (reward + gamma * V[ns])
                    v += policy[s, a] * q[a]

                new_V[s] = v
                delta = max(delta, float(abs(new_V[s] - V[s])))

            V = new_V

            START_HISTORY.append(V[state_to_idx(*STARTSTATE)])
            GOAL_HISTORY.append(V[state_to_idx(*GOALSTATE)])
            for biggoal in BIGGOALSTATES:
                BIGGOAL_HISTORY[biggoal].append(V[state_to_idx(*biggoal)])
            for hole in HOLESTATES:
                HOLE_HISTORY[hole].append(V[state_to_idx(*hole)])

            if delta < theta:
                break

        policy_stable = True
        for s in range(nS):
            old_action = int(np.argmax(policy[s]))

            if s in wall_states:
                continue

            i, j = idx_to_state(s)
            q = np.zeros(nA, dtype=np.float32)
            for a, (di, dj) in enumerate(actions):
                dirs = transition_dirs[a]
                q[a] = 0.0
                for (ddi, ddj), p in zip(dirs, probs):
                    ni, nj = i + ddi, j + ddj
                    if valid_state(ni, nj) and (ni, nj) not in WALLS:
                        ns = state_to_idx(ni, nj)
                    else:
                        ns = s

                    if (ni, nj) == GOAL:
                        reward = 1.0
                    elif (ni, nj) == BIGGOAL:
                        reward = 3.0
                    elif (ni, nj) in HOLE:
                        reward = -1.0
                    else:
                        reward = STEP_REWARD
                    q[a] += p * (reward + gamma * V[ns])

            best_action = int(np.argmax(q))
            policy[s] = 0.0
            policy[s, best_action] = 1.0
            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            break

    return V, policy, START_HISTORY, GOAL_HISTORY, BIGGOAL_HISTORY, HOLE_HISTORY


def show_state_values(V: np.ndarray):
    for i in range(row):
        for j in range(col):
            print(f"{V[state_to_idx(i, j)]:6.2f}", end=" ")
        print()
    print()


def show_policy(policy: np.ndarray):
    arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    for i in range(row):
        row_out = []
        for j in range(col):
            if (i, j) in WALLS:
                row_out.append("W")
                continue

            s = state_to_idx(i, j)
            a = int(np.argmax(policy[s]))
            arrow = arrows.get(a, ".")

            if (i, j) == GOAL:
                row_out.append("G" + arrow)
            elif (i, j) == BIGGOAL:
                row_out.append("B" + arrow)
            elif (i, j) in HOLE:
                row_out.append("H" + arrow)
            elif (i, j) == START:
                row_out.append("S" + arrow)
            else:
                row_out.append(arrow)
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
    V, policy, START_HISTORY, GOAL_HISTORY, BIGGOAL_HISTORY, HOLE_HISTORY = policy_iteration(theta=1e-4)
    print("State Values:")
    show_state_values(V)
    print("Policy:")
    show_policy(policy)
    plot_histories(START_HISTORY, GOAL_HISTORY, BIGGOAL_HISTORY, HOLE_HISTORY)
