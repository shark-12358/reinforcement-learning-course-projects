# Tips (作業方向提示)
# - 需要依照題目改變環境定義（grid size、牆、獎勵、起點等）
# - 記得加入轉移模型（ 0.8/0.1/0.1 的隨機轉移）
# - non-terminal rewards 的 case要另外考慮
# - 作業目標要畫出圖，可以用matplotlib、也可以把data用csv存起來後在matlab出圖

import numpy as np
import matplotlib.pyplot as plt


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

# Traditional 4x3 settings (row, col)
# 環境元素位置：起點、終點、陷阱、牆
START = (2, 4)
GOAL = (0, 5)  # terminal +1
BIGGOAL = (3, 0)  # terminal +3
HOLE = {(0, 0), (0, 1), (0, 4), (4, 4), (5, 0)}  # terminal -1
WALLS = {(0, 2), (2, 2), (3, 2), (5, 2)}

#畫曲線用的資料
STARTSTATE = (2, 4)
GOALSTATE = (1, 5)
BIGGOALSTATES = ((2, 0), (3, 1), (4, 0))
HOLESTATES = ((0, 3),(1, 0), (1, 1), (1, 4), (3, 4),(4, 1),(4, 5), (5, 1),(5,4))


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


def value_iteration(theta: float = 1e-4):
    # 值迭代：反覆更新 V(s) 直到收斂
    # 狀態數與動作數
    nS = row * col
    nA = len(actions)

    #存儲每次迭代的狀態價值，方便畫圖
    START_HISTORY = []
    GOAL_HISTORY = []
    BIGGOAL_HISTORY = {
        (2, 0): [],
        (3, 1): [],
        (4, 0): [],
    }
    HOLE_HISTORY = {
        (0, 3): [],
        (1, 0): [],
        (1, 1): [],
        (1, 4): [],
        (3, 4): [],
        (4, 1): [],
        (4, 5): [],
        (5, 1): [],
        (5, 4): [],
    }
    # 終止狀態索引集合
    terminal_states = {state_to_idx(*GOAL),state_to_idx(*BIGGOAL)}
    for a in HOLE:
        terminal_states.add(state_to_idx(*a))
     

    # 牆壁狀態索引集合
    wall_states = {state_to_idx(*w) for w in WALLS}

    # 初始化狀態價值
    V = np.zeros(nS, dtype=np.float32)

    # 反覆更新 V(s) 直到收斂
    while True:
        # 追蹤最大變化量以判斷收斂
        delta = 0.0
        new_V = np.zeros_like(V)
        
        for s in range(nS):
            # 終止或牆壁狀態不更新，簡單來說就是牆壁不是需要考慮的值，終點則在計算裡面給過了
            if s in terminal_states or s in wall_states:
                new_V[s] = 0.0
                continue

            i, j = idx_to_state(s)
            q = np.zeros(nA, dtype=np.float32)
            # 計算各動作的 Q 值
            for a, (di, dj) in enumerate(actions):
                
                dirs = transition_dirs[a]
                q[a] = 0 

                for (ddi, ddj), p in zip(dirs, probs):
                    ni, nj = i + ddi, j + ddj
                    
                    if valid_state(ni, nj) and (ni, nj) not in WALLS:
                        ns = state_to_idx(ni, nj)
                    else:
                        ns = s  #撞牆或出界：留在原地
    
                    # 依下一位置決定即時獎勵
                    if (ni, nj) == GOAL:
                        reward = 1.0
                    elif (ni, nj) == BIGGOAL:
                        reward = 3.0
                    elif (ni, nj) in HOLE:
                        reward = -1.0
                    else:
                        reward = STEP_REWARD
                    #最核心地方：計算 Q 值 = 即時獎勵 + 折扣 * 下一狀態價值
                    q[a] += p * (reward + gamma * V[ns])   


            # 取最大 Q 值更新 V(s)
            new_V[s] = np.max(q)
            delta = max(delta, float(abs(new_V[s] - V[s])))

        V = new_V

        START_HISTORY.append(V[state_to_idx(*STARTSTATE)])
        GOAL_HISTORY.append(V[state_to_idx(*GOALSTATE)])
        for biggoal in BIGGOALSTATES:
            BIGGOAL_HISTORY[biggoal].append(V[state_to_idx(*biggoal)])
        for hole in HOLESTATES:
            HOLE_HISTORY[hole].append(V[state_to_idx(*hole)])

        # 變化量小於門標則停止
        if delta < theta:
            break

    # greedy policy
    # 依最終 V(s) 取得 greedy policy
    # 初始化策略（終止/牆壁維持 -1）
    policy = np.full(nS, -1, dtype=np.int32)
    for s in range(nS):
        # 終止或牆壁狀態略過
        if s in terminal_states or s in wall_states:
            continue
        i, j = idx_to_state(s)
        # 計算各動作的 Q 值（用於選擇最佳動作）
        q = np.zeros(nA, dtype=np.float32)
        #建立方向跟所機率對應關係

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

        # 選擇最大 Q 值的動作
        policy[s] = int(np.argmax(q))

    # V：最優 state value
    # policy：由 V 得到的最優策略
    return V, policy,START_HISTORY,GOAL_HISTORY,BIGGOAL_HISTORY,HOLE_HISTORY





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
            if (i, j) in HOLE:
                row_out.append("H")
                continue
            if (i, j) == BIGGOAL:
                row_out.append("B")
                continue
            if (i, j) == START:
                row_out.append("S")
                continue
            s = state_to_idx(i, j)
            a = int(policy[s])
            row_out.append(arrows.get(a, "."))
        print(" ".join(row_out))
    print()


if __name__ == "__main__":
    # 執行值迭代並顯示結果
    V, policy,START_HISTORY,GOAL_HISTORY,BIGGOAL_HISTORY,HOLE_HISTORY = value_iteration(theta=1e-4)
    print("State Values:")
    show_state_values(V)
    print("Policy:")
    show_policy(policy)

#開始畫圖啦寶貝
#起點跟+1的圖
plt.figure()

plt.plot(START_HISTORY, label="Start")
plt.plot(GOAL_HISTORY, label="Adj to +1")

plt.xlabel("Iteration")
plt.ylabel("Utility")
plt.title("Utility vs Iteration at Start")
plt.legend()
plt.show()

#BIGGOAL的圖
plt.figure()
plt.xlabel("Iteration")
plt.ylabel("Utility")

for state in BIGGOAL_HISTORY:
    
    plt.plot(BIGGOAL_HISTORY[state], label="BIGGOAL"+str(state))
plt.title("Utility vs Iteration at BIGGOAL")
plt.legend()
plt.show()
#HOLE的圖
plt.figure()
plt.xlabel("Iteration")
plt.ylabel("Utility")

for state in HOLE_HISTORY:
    
    plt.plot(HOLE_HISTORY[state], label="HOLE"+str(state))
plt.title("Utility vs Iteration at HOLE")
plt.legend()
plt.show()



# 說明：
# 若程式上遇到問題，可查閱相關資料（如 GitHub、YouTube、paper、AI）或尋求助教幫忙；
# 但最終成果的影片與報告中，需說明你已對程式有完整理解。
