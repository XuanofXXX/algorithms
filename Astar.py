import numpy as np
import random
import heapq as hq
import time


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time()  
        run_time = end_time - start_time  
        print(f"函数 {func.__name__} 运行时间: {run_time} 秒")
        return result
    return wrapper


class PuzzleBoardState(object):
    """ 华容道棋盘类
    """
    def __init__(self, dim=3, random_seed=2022, data=None, parent=None):
        """ 根据给定随机数 随机初始化一个可解的华容道问题 
            dim         :   int, 华容道棋盘维度(阶数)    
            random_seed :   int, 创建随机棋盘的随机种子    
                            可尝试不同的随机数 创建不同的棋盘
            data        :   numpy.ndarray (dim*dim), 创建棋盘的数据
                            可给定一个有解的初始棋盘 (程序未设置对给定数据的可解性检查)
            parent      :   PuzzleBoardState, 设定棋盘状态的父节点状态
                            根节点/初始节点的父节点为空
        """
        self.dim = dim
        self.default_dst_data = np.array([[1,2,3], [4,5,6], [7,8,0]])
        if data is None:
            init_solvable = False
            init_count = 0
            while not init_solvable and init_count<500:
                init_data = self._get_random_data(random_seed=random_seed+init_count)
                init_count += 1
                init_solvable = self._if_solvable(init_data, self.default_dst_data)
                # print(init_data)
            data = init_data
        self.data = data
        self.parent = parent
        self.piece_x, self.piece_y = self._get_piece_index()
        # print(self.piece_x, self.piece_y)
    
    def __eq__(self, __value: object) -> bool:
        return True
    
    def __lt__(self, other):
        return True
    
    def _get_random_data(self, random_seed):
        """ 根据random_seed 生成一个dim*dim的华容道棋盘数据
            random_seed :   int, 随机数
            return      :   numpy.ndarray (dim*dim), 华容道棋盘数据
        """
        random.seed(random_seed)
        init_data = [i for i in range(self.dim**2)]
        random.shuffle(init_data)
        init_data = np.array(init_data).reshape((self.dim, self.dim))

        return init_data

    def _get_piece_index(self):
        """ 返回当前将牌(空格)位置
            return :    int, 将牌横坐标 (axis=0)
                        int, 将牌纵坐标 (axis=0)
        """
        index = np.argsort(self.data.flatten())[0]

        return index//self.dim, index%self.dim

    def h(self):
        """获得一个棋盘下的h_score

        Args:
            puzzle_board_data (_type_): _description_

        Returns:
            int : h_score
        """
        flatten_data = self.data.flatten()
        res = 0 
        for i in range(len(flatten_data)):
            if flatten_data[i] != i + 1:
                res += 1
        return res
    
    def _inverse_num(self, puzzle_board_data):
        flatten_data = puzzle_board_data.flatten()
        res = 0
        for i in range(len(flatten_data)):
            if flatten_data[i] == 0:
                continue
            for j in range(i):
                if flatten_data[j] > flatten_data[i]:
                    res += 1
        
        return res

    def _if_solvable(self, src_data, dst_data):
        """ 判断一个(src_data => dst_data)的华容道问题是否可解
            src_data : numpy.ndarray (dim*dim), 作判断的棋盘初始状态数据
            src_data : numpy.ndarray (dim*dim), 作判断的棋盘终止状态数据
            return :    boolean, True可解 False不可解
        """
        assert src_data.shape == dst_data.shape, "src_data and dst_data should share same shape."
        inverse_num_sum = self._inverse_num(src_data) + self._inverse_num(dst_data)

        return inverse_num_sum%2 == 0

    def is_final(self):
        """ 判断棋盘当前状态是否为目标终止状态  
            return :    boolean, True终止 False未终止
        """
        flatten_data = self.data.flatten()
        if flatten_data[-1] != 0:
            return False
        for i in range(self.dim**2 - 1):
            if flatten_data[i] != (i + 1):
                return False
        return True

    def next_states(self):
        """ 返回当前状态的相邻状态
            return :    list, 当前状态的相邻状态，构成的PuzzleBoardState对象列表
        """
        res = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            x2, y2 = self.piece_x + dx, self.piece_y + dy
            if 0 <= x2 < self.dim and 0 <= y2 < self.dim:
                new_data = self.data.copy()
                new_data[self.piece_x][self.piece_y] = new_data[x2][y2]
                new_data[x2][y2] = 0
                res.append(PuzzleBoardState(data=new_data, parent=self))
                # print(new_data)

        return res

    def get_data(self):
        """ 返回当前棋盘状态数据
            return :    numpy.ndarray (dim*dim), 当前棋盘的数据
        """
        return self.data

    def get_data_hash(self):
        """ 返回基于当前状态数据的哈希值 存储在set中 供判断相同状态使用 
            return :    int, 当前状态哈希值
        """
        return hash(tuple(self.data.flatten()))

    def get_parent(self):
        """ 返回当前状态的父节点状态
            return :    PuzzleBoardState, 当前状态的父节点 
        """
        return self.parent

@timing_decorator
def bfs(puzzle_board_state):
    """ 已实现的华容道广度优先算法 供参考 """
    visited = set()

    from collections import deque
    queue = deque()
    queue.append((0, puzzle_board_state))
    visited.add(puzzle_board_state.get_data_hash())

    ans = []
    while queue:
        (now, cur_state) = queue.popleft()
        if cur_state.is_final():
            while cur_state.get_parent() is not None:
                ans.append(cur_state)
                cur_state = cur_state.get_parent()
            ans.append(cur_state)
            print('bfs(visited):', len(visited))
            break

        next_states = cur_state.next_states()
        for next_state in next_states:
            if next_state.get_data_hash() in visited:
                continue
            visited.add(next_state.get_data_hash())
            queue.append((now + 1, next_state))

    return ans

    
@timing_decorator
def astar(puzzle_board_state):
    visited = {}
    queue = []
    hq.heappush(queue,(puzzle_board_state.h(), 0, puzzle_board_state))
    visited[puzzle_board_state.get_data_hash()] = 0

    ans = []
    while queue:
        # 从队列中取出估计成本最小的状态
        cos, now, cur_state = hq.heappop(queue)

        if cur_state.is_final():
            while cur_state.get_parent() is not None:
                ans.append(cur_state)
                cur_state = cur_state.get_parent()
            ans.append(cur_state)
            print('astar(visited):', len(visited))
            break

        next_states = cur_state.next_states()
        for next_state in next_states:
            if next_state.get_data_hash() not in visited or now + 1 < visited[next_state.get_data_hash()]:
                visited[next_state.get_data_hash()] = now + 1
                total_cost = next_state.h() + now + 1
                hq.heappush(queue, (total_cost, now+1, next_state))
    return ans

if __name__ == "__main__":
    
    from collections import deque    
    test_data = np.array([[5,3,6], [1,8,4], [7,2,0]])
    test_data =None 
    # random_seed = 20230919
    test_board = PuzzleBoardState(data=test_data)
    test_board.next_states()
    res1 = astar(test_board)
    res2 = bfs(test_board)
    # for i in res1:
    #     print(i.data) 
    print(len(res1), len(res2))