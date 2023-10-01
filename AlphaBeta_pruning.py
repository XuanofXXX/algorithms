#coding:utf-8
'''
    Lab 2
    井字棋(Tic tac toe)Python语言实现, 带有Alpha-Beta剪枝的Minimax算法.
'''
import random

# 棋盘位置表示（0-8）:
# 0  1  2
# 3  4  5
# 6  7  8

# 设定获胜的组合方式(横、竖、斜)
WINNING_TRIADS = ((0, 1, 2), (3, 4, 5), (6, 7, 8),
                  (0, 3, 6), (1, 4, 7),(2, 5, 8),
                  (0, 4, 8), (2, 4, 6))
# 设定棋盘按一行三个打印
PRINTING_TRIADS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
# 用一维列表表示棋盘:
SLOTS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
# -1表示X玩家 0表示空位 1表示O玩家.
X_token = -1
Open_token = 0
O_token = 1

MARKERS = ['_', 'O', 'X']
END_PHRASE = ('平局', '胜利', '失败')

HUMAN = 1
COMPUTER = 0

def print_board(board):
    """打印当前棋盘"""
    for row in PRINTING_TRIADS:
        r = ' '
        for hole in row:
            r += MARKERS[board[hole]] + ' '
        print(r)


def legal_move_left(board):
    """ 判断棋盘上是否还有空位 """
    for slot in SLOTS:
        if board[slot] == Open_token:
            return True
    return False


def winner(board):
    """ 判断局面的胜者,返回值-1表示X获胜,1表示O获胜,0表示平局或者未结束"""
    for triad in WINNING_TRIADS:
        triad_sum = board[triad[0]] + board[triad[1]] + board[triad[2]]
        if triad_sum == 3 or triad_sum == -3:
            return board[triad[0]]  # 表示棋子的数值恰好也是-1:X,1:O
    return 0


class Move:
    def __init__(self, score):
        self.pos = None
        self.score = score


def alphaBeta(board, alpha, beta , player):
    """
    This function implements the alpha-beta pruning algorithm to determine the best move for a given player on a given board.

    Args:
    - board: a list representing the current state of the board
    - alpha: the best value that the maximizing player currently can guarantee at that level or above
    - beta: the best value that the minimizing player currently can guarantee at that level or above
    - player: an integer representing the current player (either COMPUTER or HUMAN)

    Returns:
    - bestmove: a Move object representing the best move for the given player on the given board
    """
    if winner(board) != 0 or not legal_move_left(board):
        return Move(winner(board))
    if player == COMPUTER:
        bestmove = Move(float('-inf'))
        for i in range(9):
            if board[i] != Open_token:
                continue
            board[i] = O_token
            move = alphaBeta(board, alpha, beta, HUMAN)
            board[i] = Open_token
            move.pos = i
            if move.score > bestmove.score:
                bestmove = move
            alpha = max(alpha, bestmove.score)
            if alpha >= beta:
                break
    else:
        bestmove = Move(float('inf'))
        for i in range(9):
            if board[i] != Open_token:
                continue
            board[i] = X_token
            move = alphaBeta(board, alpha, beta, COMPUTER)
            board[i] = Open_token
            move.pos = i
            if move.score < bestmove.score:
                bestmove = move
            beta = min(beta, bestmove.score)
            if alpha >= beta:
                break
    return bestmove


def determine_move(board):
    """
        决定电脑(玩家O)的下一步棋(使用Alpha-beta 剪枝优化搜索效率)
        Args:
            board (list):井字棋盘

        Returns:
            next_move(int): 电脑(玩家O) 下一步棋的位置
    """
    next_move = alphaBeta(board, float('-inf'), float('inf'), COMPUTER).pos
    return next_move

def main():
    """主函数,先决定谁是X(先手方),再开始下棋"""
    next_move = HUMAN
    opt = input("请选择先手方，输入X表示玩家先手，输入O表示电脑先手：")
    if opt in ("X", "x"):
        next_move = HUMAN
    elif opt in ("O", "o"):
        next_move = COMPUTER
    else:
        print("输入有误，默认玩家先手")

    # 初始化空棋盘
    board = [Open_token for i in range(9)]

    # 开始下棋
    while legal_move_left(board) and winner(board) == Open_token:
        print()
        print_board(board)
        if next_move == HUMAN and legal_move_left(board):
            try:
                print("\n")
                humanmv = int(input("请输入你要落子的位置(0-8)："))
                if board[humanmv] != Open_token:
                    continue
                board[humanmv] = X_token
                next_move = COMPUTER
            except:
                print("输入有误，请重试")
                continue
        if next_move == COMPUTER and legal_move_left(board) and winner(board) != -1:
            mymv = determine_move(board)
            print("Computer最终决定下在", mymv)
            board[mymv] = O_token
            next_move = HUMAN

    # 输出结果
    print_board(board)
    print(["平局", "Computer赢了", "你赢了"][winner(board)])


if __name__ == '__main__':
    main()
