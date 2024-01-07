from enum import Enum

MODE = 'test'
ITERATIONS = 10000
N = 3
INVALID_MOVES = False
ACTION_SPACE = 4 * (N - 1) * 4 if INVALID_MOVES else 4 * (N - 1) * 4  - 4*N 

'''Encoding of the moves'''
class Move(Enum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3

'''Encoding of the board'''
X = 1
O = -1
EMPTY = 0

CHARS = {
    X: 'X',
    O: 'O',
    EMPTY: '.'
}

'''Reward values'''
INVALID_MOVE_REWARD = 0
MOVE_REWARD = 0.05
WIN_REWARD = 1
LOSE_REWARD = 0
DRAW_REWARD = 0

'''Values for the DQN player'''
PATH = './models/'
MODEL_NAME = f'model_{N}{"_invalid_moves" if INVALID_MOVES else ""}_{ITERATIONS}.pth'
MLP_1_HIDDEN_SIZE = 512
MLP_2_HIDDEN_SIZE = 256
EPSILON = 0.2
GAMMA = 0.8
BATCH_SIZE = 2 * ACTION_SPACE
TAU = 0.02

