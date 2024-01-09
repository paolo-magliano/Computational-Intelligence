from enum import Enum

MODE = 'test'
N = 5
VERSION = 1
ITERATIONS = 1_000_000
TEST_ITERATION = 5_000

INVALID_MOVES = False
TRANSFORMATION = False
ACTION_SPACE = 4 * (N - 1) * 4 if INVALID_MOVES else 4 * (N - 1) * 4 - 4*N 

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
MODEL_NAME = f'model_{N}_v{VERSION}_{ITERATIONS // 1000}K{"_invalid_moves" if INVALID_MOVES else ""}.pth'

MLP_1_HIDDEN_SIZE = 512
MLP_2_HIDDEN_SIZE = 256

GAMMA = 0.5
BATCH_SIZE = 16
TAU = 0.03
EPSILON_MODE = 0
EPSILON = 0.2
EPSILON_B = 1000 // BATCH_SIZE

