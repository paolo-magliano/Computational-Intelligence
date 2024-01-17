from enum import Enum

MODE = 'train'           # 'train' or 'test' 
N = 5                   # Board size
VERSION = 0             # Version of the model to use
ITERATIONS = 5_000  # Number of iterations to train
TEST_ITERATION = 5_000  # Number of iterations to test
INVALID_MOVES = False   # Include invalid moves in the action space
TRANSFORMATION = False  # Use board transformations inside the network

'''Number of possible actions'''
ACTION_SPACE = 4 * (N - 1) * 4 if INVALID_MOVES else 4 * (N - 1) * 4 - 4*N 

'''Encoding of the board'''
X = 1
O = 0
EMPTY = -1

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
MODEL_NAME = f'model1_{N}_v{VERSION}_{ITERATIONS // 1000}K{"_invalid_moves" if INVALID_MOVES else ""}.pth'

MLP_1_HIDDEN_SIZE = 512
MLP_2_HIDDEN_SIZE = 256

GAMMA = 0.5
BATCH_SIZE = 16
TAU = 0.03
EPSILON_MODE = 0
EPSILON = 0.2
EPSILON_B = 1000 // BATCH_SIZE

