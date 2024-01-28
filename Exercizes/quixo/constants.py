from enum import Enum


MODE = 'train'           # 'train' or 'test' 
N = 5                   # Board size
VERSION = 0             # Version of the model to use
ITERATIONS = 50_000  # Number of iterations to train
TEST_ITERATION = 5_000  # Number of iterations to test
INVALID_SPACE = False   # Include invalid moves in the action space
TRANSFORMATION = False  # Use board transformations inside the network
INVALID_MOVES = False     # Allow invalid moves during the training, so the agent lose if it makes an invalid move
LOAD = 'simple'         # 'simple' or 'mix' select the model to load for the environment

'''Number of possible actions'''
ACTION_SPACE = 4 * (N - 1) * 4 if INVALID_SPACE else 4 * (N - 1) * 4 - 4*N 

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
MLP_0_HIDDEN_SIZE = 0
MLP_1_HIDDEN_SIZE = 512
MLP_2_HIDDEN_SIZE = 256

GAMMA = 0.5
BATCH_SIZE = 16
CACHE_SIZE = 10_000
TAU = 0.03
EPSILON_MODE = 0
EPSILON = 0.2
EPSILON_B = 1000 // BATCH_SIZE

'''Values for load and save'''
PATH = './models/'

def path(path=PATH, n=N, version=VERSION, iterations=ITERATIONS, invalid_space=INVALID_SPACE, invalid_moves=INVALID_MOVES, transformation=TRANSFORMATION, load=LOAD, mlp_0_size=MLP_0_HIDDEN_SIZE) -> str:
    '''Returns the path of the model to use'''

    return f'{path}model_{n}_v{version}_{iterations // 1000}K{"_IS" if invalid_space else ""}{"_IM" if invalid_moves else ""}{"_T" if transformation else ""}{f"_{load.upper()}" if load != "simple" else ""}{f"_{mlp_0_size}S" if mlp_0_size != 0 else ""}.pth'

MODEL_NAME = path()
LOAD_PATHS = {
    'simple': [path(version=v) for v in range(VERSION)],
    'all': [path(version=v, iterations=i, invalid_moves=im, mlp_0_size=0) for v in range(3) for i in [100_000, 1_000_000] for im in [False, True]],
    'mix': [
        f'{PATH}model_5_v0_100K_IM.pth',
        f'{PATH}model_5_v0_100K.pth',
        f'{PATH}model_5_v0_1000K_IM.pth',
        f'{PATH}model_5_v0_1000K.pth',
        f'{PATH}model_5_v1_100K_IM.pth',
        f'{PATH}model_5_v1_100K.pth',
        f'{PATH}model_5_v1_1000K_IM.pth',
        f'{PATH}model_5_v2_100K.pth',
        f'{PATH}model_5_v2_1000K_IM.pth',
    ],
}