from abc import ABC, abstractmethod
import random
import os
import numpy as np
from copy import deepcopy
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from constants import *
from true_game import Player, Game, Move
from game_ext import GameExt
from utils import *
from transformation import *

class RandomPlayer(Player):
    '''Player that makes random moves'''

    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, N - 1), random.randint(0, N - 1))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class LastMovePlayer(Player):
    '''Player that makes a winning move if possible, really slow'''

    def __init__(self, base_player=RandomPlayer()) -> None:
        super().__init__()
        self.base_player = base_player
        self.masks = self.__get_masks()
        self.last_board = None
        self.lose_check = False
    
    def __get_masks(self) -> list[np.ndarray]:
        '''Get the masks for each possible move'''
        masks = {}
        for i in range(2*N + 2):            
            for j in range(N):
                mask = np.zeros((N, N), dtype=np.uint8)
                if i < N:
                    mask[i, :j] = 1
                    mask[i, j + 1:] = 1
                    masks[self.___mask_hash(mask)] = ((j, i), 'O')
                elif i < 2*N:
                    mask[:j, i - N] = 1
                    mask[j + 1:, i - N] = 1
                    masks[self.___mask_hash(mask)] = ((i - N, j), 'V')
                elif i == 2*N:
                    mask[:j, :j] = np.diag(np.ones(j, dtype=np.uint8))
                    mask[j + 1:, j + 1:] = np.diag(np.ones(N - j - 1, dtype=np.uint8))
                    masks[self.___mask_hash(mask)] = ((j, j), 'D')
                else:
                    mask[:j, :j] = np.diag(np.ones(j, dtype=np.uint8))
                    mask[j + 1:, j + 1:] = np.diag(np.ones(N - j - 1, dtype=np.uint8))
                    mask = np.fliplr(mask)
                    masks[self.___mask_hash(mask)] = ((N - 1 - j, j), 'D')

        return masks
    
    def ___mask_hash(self, mask: np.ndarray) -> int:
        '''Get the hash of the mask'''
        return hash(str(mask.flatten()))
    
    def __win_get_move(self, board: np.ndarray, player: int, move_pos: (int, int), orientation: str) -> tuple[tuple[int, int], Move]:
        '''Get the winning move put symbol in move_pos'''
        for move in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]:
            if move == Move.TOP and ((move_pos[1] - 1 < 0 and orientation == 'O') or (move_pos[1] - 1 >= 0 and board[move_pos[1] - 1, move_pos[0]] == player)):
                end = move_pos[1] + 1 if orientation == 'V' else N
                start = move_pos[1] if move_pos[1] != 0 else move_pos[1] + 1
                for i in range(start, end):
                    if (move_pos[0] == 0 or move_pos[0] == N - 1 or i == 0 or i == N - 1) and (board[i, move_pos[0]] == EMPTY or board[i, move_pos[0]] == player):
                        return (move_pos[0], i), move
            elif move == Move.BOTTOM and ((move_pos[1] + 1 >= N and orientation == 'O') or (move_pos[1] + 1 < N and board[move_pos[1] + 1, move_pos[0]] == player)):
                end = move_pos[1] - 1 if orientation == 'V' else -1
                start = move_pos[1] if move_pos[1] != N - 1 else move_pos[1] - 1
                for i in range(start, end, -1):
                    if (move_pos[0] == 0 or move_pos[0] == N - 1 or i == 0 or i == N - 1) and (board[i, move_pos[0]] == EMPTY or board[i, move_pos[0]] == player):
                        return (move_pos[0], i), move
            elif move == Move.LEFT and ((move_pos[0] - 1 < 0 and orientation == 'V') or (move_pos[0] - 1 >= 0 and board[move_pos[1], move_pos[0] - 1] == player)):
                end = move_pos[0] + 1 if orientation == 'O' else N
                start = move_pos[0] if move_pos[0] != 0 else move_pos[0] + 1
                for i in range(start, end):
                    if (move_pos[1] == 0 or move_pos[1] == N - 1 or i == 0 or i == N - 1) and (board[move_pos[1], i] == EMPTY or board[move_pos[1], i] == player):
                        return (i, move_pos[1]), move
            elif move == Move.RIGHT and ((move_pos[0] + 1 >= N and orientation == 'V') or (move_pos[0] + 1 < N and board[move_pos[1], move_pos[0] + 1] == player)):
                end = move_pos[0] - 1 if orientation == 'O' else -1
                start = move_pos[0] if move_pos[0] != N - 1 else move_pos[0] - 1
                for i in range(start, end, -1):
                    if (move_pos[1] == 0 or move_pos[1] == N - 1 or i == 0 or i == N - 1) and (board[move_pos[1], i] == EMPTY or board[move_pos[1], i] == player):
                        return (i, move_pos[1]), move
        return None 
    
    def __check_lose_move(self, game: 'Game', move: ((int, int), Move)) -> bool:
        test_game = GameExt(board=game.get_board(), n=N)
        ok = test_game.move(move[0], move[1], game.get_current_player())
        if not ok:
            return False
        if test_game.check_winner() == game.get_current_player():
            return True
        elif test_game.check_winner() == 1 - game.get_current_player():
            return False
        test_game.set_current_player(1 - game.get_current_player())

        board  = test_game.get_board()
        player = test_game.get_current_player()   
        win_masks = self.__mask_board(board, player)
        for win_mask in win_masks:
            hash_win_mask = self.___mask_hash(win_mask)
            if hash_win_mask in self.masks:
                move_pos, orientation = self.masks[hash_win_mask]
                move = self.__win_get_move(board, player, move_pos, orientation)
                if move:
                    return False
                
        return True

    def __mask_board(self, board: np.ndarray, player: int) -> np.ndarray:
        '''Mask the board with the player'''
        mask = np.zeros((N, N), dtype=np.uint8)
        mask_list = []
        for i in range(2*N + 2):
            'Check if there are N - 1 piace in the row, column or diagonal'
            if i < N:
                if np.count_nonzero(board[i, :] == player) == N - 1:
                    mask[i, :] = board[i, :] == player
                    mask_list.append(deepcopy(mask))
                    mask = np.zeros((N, N), dtype=np.uint8)
            elif i < 2*N:
                if np.count_nonzero(board[:, i - N] == player) == N - 1:
                    mask[:, i - N] = board[:, i - N] == player
                    mask_list.append(deepcopy(mask))
                    mask = np.zeros((N, N), dtype=np.uint8)
            elif i == 2*N:
                if np.count_nonzero(np.diag(board) == player) == N - 1:
                    mask[np.diag_indices(N)] = np.diag(board) == player
                    mask_list.append(deepcopy(mask))
                    mask = np.zeros((N, N), dtype=np.uint8)
            else:
                if np.count_nonzero(np.diag(np.fliplr(board)) == player) == N - 1:
                    rot_mask = np.fliplr(mask)
                    rot_mask[np.diag_indices(N)] = np.diag(np.fliplr(board)) == player
                    mask = np.fliplr(rot_mask)
                    mask_list.append(deepcopy(mask))
                    mask = np.zeros((N, N), dtype=np.uint8)


        return mask_list
                
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        '''Make the winning move or the not losing move if possible'''
        board = game.get_board()
        equal = self.last_board is not None and np.array_equal(self.last_board, board)
        player = game.get_current_player()
        win_masks = self.__mask_board(board, player)
        lose_masks = self.__mask_board(board, 1 - player)
        if not equal:
            self.last_board = deepcopy(board)
            self.lose_check = False

        if not equal:         
            for win_mask in win_masks:
                hash_win_mask = self.___mask_hash(win_mask)
                if hash_win_mask in self.masks:
                    move_pos, orientation = self.masks[hash_win_mask]
                    move = self.__win_get_move(board, player, move_pos, orientation)
                    if move:
                        return move
                
        hash_bools = [(self.___mask_hash(lm) in self.masks) for lm in lose_masks]
    
        if any(hash_bools) and not (equal and self.lose_check):
            possible_moves = ACTION_SPACE if type(self.base_player) == DQNPlayer else 4*N*N
            for i in range(possible_moves):
                move = self.base_player.make_move(game)
                
                ok = self.__check_lose_move(game, move)
                if ok:
                    return move
            self.lose_check = True
            if type(self.base_player) == DQNPlayer:
                self.base_player.invalid_moves = []

        move = self.base_player.make_move(game)
        return move
    
    def update(self, states: list['Game'], actions: list[tuple[tuple[int, int], Move]], rewards: list[float]) -> None:
        if type(self.base_player) == DQNPlayer:
            self.base_player.update(states, actions, rewards)
    

class HumanPlayer(Player):
    '''Player that asks the user for the move'''

    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game', player: int) -> tuple[tuple[int, int], Move]:
        game.print()
        col = int(input('Col: '))
        row = int(input('Row: '))
        move = Move[input('Move: ')]
        return (col, row), move
    
class DQN(nn.Module):
    '''Deep Q Network for the agent player'''
    def __init__(self, input_size: int = N * N, mlp_0_size: int = MLP_0_HIDDEN_SIZE, mlp_1_size: int = MLP_1_HIDDEN_SIZE, mlp_2_size: int = MLP_2_HIDDEN_SIZE, output_size: int = ACTION_SPACE) -> None:
        super().__init__()
        self.fc0 = nn.Linear(input_size, mlp_0_size) if mlp_0_size else None
        self.fc1 = nn.Linear(mlp_0_size, mlp_1_size) if mlp_0_size else nn.Linear(input_size, mlp_1_size)
        self.fc2 = nn.Linear(mlp_1_size, mlp_2_size)
        self.fc3 = nn.Linear(mlp_2_size, output_size)
        self.non_linearity = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.flatten().float()
        if self.fc0:
            x = self.non_linearity(self.fc0(x))
        x = self.non_linearity(self.fc1(x))
        x = self.non_linearity(self.fc2(x))
        x = self.fc3(x)

        return x

class DQNPlayer(Player):
    def __init__(self, mode: str = 'train', load: bool = False, path: str = MODEL_NAME) -> None:
        super().__init__()
        '''Attribute about the agent'''
        self.mode = mode
        self.n_steps = 0
        self.previous_games = []  
        self.invalid_moves = []
        self.invalid_game = None
        self.transformation_cache = {}
        self.path = ""

        '''Attributes about the network'''

        if (self.mode == 'test' or load) and os.path.exists(path):
            mlp_0_size = int(re.search(r'_(\d+)S', path).group(1)) if re.search(r'_(\d+)S', path) else 0
            self.policy_net = DQN(mlp_0_size=mlp_0_size)
            self.target_net = DQN(mlp_0_size=mlp_0_size)
            self.policy_net.load_state_dict(torch.load(path))
            self.policy_net.eval()
            self.path = path
        else:
            self.policy_net = DQN()
            self.target_net = DQN()
            
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()
                  

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        board = torch.tensor(game.get_board())
        
        norm_board, transformations = self.__normalize(board) if TRANSFORMATION else (board, [])
        
        epsilon = EPSILON if EPSILON_MODE == 0 else EPSILON_B / (EPSILON_B + self.n_steps)
        if random.random() < epsilon and self.mode == 'train':
            '''Exploration choice, the move is chosen with a probability proportional to the q values and that is not the best one'''
            ok = False

            actions_score = self.policy_net(norm_board)
            actions_score = F.softmax(actions_score, dim=0)
            actions_score[torch.argmax(actions_score)] = 0
            actions_score = actions_score / torch.sum(actions_score)

            '''Get a possible valid move according to the probability distribution of the q values'''
            while not ok:
                action_index = random.choices(range(ACTION_SPACE), weights=actions_score.tolist())[0]
                norm_from_pos, norm_move = get_move_from_index(action_index)
                # print(f'Norm from pos: {norm_from_pos}, norm move: {norm_move}')
                # print(f'Inverse')
                from_pos, move = transform_move(norm_from_pos, norm_move, get_move_transformations(get_inverse_transformation(transformations))) if TRANSFORMATION else (norm_from_pos, norm_move)
                if self.invalid_game and np.array_equal(self.invalid_game.get_board(), game.get_board()) and (from_pos, move) in self.invalid_moves:
                    ok = False
                else:
                    ok = True

        else:
            '''Exploitation choice, the move is chosen with the highest q value'''  

            '''Get the vector of q values for each move'''
            if self.mode == 'test':
                with torch.no_grad():
                    actions_score = self.policy_net(norm_board)
            else:
                actions_score = self.policy_net(norm_board)

            '''Get a possible valid move with the highest q value'''
            ok = False
            k = 0
            while not ok:
                action_index = torch.topk(actions_score, 1 + k).indices[-1].item()
                norm_from_pos, norm_move = get_move_from_index(action_index)
                from_pos, move = transform_move(norm_from_pos, norm_move, get_move_transformations(get_inverse_transformation(transformations))) if TRANSFORMATION else (norm_from_pos, norm_move)
                if self.invalid_game and np.array_equal(self.invalid_game.get_board(), game.get_board()) and (from_pos, move) in self.invalid_moves:
                    k += 1
                    if k >= ACTION_SPACE - 1:
                        k = 0
                        self.invalid_moves = []
                else:
                    ok = True

            '''Print the q values for each move'''
            # array_to_print = list(zip([get_move_from_index(i) for i in range(ACTION_SPACE)], actions_score.tolist()))
            # for i in range(ACTION_SPACE//4):
            #     for j in range(4):
            #         action, score = array_to_print[i + j * ACTION_SPACE//4] 
            #         print(f'  Move {action}: {score:.2f} {"CHOSEN" if action == (from_pos, move) else "      "}', end='')
            #     print()

            '''Save the move and the game to check later if the agent choose an invalid move'''
            if self.invalid_game and np.array_equal(self.invalid_game.get_board(), game.get_board()):
                self.invalid_moves.append((from_pos, move))
            else:
                self.invalid_game = deepcopy(game)
                self.invalid_moves = [(from_pos, move)]

        return from_pos, move
    
    def update(self, states: list['Game'], actions: list[tuple[tuple[int, int], Move]], rewards: list[float]) -> None:
        '''Update the network using the previous games'''
        self.previous_games.append((states, rewards, actions))
        
        '''Update the netowrk only if there are enough games'''
        if len(self.previous_games) >= BATCH_SIZE:
            random_games = random.choices(self.previous_games, k=BATCH_SIZE)

            for states, rewards, actions in random_games:
                for i in range(len(states) - 1):
                    state = torch.tensor(states[i]._board)
                    norm_state, transformation_state = self.__normalize(state) if TRANSFORMATION else (state, [])
                    action = actions[i]
                    norm_action = transform_move(action[0], action[1], get_move_transformations(transformation_state)) if TRANSFORMATION else action
                    reward = rewards[i]
                    next_state = torch.tensor(states[i + 1]._board) if i + 1 < len(states) - 1 else None
                    norm_next_state, _ = self.__normalize(next_state) if next_state is not None else (None, None) if TRANSFORMATION else (next_state, [])
                    
                    action_index = get_index_from_move(norm_action)

                    '''Compute the current q values'''
                    q_values = self.policy_net(norm_state)

                    '''Compute the expected q values using the target network'''
                    expected_q_values = q_values.clone()
                    expected_q_values[action_index] = reward + GAMMA * torch.max(self.target_net(norm_next_state)) if norm_next_state is not None else reward
                    
                    '''Acccumulate the gradients for the loss function'''
                    loss = self.loss_function(q_values, expected_q_values)
                    loss.backward()

            '''Update the network'''
            self.optimizer.step()
            self.optimizer.zero_grad()

            '''Update the target network'''
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            self.target_net.load_state_dict(target_net_state_dict)

            '''Reset the previous games'''
            self.previous_games = []
            self.n_steps += 1

    def __normalize(self, board: torch.Tensor) -> tuple[torch.Tensor, list[(Callable, Union[int, None])]]:
        '''Normalize the board'''
        if len(self.transformation_cache) > CACHE_SIZE:
            self.transformation_cache.pop(next(iter(self.transformation_cache)))
        if self.transformation_cache.get(board):
            normalized_board, transformations = self.transformation_cache.pop(board)
        else:            
            normalized_board, transformations = normalize_board(board)
        self.transformation_cache[board] = (normalized_board, transformations)
        return normalized_board, transformations

