from abc import ABC, abstractmethod
import random
import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from constants import *
from true_game import Player, Game, Move
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

class WinMovePlayer(Player):
    '''Player that makes a winning move if possible, really slow'''

    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        player = X if len([1 for cell in game.get_board() if cell != EMPTY]) % 2 == 0 else O
        for i in range(ACTION_SPACE):
            from_pos, move = get_move_from_index(i)
            next_state = deepcopy(game)
            ok = next_state.move(from_pos, move, player)
            if ok and next_state.check_winner() == player:
                return from_pos, move
        
        return RandomPlayer().make_move(game)

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
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(N * N, MLP_1_HIDDEN_SIZE)
        self.fc2 = nn.Linear(MLP_1_HIDDEN_SIZE, MLP_2_HIDDEN_SIZE)
        self.fc3 = nn.Linear(MLP_2_HIDDEN_SIZE, ACTION_SPACE)
        self.non_linearity = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        '''The network can transform the board in a canonical form using rotarions and reflections'''
        if TRANSFORMATION:
            x, transformations = normalize_board(x)
        
        x = x.flatten().float()
        x = self.non_linearity(self.fc1(x))
        x = self.non_linearity(self.fc2(x))
        x = self.fc3(x)

        '''The q values are transformed back to the original board'''
        if TRANSFORMATION:
            inverse = get_inverse_transformation(transformations)
            move_transformations = get_move_transformations(inverse)
            actions = [get_move_from_index(i) for i in range(len(x))]
            transformed_actions = [transform_move(from_pos, move, move_transformations) for from_pos, move in actions]
            transformed_index = [get_index_from_move(action) for action in transformed_actions]
            x = x.scatter(0, torch.tensor(transformed_index), x)

            # print(f'Original: {transformations}')
            # print(f'Inverse: {inverse}')
            # print(f'Move transformations: {move_transformations}')
            # for action, transformed_action, index in zip(actions, transformed_actions, transformed_index):
            #     print(f'{action} -> {transformed_action} -> {index}')

        return x

class DQNPlayer(Player):
    def __init__(self, mode: str = 'train', path: str = f'{PATH}{MODEL_NAME}') -> None:
        super().__init__()
        '''Attribute about the agent'''
        self.mode = mode
        self.n_steps = 0
        self.previous_games = []  
        self.invalid_moves = []
        self.invalid_game = None

        '''Attributes about the network'''
        self.policy_net = DQN()
        self.target_net = DQN()

        if self.mode == 'test' and os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path))
            self.policy_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()
                  

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        epsilon = EPSILON if EPSILON_MODE == 0 else EPSILON_B / (EPSILON_B + self.n_steps)

        
        if random.random() < epsilon and self.mode == 'train':
            '''Exploration choice, the move is chosen with a probability proportional to the q values and that is not the best one'''
            ok = False

            actions_score = self.policy_net(torch.tensor(game.get_board()))
            actions_score = F.softmax(actions_score, dim=0)
            actions_score[torch.argmax(actions_score)] = 0
            actions_score = actions_score / torch.sum(actions_score)

            '''Get a possible valid move according to the probability distribution of the q values'''
            while not ok:
                action_index = random.choices(range(ACTION_SPACE), weights=actions_score.tolist())[0]
                from_pos, move = get_move_from_index(action_index)
                if self.invalid_game and np.array_equal(self.invalid_game.get_board(), game.get_board()) and (from_pos, move) in self.invalid_moves:
                    ok = False
                else:
                    ok = True

        else:
            '''Exploitation choice, the move is chosen with the highest q value'''  

            '''Get the vector of q values for each move'''
            if self.mode == 'test':
                with torch.no_grad():
                    actions_score = self.policy_net(torch.tensor(game.get_board()))
            else:
                actions_score = self.policy_net(torch.tensor(game.get_board()))

            '''Get a possible valid move with the highest q value'''
            ok = False
            k = 0
            while not ok:
                action_index = torch.topk(actions_score, 1 + k).indices[-1].item()
                from_pos, move = get_move_from_index(action_index)
                if self.invalid_game and np.array_equal(self.invalid_game.get_board(), game.get_board()) and (from_pos, move) in self.invalid_moves:
                    k += 1
                else:
                    ok = True

            '''Print the q values for each move'''
            # for action, score in zip([get_move_from_index(i) for i in range(ACTION_SPACE)], actions_score.tolist()):
            #     print(f'\tMove {action}: {score:.2f}\t{"CHOSEN" if action == (from_pos, move) else ""}')

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
                    state = states[i]
                    action = actions[i]
                    reward = rewards[i]
                    next_state = states[i + 1] if i + 1 < len(states) - 1 else None
                    action_index = get_index_from_move(action)

                    '''Compute the current q values'''
                    q_values = self.policy_net(torch.tensor(state._board))

                    '''Compute the expected q values using the target network'''
                    expected_q_values = q_values.clone()
                    expected_q_values[action_index] = reward + GAMMA * torch.max(self.target_net(torch.tensor(next_state._board))) if next_state else reward
                    
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



