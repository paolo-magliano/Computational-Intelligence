from abc import ABC, abstractmethod
import random
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from constants import *
from game import *
from utils import *
from transformation import *

class Player(ABC):
    def __init__(self) -> None:
        '''You can change this for your player if you need to handle state/have memory'''
        pass

    @abstractmethod
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        '''
        game: the Quixo game. You can use it to override the current game with yours, but everything is evaluated by the main game
        return values: this method shall return a tuple of X,Y positions and a move among TOP, BOTTOM, LEFT and RIGHT
        '''
        pass

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game', player: int) -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, N - 1), random.randint(0, N - 1))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class WinMovePlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game', player: int) -> tuple[tuple[int, int], Move]:
        for i in range(ACTION_SPACE):
            from_pos, move = get_move_from_index(i)
            next_state = deepcopy(game)
            ok = next_state.move(from_pos, move, player)
            if ok and next_state.check_winner() == player:
                return from_pos, move
        
        return RandomPlayer().make_move(game, player)

class HumanPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game', player: int) -> tuple[tuple[int, int], Move]:
        game.print()
        col = int(input('Col: '))
        row = int(input('Row: '))
        move = Move[input('Move: ')]
        return (col, row), move
    
class DQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(N * N, MLP_1_HIDDEN_SIZE)
        self.fc2 = nn.Linear(MLP_1_HIDDEN_SIZE, MLP_2_HIDDEN_SIZE)
        self.fc3 = nn.Linear(MLP_2_HIDDEN_SIZE, ACTION_SPACE)
        self.non_linearity = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x, transformations = normalize_board(x)
        
        x = x.flatten().float()
        x = self.non_linearity(self.fc1(x))
        x = self.non_linearity(self.fc2(x))
        x = self.fc3(x)

        # for action, score in zip([get_move_from_index(i) for i in range(ACTION_SPACE)], x.tolist()):
        #     print(f'\tMove {action}: {score:.2f}\t')

        # inverse = get_inverse_transformation(transformations)

        # move_transformations = get_move_transformations(inverse)

        # actions = [get_move_from_index(i) for i in range(len(x))]

        # transformed_actions = [transform_move(from_pos, move, move_transformations) for from_pos, move in actions]

        # transformed_index = [get_index_from_move(action) for action in transformed_actions]

        # # print(f'Original: {transformations}')
        # # print(f'Inverse: {inverse}')
        # # print(f'Move transformations: {move_transformations}')
        # # for action, transformed_action, index in zip(actions, transformed_actions, transformed_index):
        # #     print(f'{action} -> {transformed_action} -> {index}')
        
        # x = x.scatter(0, torch.tensor(transformed_index), x)

        return x

class DQNPlayer(Player):
    def __init__(self, mode: str = 'train', load: bool = False) -> None:
        super().__init__()
        self.mode = mode

        self.policy_net = DQN()
        self.target_net = DQN()

        if load and os.path.exists(f'{PATH}{MODEL_NAME}'):
            self.policy_net.load_state_dict(torch.load(f'{PATH}{MODEL_NAME}'))
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()
        self.previous_games = []

        if mode == 'test':
            self.policy_net.eval()

    def make_move(self, game: 'Game', player: int) -> tuple[tuple[int, int], Move]:
        if random.random() < EPSILON and self.mode == 'train': #_B / (EPSILON_B + len(self.previous_games))
            actions_score = self.policy_net(torch.tensor(game._board))
            actions_score = F.softmax(actions_score, dim=0)
            actions_score[torch.argmax(actions_score)] = 0
            actions_score = actions_score / torch.sum(actions_score)
            action_index = random.choices(range(ACTION_SPACE), weights=actions_score.tolist())[0]

            from_pos, move = get_move_from_index(action_index)
            
        else:
            if self.mode == 'test':
                with torch.no_grad():
                    actions_score = self.policy_net(torch.tensor(game._board))
            else:
                actions_score = self.policy_net(torch.tensor(game._board))
            action_index = torch.argmax(actions_score).item()
            from_pos, move = get_move_from_index(action_index)

            # for action, score in zip([get_move_from_index(i) for i in range(ACTION_SPACE)], actions_score.tolist()):
            #     print(f'\tMove {action}: {score:.2f}\t{"CHOSEN" if action == (from_pos, move) else ""}')

        return from_pos, move
    
    def update(self, states: list['Game'], actions: list[tuple[tuple[int, int], Move]], rewards: list[float]) -> None:
        self.previous_games.append((states, rewards, actions))
        
        if len(self.previous_games) >= BATCH_SIZE:
            random_games = random.choices(self.previous_games, k=BATCH_SIZE)

            # print("Optimizing", len(random_games), "games")

            for states, rewards, actions in random_games:
                # print(f'Game with {len(states)} steps and reward {rewards[-1]}')
                # print()
                # print(f'States: {len(states)}')
                # for state in states:
                #     print(state._board.flatten())
                    
                # print(f'Actions: {len(actions)} {actions}')
                # print(f'Rewards: {len(rewards)} {rewards}')

                for i in range(len(states) - 1):
                    state = states[i]
                    action = actions[i]
                    reward = rewards[i]
                    next_state = states[i + 1] if i + 1 < len(states) - 1 else None
                    action_index = get_index_from_move(action)

                    # print(f'State: ')
                    # state.print()
                    # print(f'Action: {action}')
                    # print(f'Reward: {reward}')
                    # print(f'Next state: ')
                    # next_state.print() if next_state else print('Terminal state')

                    q_values = self.policy_net(torch.tensor(state._board))

                    # print(f'Q values: {q_values[action_index]}')

                    expected_q_values = q_values.clone()
                    
                    expected_q_values[action_index] = reward + GAMMA * torch.max(self.target_net(torch.tensor(next_state._board))) if next_state else reward

                    # print(f'Expected Q values: {expected_q_values[action_index]}')
                    
                    loss = self.loss_function(q_values, expected_q_values)
                    loss.backward()

                    # input("Press Enter to continue...")
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)
