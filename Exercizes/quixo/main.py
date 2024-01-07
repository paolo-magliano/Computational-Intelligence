import random
import time

from constants import *
from transformable_game import *
from game import *
from utils import *
from player import *
from environment import *
from transformation import *


def game(env: Environment, player: DQNPlayer, verbose: bool = False) -> int:
    '''Run a game between the player and the environment'''
    states, actions, rewards = [], [], []
    state, done = env.reset()

    states.append(state)
    if verbose:
        print('Initial state:' + ' ' * 50)
        state.print()

    while not done:
        
        action = player.make_move(env.game, X + O - env.env_player_id)
        actions.append(action)

        if verbose:
            print(f'Move: {action}')

        state, reward, done = env.step(action)
        rewards.append(reward)
        states.append(state)

        if verbose:
            state.print()
            print(f'Reward: {reward}')
            print()

    if MODE == 'train':
        player.update(states, actions, rewards)
    return rewards

if __name__ == '__main__':
    win = 0
    assert all([get_index_from_move(get_move_from_index(i)) == i for i in range(ACTION_SPACE)]), 'Wrong index conversion'

    if MODE == 'train':
        player = DQNPlayer(mode='train', load=False)
        env = Environment(RandomPlayer())
    else:
        player = DQNPlayer(mode='test', load=True)
        env = Environment(RandomPlayer())
    
    start = time.time()
    for i in range(ITERATIONS):
        rewards = game(env, player)
        if rewards[-1] == WIN_REWARD:
            win += 1
        print(f'Game {i} - N turns: {len(rewards)} - Reward: {rewards[-1]} - Win rate: {win * 100 / (i + 1):.2f} %', end='\r')
        # if i % 100 == 0 and i > 0 and MODE == 'train':
        #     print(f'{i//100}/{ITERATIONS//100} Q values of the first move:' + ' ' * 50)
        #     for move, policy, target in zip([get_move_from_index(i) for i in range(ACTION_SPACE)], player.policy_net(torch.tensor(Game()._board).flatten().float()).tolist(), player.target_net(torch.tensor(Game()._board).flatten().float()).tolist()):
        #         print(f'\tMove {move}: {policy:.2f} -> {target:.2f}\t{"Invalid move" if round(target) == -1 else "" }')
        #     print()
    stop = time.time()

    print(f'Win rate: {win * 100 / ITERATIONS:.2f} % - Time: {stop - start:.2e} s - Time/iteratio: {(stop - start)/ITERATIONS:.2f} s' + ' ' * 50)
    if MODE == 'train':
        torch.save(player.target_net.state_dict(), f'{PATH}{MODEL_NAME}')
