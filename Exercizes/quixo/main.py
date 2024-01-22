import random
import time

from constants import *
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
        state, reward, done = None, None, None

        while state is None and reward is None and done is None:
            '''Agent move'''
            action = player.make_move(env.game)

            '''Environment move'''
            state, reward, done = env.step(action)
            if verbose and state is None and reward is None and done is None:
                print(f'Invalid move : {action}')
        
        actions.append(action)
        rewards.append(reward)
        states.append(state)

        if verbose:
            print(f'Move: {action}')
            state.print()
            print(f'Reward: {reward}')
            print()

    if MODE == 'train':
        player.update(states, actions, rewards)
    return rewards

if __name__ == '__main__':
    win = 0
    iteration = ITERATIONS if MODE == 'train' else TEST_ITERATION
    assert all([get_index_from_move(get_move_from_index(i)) == i for i in range(ACTION_SPACE)]), 'Wrong index conversion'

    '''Select the player for the agent'''
    player = DQNPlayer(mode=MODE)
    # if MODE == 'train':
    #     player = DQNPlayer(mode=MODE, load=True, path=f'{PATH}model_{N}_v{VERSION - 1}_{ITERATIONS // 1000}K{"_IS" if INVALID_SPACE else ""}{"_IM" if INVALID_MOVES else ""}{"_T" if TRANSFORMATION else ""}.pth')
    # else:
    #     player = DQNPlayer(mode=MODE)

    '''Select all the players for the environment, also the DQNPlayer can be used'''
    env_player = [RandomPlayer()] + [DQNPlayer(mode='test', path=p) for p in MODEL_PATHS]
    #+ [DQNPlayer(mode='test', path=f'{PATH}model_{N}_v{VERSION - 1 - i}_{ITERATIONS // 1000}K{"_IS" if INVALID_SPACE else ""}{"_IM" if INVALID_MOVES else ""}{"_T" if TRANSFORMATION else ""}.pth') for i in range(VERSION)]

    print(f'Play against: ')
    for p in env_player:
        if type(p) == RandomPlayer:
            print(f'\tRandom player')
        else:
            print(f'\tDQN player {p.path}')

    start = time.time()
    for i in range(iteration):

        env = Environment(random.choice(env_player))

        '''Play a game'''
        rewards = game(env, player)

        '''Print and update the win rate'''
        if rewards[-1] == WIN_REWARD:
            win += 1
        if i % BATCH_SIZE == 0:
            execution_time = time.time() - start
            print(f'Game {i} - N turns: {len(rewards)} - Reward: {rewards[-1]} - Win rate: {win * 100 / (i + 1):.2f} % {f"E: {EPSILON_B / (EPSILON_B + i//BATCH_SIZE):.3f}" if MODE == "train" and EPSILON_MODE == 1 else ""} {int(execution_time//3600)}h {int((execution_time%3600)//60)}m/{int((execution_time*ITERATIONS/(i+1))//3600)}h {int(((execution_time*ITERATIONS/(i+1))%3600)//60)}m', end='\r')
    
    stop = time.time()

    print(f'Win rate: {win * 100 / iteration:.2f} % - Time: {stop - start:.2e} s - Time/iteration: {(stop - start)/iteration:.3f} s' + ' ' * 50)
    
    '''Save the trained model'''
    if MODE == 'train':
        torch.save(player.target_net.state_dict(), f'{PATH}{MODEL_NAME}')
