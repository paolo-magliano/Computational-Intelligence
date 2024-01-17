import numpy as np
import random
from copy import deepcopy

from constants import *
from game_ext import *
from player import *
from utils import *

class Environment(object):
    '''The environment is responsable for manage the game structure, the rewards for the agent at each step and the adversary moves '''

    def __init__(self, env_player: Player,  env_player_id: int = None) -> None:
        self.game = GameExt()
        '''Information about the environment player strategy'''
        self.env_player = env_player
        self.env_player_id = env_player_id if env_player_id else random.choice([X, O])

        self.mode = 'fixed' if env_player_id else 'random'

    def reset(self) -> tuple[Game, int, bool]:
        '''Returns the initial state, the reward and if the game is over'''
        self.game = GameExt()
        if self.mode == 'random':
            self.env_player_id = random.choice([X, O])
        
        '''The case where the environment player is the first to move'''
        if self.env_player_id == X:
            ok = False
            while not ok:
                from_pos, slide = self.env_player.make_move(self.game)
                ok = self.game.move(from_pos, slide, self.env_player_id)

        return deepcopy(self.game), False

    def step(self, action: tuple[tuple[int, int], Move]) -> tuple[Game, int, bool]:
        '''Returns the next state, the reward and if the game is over'''

        '''Agent move'''
        from_pos, slide = action

        ok = self.game.move(from_pos, slide, X + O - self.env_player_id)
        if not ok:
            return None, None, None
        
        '''Check if the game is over'''
        winner = self.game.check_winner()
        if winner != EMPTY:
            return deepcopy(self.game), LOSE_REWARD if winner == self.env_player_id else WIN_REWARD, True
        
        '''Environment move'''
        ok = False
        while not ok:
            from_pos, slide = self.env_player.make_move(self.game)
            ok = self.game.move(from_pos, slide, self.env_player_id)
            
        winner = self.game.check_winner()

        '''Check if the game is over'''
        if winner != EMPTY:
            return deepcopy(self.game), LOSE_REWARD if winner == self.env_player_id else WIN_REWARD, True
        
        return deepcopy(self.game), MOVE_REWARD, False
