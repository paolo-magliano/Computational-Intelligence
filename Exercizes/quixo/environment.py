import numpy as np
import random
from copy import deepcopy


from constants import *
from game import *
from player import *
from utils import *

class Environment(object):
    def __init__(self, env_player: Player,  env_player_id: int = None) -> None:
        self.game = Game()
        self.env_player = env_player
        self.env_player_id = env_player_id if env_player_id else random.choice([X, O])
        self.mode = 'fixed' if env_player_id else 'random'
        self.transformations = []

    def reset(self) -> tuple[Game, int, bool]:
        '''Returns the initial state, the reward and if the game is over'''
        self.game = Game()
        self.transformations = []
        if self.mode == 'random':
            self.env_player_id = random.choice([X, O])
        
        if self.env_player_id == X:
            ok = False
            while not ok:
                from_pos, slide = self.env_player.make_move(self.game, self.env_player_id)
                ok = self.game.move(from_pos, slide, self.env_player_id)
                if not ok and type(self.env_player) == DQNPlayer:
                    self.env_player.track_invalid_move(self.game)

        return deepcopy(self.game), False

    def step(self, action: tuple[tuple[int, int], Move]) -> tuple[Game, int, bool]:
        '''Returns the next state, the reward and if the game is over'''
        from_pos, slide = action

        ok = self.game.move(from_pos, slide, X + O - self.env_player_id)
        if not ok:
            return deepcopy(self.game), INVALID_MOVE_REWARD, True
        
        winner = self.game.check_winner()
        if winner != EMPTY:
            return deepcopy(self.game), LOSE_REWARD if winner == self.env_player_id else WIN_REWARD, True
        
        ok = False
        while not ok:
            from_pos, slide = self.env_player.make_move(self.game, self.env_player_id)
            ok = self.game.move(from_pos, slide, self.env_player_id)
            if not ok and type(self.env_player) == DQNPlayer:
                self.env_player.track_invalid_move(self.game)
        winner = self.game.check_winner()

        if winner != EMPTY:
            return deepcopy(self.game), LOSE_REWARD if winner == self.env_player_id else WIN_REWARD, True
        
        return deepcopy(self.game), MOVE_REWARD, False
