import numpy as np
from typing import Union, Callable

from constants import *
from game import *

class TransformableGame(Game):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, transformations: list[(Callable, Union[int, None])]) -> 'TransformableGame':
        '''Returns the transformed game'''
        transformed_game = deepcopy(self)
        for function, arg in transformations:
            if arg is None:
                transformed_game._board = function(transformed_game._board)
            else:
                transformed_game._board = function(transformed_game._board, arg)
        
        return transformed_game
    
    @staticmethod
    def transform_move(from_pos: tuple[int, int], slide: Move, transformations: list[(Callable, Union[int, None])]) -> tuple[tuple[int, int], Move]:
        for function, arg in transformations:
            print(function, arg)
            if arg is None:
                from_pos, slide = function(from_pos, slide)
            else:
                from_pos, slide = function(from_pos, slide, arg)
        
        return from_pos, slide
    
    def normalize(self) -> tuple['TransformableGame', list[(Callable, Union[int, None])]]:
        '''Returns the transformations to apply to the game to obtain the normalized game'''
        game_hash = hash(str(self._board.flatten()))
        normalized_game = deepcopy(self)
        transformations = []

        equivalent_game = deepcopy(self) 

        for i in range(2):
            for j in range(4):
                assert game_hash != hash(str(equivalent_game._board.flatten())) or (game_hash == hash(str(normalized_game._board.flatten())) and np.array_equal(normalized_game._board, equivalent_game._board)), "Hashes are not equal"
                if game_hash > hash(str(equivalent_game._board.flatten())):
                    game_hash = hash(str(equivalent_game._board.flatten())) 
                    transformations = [(np.flip, None)] * i + [(np.rot90, j)] if j > 0 else []
                    normalized_game = deepcopy(equivalent_game)
                equivalent_game = equivalent_game.transform([(np.rot90, 1)])
            equivalent_game = self.transform([(np.flip, None)])

        return normalized_game, transformations
    
    @staticmethod
    def get_move_transformations(transformations: list[(Callable, Union[int, None])]) -> list[(Callable, Union[int, None])]:
        '''Returns the transformations to apply to the move to obtain the normalized move'''
        move_transformations = []
        for function, args in transformations:
            if function == np.flip:
                move_transformations.append((TransformableGame.move_flip, None))
            elif function == np.rot90:
                move_transformations.append((TransformableGame.move_rot90, args))

        return move_transformations
    
    @staticmethod
    def get_inverse_transformation(transformations: list[(Callable, Union[int, None])]) -> list[(Callable, Union[int, None])]:
        '''Returns the inverse transformation of the given transformations'''
        inverse_transformations = []
        for function, args in transformations:
            if args is None:
                inverse_transformations.append((function, None))
            else:
                inverse_transformations.append((function, -args))

        return inverse_transformations
    
    @staticmethod
    def move_rot90(from_pos: tuple[int, int], slide: Move, times: int) -> tuple[tuple[int, int], Move]:
        '''Returns the move after the given number of rotations'''
        for _ in range(abs(times)):
            from_pos = (from_pos[1], N - 1 - from_pos[0]) if times > 0 else (N - 1 - from_pos[1], from_pos[0])
            if slide == Move.TOP:
                slide = Move.LEFT if times > 0 else Move.RIGHT
            elif slide == Move.BOTTOM:
                slide = Move.RIGHT if times > 0 else Move.LEFT
            elif slide == Move.LEFT:
                slide = Move.BOTTOM if times > 0 else Move.TOP
            else:
                slide = Move.TOP if times > 0 else Move.BOTTOM

        return from_pos, slide

    @staticmethod
    def move_flip(from_pos: tuple[int, int], slide: Move) -> tuple[tuple[int, int], Move]:
        '''Returns the move after the given number of rotations'''
        if slide == Move.TOP:
            slide = Move.BOTTOM
        elif slide == Move.BOTTOM:
            slide = Move.TOP
        elif slide == Move.LEFT:
            slide = Move.RIGHT
        else:
            slide = Move.LEFT

        return (from_pos[0], N - 1 - from_pos[1]), slide
