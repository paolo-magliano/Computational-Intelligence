import torch
from typing import Union, Callable

from constants import *
from game import Move

def transform_board(board: torch.Tensor, transformations: list[(Callable, Union[int, tuple])]) -> torch.Tensor:
    '''Returns the transformed board after applying the given transformations'''
    
    transformed_board = board.clone()

    for function, arg in transformations:
            transformed_board = function(transformed_board, arg)
    
    return transformed_board

def transform_move(from_pos: tuple[int, int], slide: Move, transformations: list[(Callable, Union[int, None])]) -> tuple[tuple[int, int], Move]:
    '''Returns the transformed move after applying the given transformations'''
    
    for function, arg in transformations:
        if arg is None:
            from_pos, slide = function(from_pos, slide)
        else:
            from_pos, slide = function(from_pos, slide, arg)
    
    return from_pos, slide

def normalize_board(board: torch.Tensor) -> tuple[torch.Tensor, list[(Callable, Union[int, None])]]:
    '''Returns the transformations to apply to the game to obtain the normalized board, all the equivalent boards have the same normalized board'''
    
    board_hash = hash(str(board.flatten()))
    normalized_board = board.clone()
    transformations = []

    equivalent_board = board.clone() 

    for i in range(2):
        for j in range(4):
            assert board_hash != hash(str(equivalent_board.flatten())) or (board_hash == hash(str(normalized_board.flatten())) and torch.equal(normalized_board, equivalent_board)), "Hashes are not equal"
            
            if board_hash > hash(str(equivalent_board.flatten())):
                board_hash = hash(str(equivalent_board.flatten())) 
                transformations = [(torch.flip, None)] * i + [(torch.rot90, j)] if j > 0 else []
                normalized_board = equivalent_board.clone()

            equivalent_board = transform_board(equivalent_board, [(torch.rot90, 1)])
        equivalent_board = transform_board(equivalent_board, [(torch.flip, (0,))])

    return normalized_board, transformations

def get_move_transformations(transformations: list[(Callable, Union[int, None])]) -> list[(Callable, Union[int, None])]:
    '''Returns the transformations to apply to the move from the board transformations'''
    
    move_transformations = []
    
    for function, args in transformations:
        if function == torch.flip:
            move_transformations.append((move_flip, None))
        elif function == torch.rot90:
            move_transformations.append((move_rot90, args))

    return move_transformations

def get_inverse_transformation(transformations: list[(Callable, Union[int, None])]) -> list[(Callable, Union[int, None])]:
    '''Returns the inverse transformation of the given transformations'''
    
    inverse_transformations = []
    
    for function, args in transformations:
        if args is None or type(args) != int:
            inverse_transformations.append((function, args))
        else:
            inverse_transformations.append((function, -args))

    return inverse_transformations

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

def move_flip(from_pos: tuple[int, int], slide: Move) -> tuple[tuple[int, int], Move]:
    '''Returns the move after the given number of rotations'''
    
    if slide == Move.TOP:
        slide = Move.BOTTOM
    elif slide == Move.BOTTOM:
        slide = Move.TOP

    return (from_pos[0], N - 1 - from_pos[1]), slide
