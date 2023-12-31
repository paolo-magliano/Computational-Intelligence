
from copy import deepcopy
import numpy as np
import random

from constants import *
from player import Player

# Rules on PDF

class Game(object):
    def __init__(self, player: int = None) -> None:
        self._board =  np.ones((N, N), dtype=np.int8) * EMPTY 
        self.player = player if player else random.choice([X, O])

    def get_board(self) -> np.ndarray:
        '''
        Returns the board
        '''
        return deepcopy(self._board)

    def get_current_player(self) -> int:
        '''
        Returns the current player
        '''
        return deepcopy(self.current_player_idx)

    def print(self):
        '''Prints the board'''

        for row in self._board:
            for cell in row:
                print(CHARS[cell], end=' ')
            print()

    def check_winner(self) -> int:
        '''Check the winner. Returns the player ID of the winner if any, otherwise EMPTY'''
        # for each row
        for x in range(self._board.shape[0]):
            # if a player has completed an entire row
            if self._board[x, 0] != EMPTY and all(self._board[x, :] == self._board[x, 0]):
                # return the relative id
                return self._board[x, 0]
        # for each column
        for y in range(self._board.shape[1]):
            # if a player has completed an entire column
            if self._board[0, y] != EMPTY and all(self._board[:, y] == self._board[0, y]):
                # return the relative id
                return self._board[0, y]
        # if a player has completed the principal diagonal
        if self._board[0, 0] != EMPTY and all(
            [self._board[x, x]
                for x in range(self._board.shape[0])] == self._board[0, 0]
        ):
            # return the relative id
            return self._board[0, 0]
        # if a player has completed the secondary diagonal
        if self._board[0, -1] != EMPTY and all(
            [self._board[x, -(x + 1)]
             for x in range(self._board.shape[0])] == self._board[0, -1]
        ):
            # return the relative id
            return self._board[0, -1]
        return EMPTY
    
    def play(self, player1: Player, player2: Player) -> int:
        '''Play the game. Returns the winning player'''
        states = []
        rewards = []
        players = {
            X: player1,
            O: player2
        }
        
        winner = EMPTY
        while winner == EMPTY:
            self.player = X + O - self.player 
            ok = False
            while not ok:
                from_pos, slide = players[self.player].make_move(self)
                ok = self.move(from_pos, slide, self.player)
            winner = self.check_winner()
        return winner

    def move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id != X and player_id != O:
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.__take((from_pos[1], from_pos[0]), player_id)
        if acceptable:
            acceptable = self.__slide((from_pos[1], from_pos[0]), slide)
            if not acceptable:
                self._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable

    def __take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        '''Take piece'''
        # acceptable only if in border
        acceptable: bool = (
            # check if it is in the first row
            (from_pos[0] == 0 and from_pos[1] < N)
            # check if it is in the last row
            or (from_pos[0] == N - 1 and from_pos[1] < N)
            # check if it is in the first column
            or (from_pos[1] == 0 and from_pos[0] < N)
            # check if it is in the last column
            or (from_pos[1] == N - 1 and from_pos[0] < N)
            # and check if the piece can be moved by the current player
        ) and (self._board[from_pos] == EMPTY or self._board[from_pos] == player_id)
        if acceptable:
            self._board[from_pos] = player_id
        return acceptable

    def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        '''Slide the other pieces'''
        acceptable = Game.acceptable_slide(from_pos, slide)
        # if it is
        if acceptable:
            # take the piece
            piece = self._board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                self._board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], self._board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    self._board[(i, from_pos[1])] = self._board[(
                        i - 1, from_pos[1])]
                # move the piece up
                self._board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], self._board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    self._board[(i, from_pos[1])] = self._board[(
                        i + 1, from_pos[1])]
                # move the piece down
                self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
        return acceptable
    
    @staticmethod
    def acceptable_slide(from_pos: tuple[int, int], slide: Move) -> bool:
                # define the corners
        SIDES = [(0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == N - 1 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == N - 1 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == SIDES[0] and (
                slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == SIDES[1] and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == SIDES[2] and (
                slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == SIDES[3] and (
                slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right

        return acceptable

