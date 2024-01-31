import random
from game import Game, Move, Player
from player import DQNPlayer, LastMovePlayer

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class MyPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


if __name__ == '__main__':
    sum = 0

    player1 = LastMovePlayer(DQNPlayer(mode='test', path='models/model_5_v0_2000K_MIX_1024S.pth'))
    player0 = LastMovePlayer(DQNPlayer(mode='test', path='models/model_5_v0_1000K_LM_MIX.pth')
    for i in range(1000):
        g = Game()
        # g.print()
        winner = g.play(player0, player1)
        sum = sum + winner
        # g.print()
        print(f"{i}/1000 Winner: Player {winner}", end="\r")

    print(f"Average: {sum / 1000}" + " " * 50)
