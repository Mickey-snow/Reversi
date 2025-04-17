import numpy as np
import pytest
from core.GreedyBot import GreedyBot
from othello.env import OthelloEnv


@pytest.fixture
def bot():
    return GreedyBot()


def test_predict_no_moves_returns_none(bot):
    # empty board → no valid moves
    env = OthelloEnv()
    env.board = np.zeros((8, 8), dtype=int)
    move = bot.predict(env)
    assert move is None


def test_predict_initial_board_returns_first_valid(bot):
    # initial Othello setup: valid_moves = [(2,3),(3,2),(4,5),(5,4)]
    env = OthelloEnv()
    env.reset()
    move = bot.predict(env)
    # valid_moves is row-major; first is (2,3)
    assert move == (2, 3), env.render()


def test_predict_picks_move_with_max_flips(bot):
    env = OthelloEnv()

    # Construct a board with exactly two valid moves for white (player=1):
    # - Move A at (4,2): flips two blacks horizontally
    # - Move B at (2,5): flips one black vertically
    env.board = np.zeros((8, 8), dtype=int)
    # horizontal line: (4,3),(4,4) are black; (4,5) is white
    env.board[4, 3] = -1
    env.board[4, 4] = -1
    env.board[4, 5] = 1
    # vertical line: (3,5) is black; (4,5) already white
    env.board[3, 5] = -1

    # supposing current player is white
    env.current_player = 1
    move = bot.predict(env)
    # should pick the one flipping two pieces → (4,2)
    assert move == (4, 2), env.render()


def test_predict_does_not_mutate_input_board(bot):
    env = OthelloEnv()
    env.reset()
    board = env.board
    before = board.copy()
    _ = bot.predict(env)

    assert np.array_equal(board, before)
