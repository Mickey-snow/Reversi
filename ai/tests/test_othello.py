import numpy as np
from othello.env import OthelloEnv

##############################
# Tests for OthelloEnv Class
##############################


def test_reset_initial_setup():
    """
    Test that after reset the board is properly initialized with the correct center pieces
    and all other spaces set to 0.
    """
    env = OthelloEnv()
    board = env.board
    bs = env.board_size
    mid = bs // 2

    # Check the four center positions.
    assert board[mid - 1][mid - 1] == 1, "Center top-left should be white (1)"
    assert board[mid][mid] == 1, "Center bottom-right should be white (1)"
    assert board[mid - 1][mid] == -1, "Center top-right should be black (-1)"
    assert board[mid][mid - 1] == -1, "Center bottom-left should be black (-1)"

    # Check all other positions are still 0.
    for i in range(bs):
        for j in range(bs):
            if (i, j) not in [
                (mid - 1, mid - 1),
                (mid, mid),
                (mid - 1, mid),
                (mid, mid - 1),
            ]:
                assert board[i][j] == 0, f"Board position {(i, j)} should be empty (0)"


def test_get_state_current_player():
    """
    Test that get_state returns the correct two-channel representation
    for both current player cases.
    """
    env = OthelloEnv()

    env.current_player = 1
    state = env.get_state()
    # First channel for white pieces, second for black pieces.
    np.testing.assert_array_equal(
        state[0],
        (env.board == 1).astype(np.float32),
        err_msg="State channel 0 should represent the current player's (white) pieces.",
    )
    np.testing.assert_array_equal(
        state[1],
        (env.board == -1).astype(np.float32),
        err_msg="State channel 1 should represent the opponent's (black) pieces.",
    )

    # Now test for the opponent's perspective.
    env.current_player = -1
    state = env.get_state()
    np.testing.assert_array_equal(
        state[0],
        (env.board == -1).astype(np.float32),
        err_msg="For black as current player, channel 0 should represent black pieces.",
    )
    np.testing.assert_array_equal(
        state[1],
        (env.board == 1).astype(np.float32),
        err_msg="For black as current player, channel 1 should represent white pieces.",
    )


def test_valid_moves_initial():
    """
    Test that the valid_moves function returns the correct set of initial moves.
    """
    env = OthelloEnv()

    env.current_player = -1
    valid = env.valid_moves()
    expected_moves = {(2, 3), (3, 2), (5, 4), (4, 5)}
    assert (
        set(valid) == expected_moves
    ), f"Expected valid moves {expected_moves} but got {set(valid)}"

    env.current_player = 1
    valid = env.valid_moves()
    expected_moves = {(2, 4), (3, 5), (4, 2), (5, 3)}
    assert (
        set(valid) == expected_moves
    ), f"Expected valid moves {expected_moves} but got {set(valid)}"


def test_step_flip():
    """
    Test the step function by simulating one move and checking that the correct opponent's
    pieces are flipped.
    """
    env = OthelloEnv()

    env.current_player = 1
    # Choose a known valid move for white (player 1): (2,4)
    valid = env.valid_moves()
    assert (
        2,
        4,
    ) in valid, f"(2,4) should be a valid move in the initial state. Got: {valid}"

    # Execute the move.
    env.step((2, 4))

    # After placing at (2,4), the direction (1,0) should capture opponent's piece at (3,4).
    assert env.board[2][4] == 1, "Piece should be placed at (2,4)"
    assert env.board[3][4] == 1, "Opponent piece at (3,4) should have flipped to white"

    # The rest of the center should remain unchanged:
    # (3,3) should still be white, (4,4) should still be white, and (4,3) remains black.
    assert env.board[3][3] == 1, "Center piece at (3,3) should remain white"
    assert env.board[4][4] == 1, "Center piece at (4,4) should remain white"
    assert env.board[4][3] == -1, "Center piece at (4,3) should remain black"


def test_is_game_over():
    """
    Test that is_game_over returns True when no valid moves exist for either player.
    For simplicity, fill the board with pieces so that there are no empty squares.
    """
    env = OthelloEnv()
    # Fill the board completely with 1's (white pieces)
    env.board = np.ones((env.board_size, env.board_size), dtype=int)
    assert (
        env.is_game_over()
    ), "Game should be over when the board is completely filled."


def test_get_winner():
    """
    Test the get_winner function by setting up scenarios where:
    - White wins.
    - Black wins.
    - The game is a tie.
    """

    env = OthelloEnv()

    # Scenario 1: White wins.
    env.board = np.zeros((env.board_size, env.board_size), dtype=int)
    bs = env.board_size
    # Fill top half with white and bottom half with black.
    env.board[: bs // 2, :] = 1
    env.board[bs // 2 :, :] = -1
    # Now adjust one cell to tip the balance in favor of white.
    env.board[bs // 2, 0] = 1  # white gets one extra piece.
    assert (
        env.get_winner() == 1
    ), "White should win when white pieces outnumber black pieces."

    # Scenario 2: Black wins.
    env.board = np.zeros((env.board_size, env.board_size), dtype=int)
    env.board[: bs // 2, :] = 1
    env.board[bs // 2 :, :] = -1
    # Adjust one cell to tip the balance in favor of black.
    env.board[0, 0] = -1  # black gets one extra piece.
    assert (
        env.get_winner() == -1
    ), "Black should win when black pieces outnumber white pieces."

    # Scenario 3: Tie.
    # Create a perfectly balanced board
    env.board = np.empty((env.board_size, env.board_size), dtype=int)
    env.board[: bs // 2, :] = 1
    env.board[bs // 2 :, :] = -1
    # Here, white and black both have the same number of pieces.
    assert (
        env.get_winner() == 0
    ), "Game should be a tie when white and black have equal pieces."


def test_count():
    env = OthelloEnv()

    assert env.count(1) == 2
    assert env.count(-1) == 2

    env.board[: env.board_size // 2, :] = 1
    assert env.count(1) == 32 + 1
