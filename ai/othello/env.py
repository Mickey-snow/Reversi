import numpy as np


class OthelloEnv:
    def __init__(self):
        self.board_size = 8
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        mid = self.board_size // 2
        # Initialize center pieces: 1 = white, -1 = black.
        self.board[mid - 1][mid - 1] = 1
        self.board[mid][mid] = 1
        self.board[mid - 1][mid] = -1
        self.board[mid][mid - 1] = -1
        # Set starting player; convention: black plays first
        self.current_player = -1
        return self.get_state()

    def get_state(self):
        # Two channels: first for current player's pieces, second for opponent's pieces.
        if self.current_player == 1:
            return np.array([self.board == 1, self.board == -1]).astype(np.float32)
        else:
            return np.array([self.board == -1, self.board == 1]).astype(np.float32)

    def valid_moves(self, board=None, player=None):
        if board is None:
            board = self.board
        if player is None:
            player = self.current_player
        valid = []
        opponent = -player
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r][c] != 0:
                    continue
                for dr, dc in directions:
                    i, j = r + dr, c + dc
                    found_opponent = False
                    while 0 <= i < self.board_size and 0 <= j < self.board_size:
                        if board[i][j] == opponent:
                            found_opponent = True
                        elif board[i][j] == player and found_opponent:
                            valid.append((r, c))
                            break
                        else:
                            break
                        i += dr
                        j += dc
                    if (r, c) in valid:
                        break
        return valid

    def step(self, move):
        # Assumes move is valid.
        r, c = move
        self.board[r][c] = self.current_player
        opponent = -self.current_player
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        flips = []
        for dr, dc in directions:
            line = []
            i, j = r + dr, c + dc
            while 0 <= i < self.board_size and 0 <= j < self.board_size:
                if self.board[i][j] == opponent:
                    line.append((i, j))
                elif self.board[i][j] == self.current_player:
                    flips.extend(line)
                    break
                else:
                    break
                i += dr
                j += dc
        for i, j in flips:
            self.board[i][j] = self.current_player
        # Switch player. If the next player has no valid moves then remain with the current player.
        self.current_player *= -1
        if len(self.valid_moves()) == 0:
            self.current_player *= -1
        return self.get_state()

    def is_game_over(self):
        # Game over when neither player has a valid move.
        current_valid = len(self.valid_moves())
        self.current_player *= -1
        opp_valid = len(self.valid_moves())
        self.current_player *= -1
        return current_valid == 0 and opp_valid == 0

    def count(self, player=None):
        if player is None:
            player = self.current_player
        return np.sum(self.board == player)

    def get_winner(self):
        white = np.sum(self.board == 1)
        black = np.sum(self.board == -1)
        if white > black:
            return 1
        elif black > white:
            return -1
        else:
            return 0

    def render(self):
        for r in range(self.board_size):
            line = ""
            for c in range(self.board_size):
                if self.board[r][c] == 1:
                    line += "W "
                elif self.board[r][c] == -1:
                    line += "B "
                else:
                    line += ". "
            print(line)
        print("Current player:", "W" if self.current_player == 1 else "B")
