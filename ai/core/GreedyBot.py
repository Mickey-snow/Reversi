from copy import deepcopy

from othello.env import OthelloEnv

from core.IBot import IBot


class GreedyBot(IBot):
    def __init__(self):
        pass

    def predict(self, env: OthelloEnv):
        player = env.current_player
        move, count = None, 0
        for mov in env.valid_moves():
            next_env = deepcopy(env)
            next_env.step(mov)
            cnt = next_env.count(player)

            if cnt > count:
                move, count = mov, cnt
        return move
