"""
Optimization environment, a real deck recommendation evaluator f(x_p, x_o, A_p, A_o)
Currently, this environment is assuming two players are playing warriors, 100 games, using
GreedyOptimizeMove AI, 312 total available cards, and 30 deck size.
"""
import environment.env_nn as env_nn
import os
import pickle
import numpy
import subprocess
import os


class Environment(env_nn.Environment):

    def __init__(self, k, d, COEF_SEED=1234, fixed_xo=None):
        assert k == 312 and d == 30
        self.k = k
        self.d = d
        self.COEF_SEED = COEF_SEED
        if fixed_xo is not None:
            self.set_fixed_xo(fixed_xo)
        self.reset()

    def output(self, state, delay=0.0, noise_var=0.0):
        """ output by calling java program """
        assert len(state.shape) == 1 and state.shape[0] == 2 * self.k + 1
        # shadow.jar is under the same directory
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        cmds = ["java", "-jar", os.path.join(cur_dir, "shadow.jar"), "100", "greedymove", "greedymove", "warrior"]
        x_o, x_p = state[:self.k], state[self.k:-1]
        player1_card_idx = ','.join(map(str, numpy.nonzero(x_o)[0].tolist()))
        player2_card_idx = ','.join(map(str, numpy.nonzero(x_p)[0].tolist()))
        cmds.append(player1_card_idx)
        cmds.append(player2_card_idx)
        # print(cmds)
        result = subprocess.run(cmds, stdout=subprocess.PIPE).stdout.decode('utf-8')
        # print(result)
        player2_game_won, player2_game_lost = result.split('\n')[4].split(':')
        player2_game_won, player2_game_lost = float(player2_game_won), float(player2_game_lost)
        out = player2_game_won / (player2_game_lost + player2_game_won)
        # print(player2_game_lost, player2_game_won, out)
        out = self.distill(out)
        return out

    def outputs(self, states):
        raise NotImplementedError