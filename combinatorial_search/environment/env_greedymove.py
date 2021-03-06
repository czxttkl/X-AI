"""
Optimization environment, a real deck recommendation evaluator f(x_p, x_o, A_p, A_o)
Currently, this environment is assuming two players are playing warriors, 300 games, using
GreedyOptimizeMove AI, 312 total available cards, and 30 deck size (15 distinct cards,
each having 2 copies).
"""
import environment.env_nn as env_nn
import numpy
import subprocess
import os


class Environment(env_nn.Environment):

    def __init__(self, k, d, COEF_SEED=1234, fixed_xo=None):
        assert k == 312 and d == 15
        self.k = k
        self.d = d
        self.COEF_SEED = COEF_SEED
        if fixed_xo is not None:
            self.set_fixed_xo(fixed_xo)
        self.reset()

    def output(self, state):
        """ output with variance by calling java program 300 games """
        assert len(state.shape) == 1 and state.shape[0] == 2 * self.k + 1
        x_o, x_p = state[:self.k], state[self.k:-1]
        out = self.call_java_program(x_o, x_p, 300)
        # print(player2_game_lost, player2_game_won, out)
        out = self.distill(out)
        return out

    def outputs(self, states):
        raise NotImplementedError

    def call_java_program(self, x_o, x_p, number_of_games):
        """
        Return win rate of deck evaluation
        """
        # shadow.jar is under the same directory
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        cmds = ["java",
                "-jar",
                os.path.join(cur_dir, "shadow.jar"),
                str(number_of_games),
                "greedymove",
                "greedymove",
                "warrior"]
        player1_card_idx = ','.join(map(str, numpy.nonzero(x_o)[0].tolist()))
        player2_card_idx = ','.join(map(str, numpy.nonzero(x_p)[0].tolist()))
        cmds.append(player1_card_idx)
        cmds.append(player2_card_idx)
        # print(cmds)
        result = subprocess.run(cmds, stdout=subprocess.PIPE).stdout.decode('utf-8')
        # print(result)
        player2_game_won, player2_game_lost = result.split('\n')[4].split(':')
        try:
            player2_game_won, player2_game_lost = float(player2_game_won), float(player2_game_lost)
            out = player2_game_won / (player2_game_lost + player2_game_won)
            # print(player2_game_lost, player2_game_won, out)
            assert 0. <= out
            assert out <= 1.
            return out
        except:
            print("Run java program causing problem, return 0.5 instead")
            print(result)
            return 0.5

    def monte_carlo(self,
                    MONTE_CARLO_ITERATIONS=20000,
                    WALL_TIME_LIMIT=9e30):
        random_state = numpy.zeros(2 * self.k + 1)
        return 0, random_state, 0, random_state, 0, 0

    def output_noiseless(self, state):
        """ noiseless output = output by calling java program 300 times """
        return self.output(state)
