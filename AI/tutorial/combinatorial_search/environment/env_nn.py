"""
Optimization environment, a function, which is feed-forward neural network.
Moreover, the state takes time_step into account.
This environment simulates f(x_p, x_o, A_p, A_o) where x_o is considered as fixed
"""
import numpy
import time
import math
from scipy.special import expit
import environment.env_nn_xo as env_nn_xo
import tensorflow as tf


MONTE_CARLO_ITERATIONS = 20000     # use monte carlo samples to determine max and min
COEF_SEED = 1234      # seed for coefficient generation
n_hidden_func = 100   # number of hidden units in the black-box function


class Environment(env_nn_xo.Environment):

    def reset(self):
        random_xo = numpy.zeros(self.k)
        # fix random_xo
        random_xo[:self.d] = 1

        random_xp = numpy.zeros(self.k + 1)  # the last component is step
        one_idx = numpy.random.choice(self.k, self.d, replace=False)
        random_xp[one_idx] = 1

        self.cur_state = numpy.hstack((random_xo, random_xp))
        return self.cur_state.copy()
