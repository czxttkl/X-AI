"""
Very similar to env_nn.Environment.
However, it is added with output noise
such that the output behavior is closer to deck evaluation
"""
import environment.env_nn as env_nn
import numpy
import time


class Environment(env_nn.Environment):

    def output(self, state):
        """ output with noise"""
        # WHY noise_var=0.07 BY DEFAULT?
        # based on preliminary tests, random plays have ~7% std
        assert len(state.shape) == 1 and state.shape[0] == 2 * self.k + 1
        out = self.nn(state)
        noise_var = 0.03
        noise = numpy.random.normal(0, noise_var)
        out += noise
        out = min(max(out, 0), 1)  # clip to be [0, 1]
        out = self.distill(out)
        return out

    def outputs(self, states):
        raise NotImplementedError

    def output_noiseless(self, state):
        assert len(state.shape) == 1 and state.shape[0] == 2 * self.k + 1
        out = self.nn(state)
        out = self.distill(out)
        return out
