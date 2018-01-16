"""
Optimization environment, a function, which is feed-forward neural network.
The neural network's input has two parts, each of k-length-and-d-activation.
The first part is x_o, and the second part is x_p. An optimizer can only optimize on x_p.
Moreover, the state takes time_step into account.
This environment simulates f(x_p, x_o, A_p, A_o) where x_o is varied.
"""
import numpy
import time
import math
from scipy.special import expit
import tensorflow as tf


MONTE_CARLO_ITERATIONS = 20000     # use monte carlo samples to determine max and min
COEF_SEED = 1234      # seed for coefficient generation
n_hidden_func = 100   # number of hidden units in the black-box function


class Environment:
    def __init__(self, k, d):
        self.k = k
        self.d = d
        self.func_generate()
        self.reset()

    def func_generate(self):
        # back up original random seed
        s = numpy.random.get_state()
        # set seed for nn coef
        numpy.random.seed(COEF_SEED)
        # input has two parts: x_o, x_p
        self.w1 = numpy.random.randn(self.k * 2, n_hidden_func) * 0.1
        self.b1 = numpy.random.randn(n_hidden_func) * 0.1
        self.w2 = numpy.random.randn(n_hidden_func, 1) * 0.1
        self.b2 = numpy.random.randn(1) * 0.1
        # restore random seed for other randomness
        numpy.random.set_state(s)

    def monte_carlo(self, logger=None):
        """ Use monte carlo to find the max value """
        min_val = 9e16
        max_val = -9e16

        start_time = time.time()
        for i in range(MONTE_CARLO_ITERATIONS):
            x_o = self.cur_state[:self.k]

            random_xp = numpy.zeros(self.k + 1)  # state + step
            one_idx = numpy.random.choice(self.k, self.d, replace=False)
            random_xp[one_idx] = 1

            random_state = numpy.hstack((x_o, random_xp))
            random_state_output = self.output(random_state)

            if random_state_output < min_val:
                min_val = random_state_output
                min_state = random_state
            if random_state_output > max_val:
                max_val = random_state_output
                max_state = random_state

        duration = time.time() - start_time

        return self.still(max_val), max_state, self.still(min_val), min_state, duration

    def reset(self):
        random_xo = numpy.zeros(self.k)
        one_idx = numpy.random.choice(self.k, self.d, replace=False)
        random_xo[one_idx] = 1

        random_xp = numpy.zeros(self.k + 1)  # the last component is step
        one_idx = numpy.random.choice(self.k, self.d, replace=False)
        random_xp[one_idx] = 1

        self.cur_state = numpy.hstack((random_xo, random_xp))
        return self.cur_state.copy()

    @staticmethod
    def sigmoid(x):
        """
        This python native function is faster than scipy.special.expit for single input.
        See: https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python/
        """
        try:
            res = 1 / (1 + math.exp(-x))
        except OverflowError:
            res = 0.0
        return res

    @staticmethod
    def distill(out):
        """ we did experiments and find distilled rewards help converge faster """
        return numpy.exp(out * 10)
        # return out

    @staticmethod
    def still(distill_out):
        return numpy.log(distill_out) / 10.
        # return distill_out

    def output(self, state):
        # t1 = time.time()
        assert len(state.shape) == 1 and state.shape[0] == 2 * self.k + 1
        out = numpy.dot(
                        self.relu(
                                  numpy.dot(state[:-1], self.w1)
                                  + self.b1
                                 ),
                        self.w2) + self.b2
        out = self.sigmoid(out[0])
        out = self.distill(out)
        # t2 = time.time()
        # print('step output time', t2 - t1)
        return out

    def outputs(self, states):
        assert len(states.shape) == 2 and states.shape[1] == 2 * self.k + 1
        outs = numpy.dot(
                         self.relu(
                                   numpy.dot(states[:, :-1], self.w1)
                                   + self.b1
                                  ),
                         self.w2) + self.b2
        outs = expit(outs)  # sigmoid function for vectors
        outs = self.distill(outs)
        return outs

    def relu(self, x):
        return x * (x > 0)

    def all_possible_next_state_action(self, state_and_step):
        assert len(state_and_step.shape) == 1 and state_and_step.shape[0] == 2 * self.k + 1

        # x_p and step
        state, step = state_and_step[self.k:-1], state_and_step[-1]
        # action format (idx_to_remove, idx_to_add)
        zero_idx = numpy.where(state == 0)[0]
        one_idx = numpy.where(state == 1)[0]
        next_state_template = state_and_step.copy().reshape(1, -1)
        next_state_template[0, -1] = step + 1
        next_states = numpy.repeat(next_state_template,
                                   repeats=len(zero_idx) * len(one_idx) + 1, axis=0)

        # pythonic implementation (faster):
        next_actions = []
        action_idx = 0
        for zi in zero_idx:
            for oi in one_idx:
                next_states[action_idx, self.k + oi] = 0
                next_states[action_idx, self.k + zi] = 1
                next_actions.append((self.k + oi, self.k + zi))
                action_idx += 1
        # the last row of next_states means don't change any card
        next_actions.append((self.k + zi, self.k + oi))

        # numpy implementation:
        # row_idx = numpy.arange(len(zero_idx) * len(one_idx))
        # # first row: one_idx to change to 0, second row: zero idx to change to 1
        # col_idx = numpy.array(numpy.meshgrid(one_idx, zero_idx)).reshape(2, -1)
        # next_states[row_idx, col_idx[0]] = 0
        # next_states[row_idx, col_idx[1]] = 1
        #
        # no_change_action = numpy.array([zero_idx[0], one_idx[0]]).reshape(1, -1)
        # next_actions = numpy.repeat(no_change_action,
        #                             repeats=len(zero_idx) * len(one_idx) + 1, axis=0)
        # next_actions[row_idx, 0] = col_idx[0]
        # next_actions[row_idx, 1] = col_idx[1]
        # the last row of next_states means don't change any card
        return next_states, next_actions

    def step(self, action):
        """ step an action on self.cur_state """
        # action format(idx_to_remove, idx_to_add)
        idx_to_remove, idx_to_add = action[0], action[1]
        self.cur_state[idx_to_remove] = 0
        self.cur_state[idx_to_add] = 1
        self.cur_state[-1] += 1    # increase the step
        new_out = self.output(self.cur_state)
        reward = new_out
        return self.cur_state.copy(), reward

    def all_possible_next_states(self, state_and_step):
        assert len(state_and_step.shape) == 1 and state_and_step.shape[0] == 2 * self.k + 1

        state, step = state_and_step[self.k:-1], state_and_step[-1]
        zero_idx = numpy.where(state == 0)[0]
        one_idx = numpy.where(state == 1)[0]
        next_state_template = state_and_step.copy().reshape(1, -1)
        next_state_template[0, -1] = step + 1
        next_states = numpy.repeat(next_state_template,
                                   repeats=len(zero_idx) * len(one_idx) + 1, axis=0)

        # pythonic implementation (faster):
        action_idx = 0
        for zi in zero_idx:
            for oi in one_idx:
                next_states[action_idx, self.k + oi] = 0
                next_states[action_idx, self.k + zi] = 1
                action_idx += 1

        # numpy implementation:
        # row_idx = numpy.arange(len(zero_idx) * len(one_idx))
        # # first row: one_idx to change to 0, second row: zero idx to change to 1
        # col_idx = numpy.array(numpy.meshgrid(one_idx, zero_idx)).reshape(2, -1)
        # next_states[row_idx, col_idx[0]] = 0
        # next_states[row_idx, col_idx[1]] = 1
        # the last row of next_states means don't change any card
        return next_states

    def step_state(self, state_and_step, action):
        """ step an action on state_and_step """
        idx_to_remove, idx_to_add = action[0], action[1]
        state_and_step = state_and_step.copy()
        state_and_step[idx_to_remove] = 0
        state_and_step[idx_to_add] = 1
        state_and_step[-1] += 1  # increase the step
        return state_and_step

    def test(self, TRIAL_SIZE, RANDOM_SEED, learn_step_counter, cpu_time, env_name, rl_model):
        cur_state = self.reset()
        duration_rl = time.time()

        for i in range(TRIAL_SIZE):
            next_possible_states, next_possible_actions = self.all_possible_next_state_action(cur_state)
            action, q_val = rl_model.choose_action(cur_state, next_possible_states, next_possible_actions,
                                                   epsilon_greedy=False)
            cur_state, reward = self.step(action)
            test_output = self.output(cur_state)
            test_output = self.still(test_output)
            print('TEST  :{}:output: {:.5f}, at {}, qval: {:.5f}, reward {:.5f}'.
                  format(i, test_output, cur_state, q_val, reward))

        duration_rl = time.time() - duration_rl

        max_val_mc, max_state_mc, _, _, duration_mc = self.monte_carlo()
        rl_model.logger.log_test(output_mc=max_val_mc, state_mc=max_state_mc, duration_mc=duration_mc,
                                 output_rl=test_output, state_rl=cur_state, duration_rl=duration_rl,
                                 learn_step_counter=learn_step_counter, cpu_time=cpu_time)

        rl_model.tb_write(tags=['Prioritized={0}, gamma={1}, seed={2}, env={3}/Ending Output (RL-MC)'.
                          format(rl_model.prioritized, rl_model.gamma, RANDOM_SEED, env_name),
                                'Prioritized={0}, gamma={1}, seed={2}, env={3}/Ending Output (RL)'.
                          format(rl_model.prioritized, rl_model.gamma, RANDOM_SEED, env_name),
                                ],
                          values=[test_output - max_val_mc, test_output],
                          step=learn_step_counter)


