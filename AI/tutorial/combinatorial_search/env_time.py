"""
Optimization environment, a function, which is feed-forward neural network.
However, the state takes into account time.
"""
import numpy
import tensorflow as tf


MONTE_CARLO_ITERATIONS = 20000     # use monte carlo samples to determine max and min
COEF_SEED = 1234      # seed for coefficient generation
n_hidden_func = 100   # number of hidden units in the black-box function


class Environment:
    def __init__(self, k, d):
        self.k = k
        self.d = d
        self.func_generate()
        self.monte_carlo()

    def func_generate(self):
        # back up original random seed
        s = numpy.random.get_state()
        # set seed for nn coef
        numpy.random.seed(COEF_SEED)
        self.w1 = numpy.random.randn(self.k, n_hidden_func) * 10
        self.b1 = numpy.random.randn(n_hidden_func) * 10
        self.w2 = numpy.random.randn(n_hidden_func, 1) * 10
        self.b2 = numpy.random.randn(1) * 10
        # restore random seed for other randomness
        numpy.random.set_state(s)

    def monte_carlo(self):
        """ Use monte carlo to find the max value """
        min_val = 9e16
        max_val = -9e16
        for i in range(MONTE_CARLO_ITERATIONS):
            random_state = numpy.zeros(self.k + 1)   # state + step
            one_idx = numpy.random.choice(self.k, self.d, replace=False)
            random_state[one_idx] = 1
            random_state_output = self.output(random_state)
            if random_state_output < min_val:
                min_val = random_state_output
                min_state = random_state
            if random_state_output > max_val:
                max_val = random_state_output
                max_state = random_state
        print("monte carlo max: {0} at {1}\nmin: {2} at {3}".format(max_val, max_state, min_val, min_state))

    def reset(self):
        random_state = numpy.zeros(self.k + 1)  # the last component is step
        one_idx = numpy.random.choice(self.k, self.d, replace=False)
        random_state[one_idx] = 1
        self.cur_state = random_state
        return self.cur_state.copy()

    def output(self, state):
        # t1 = time.time()
        assert len(state.shape) == 1 and state.shape[0] == self.k + 1
        out = numpy.dot(
                        self.relu(
                                  numpy.dot(state[:-1], self.w1)
                                  + self.b1
                                 ),
                        self.w2) + self.b2
        # t2 = time.time()
        # print('step output time', t2 - t1)
        return out[0]

    def outputs(self, states):
        assert len(states.shape) == 2 and states.shape[1] == self.k + 1
        outs = numpy.dot(
                         self.relu(
                                   numpy.dot(states[:, :-1], self.w1)
                                   + self.b1
                                  ),
                         self.w2) + self.b2
        return outs

    def relu(self, x):
        return x * (x > 0)

    def all_possible_next_state_action(self, state_and_step):
        state, step = state_and_step[:-1], state_and_step[-1]
        # action format (idx_to_remove, idx_to_add)
        zero_idx = numpy.where(state == 0)[0]
        one_idx = numpy.where(state == 1)[0]
        next_state_template = state_and_step.copy().reshape(1, -1)
        next_state_template[0, -1] = step + 1
        next_states = numpy.repeat(next_state_template,
                                   repeats=len(zero_idx) * len(one_idx) + 1, axis=0)

        # pythonic implementation (faster)
        next_actions = []
        action_idx = 0
        for zi in zero_idx:
            for oi in one_idx:
                next_states[action_idx, oi] = 0
                next_states[action_idx, zi] = 1
                next_actions.append((oi, zi))
                action_idx += 1
        # the last row of next_states means don't change any card
        next_actions.append((zi, oi))

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
        old_out = self.output(self.cur_state)
        # action format(idx_to_remove, idx_to_add)
        idx_to_remove, idx_to_add = action[0], action[1]
        self.cur_state[idx_to_remove] = 0
        self.cur_state[idx_to_add] = 1
        self.cur_state[-1] += 1    # increase the step
        new_out = self.output(self.cur_state)
        # reward = new_out - old_out
        # reward = old_out - new_out
        # if new_out > 4407:
        #     reward = new_out
        # else:
        #     reward = -1
        reward = new_out
        # reward = numpy.exp(new_out / 100. - 36.)  # distilled reward
        return self.cur_state.copy(), reward

    def all_possible_next_states(self, state_and_step):
        state, step = state_and_step[:-1], state_and_step[-1]
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
                next_states[action_idx, oi] = 0
                next_states[action_idx, zi] = 1
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


