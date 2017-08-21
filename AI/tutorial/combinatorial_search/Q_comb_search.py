"""
Use Q-learning to maximize a function, which is a linear combination of power of one, two and interaction
"""

import numpy
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
from QLearning import QLearning
import time
from tfboard import TensorboardWriter

sess = tf.Session()
k = 60   # total available card size
d = 30    # deck size
use_prioritized_replay = True
noisy = False         # whether the reward is noisy
# when noisy=True, we should normalize reward between 0 and 1
reward_normalize_std = 800.
reward_normalize_mean = 2200.
gamma = 0.9
# the function to optimize is a linear combination of power of one, two and interaction
kk = k + k + k * (k-1) // 2  # polynomial feature size
n_actions = d * (k-d) + 1    # number of one-card modification
n_hidden = d * 2             # number of hidden units in Qlearning NN
BATCH_SIZE = 64
MEMORY_SIZE = 64000
MEMORY_SIZE_START_LEARNING = 1000
TRIAL_SIZE = d                   # how many card modification allowed
EPISODE_SIZE = 1000              # the size of training episodes
MONTE_CARLO_ITERATIONS = 100     # use monte carlo samples to determine max and min
TEST_PERIOD = 10                 # how many per training episodes to do testing
# np.random.seed(1)
# tf.set_random_seed(1)

RL = QLearning(
    n_features=k, n_actions=n_actions, n_hidden=n_hidden, memory_size=MEMORY_SIZE,
    e_greedy_increment=0.0005, sess=sess, prioritized=use_prioritized_replay,
    reward_decay=gamma, n_total_episode=EPISODE_SIZE, n_mem_size_learn_start=MEMORY_SIZE_START_LEARNING,
    batch_size=BATCH_SIZE
)

tb_writer = TensorboardWriter(folder_name="comb_search_k{0}_d{1}/{2}".format(k, d, time.time()), session=sess)

sess.run(tf.global_variables_initializer())


class Environment():
    def __init__(self, k, d, kk):
        self.k = k
        self.d = d
        self.kk = kk
        self.poly = PolynomialFeatures(2, include_bias=False)
        # generate fixed coefficient
        numpy.random.seed(1234)
        self.coef = 10 * numpy.random.rand(self.kk)
        numpy.random.seed()
        self.monte_carlo()

    def monte_carlo(self):
        """ Use monte carlo to find the max value """
        min_val = 9e16
        max_val = -9e16
        for i in range(MONTE_CARLO_ITERATIONS):
            random_state = numpy.zeros(self.k)
            one_idx = numpy.random.choice(self.k, self.d, replace=False)
            random_state[one_idx] = 1
            random_state_output = self.output(random_state)
            if random_state_output < min_val:
                min_val = random_state_output
                min_state = random_state
            if random_state_output > max_val:
                max_val = random_state_output
                max_state = random_state
        print('coefficient=', self.coef)
        print("monte carlo max: {0} at {1}\nmin: {2} at {3}".format(max_val, max_state, min_val, min_state))

    def reset(self):
        random_state = numpy.zeros(self.k)
        one_idx = numpy.random.choice(self.k, self.d, replace=False)
        random_state[one_idx] = 1
        self.cur_state = random_state
        return self.cur_state.copy()

    def output(self, state):
        # t1 = time.time()
        assert len(state.shape) == 1
        trans_feature = self.poly.fit_transform(state.reshape((1, -1))).flatten()
        out = numpy.dot(self.coef, trans_feature)
        # t2 = time.time()
        # print('step output time', t2 - t1)
        return out

    def outputs(self, states):
        assert len(states.shape) == 2
        trans_features = self.poly.fit_transform(states)
        outs = numpy.dot(self.coef, trans_features.T)
        return outs

    def all_possible_next_state_action(self, state):
        # action format (idx_to_remove, idx_to_add)
        zero_idx = numpy.where(state == 0)[0]
        one_idx = numpy.where(state == 1)[0]
        next_states = numpy.repeat(state.reshape(1, -1), repeats=len(zero_idx) * len(one_idx) + 1, axis=0)
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
        return next_states, next_actions

    def step(self, action):
        old_out = self.output(self.cur_state)
        idx_to_remove, idx_to_add = action[0], action[1]
        self.cur_state[idx_to_remove] = 0
        self.cur_state[idx_to_add] = 1
        new_out = self.output(self.cur_state)
        # reward = new_out - old_out
        # reward = old_out - new_out
        if noisy:
            # generate 0 or 1
            reward = numpy.random.binomial(n=1,
                                           p=(new_out - reward_normalize_mean) / reward_normalize_std)
        else:
            reward = (new_out - reward_normalize_mean) / reward_normalize_std
        return self.cur_state.copy(), reward


env = Environment(k=k, d=d, kk=kk)
total_steps = 0

for i_episode in range(EPISODE_SIZE):
    episode_steps = 0
    cur_state = env.reset()
    for i_epsisode_step in range(TRIAL_SIZE):
        next_possible_states, next_possible_actions = env.all_possible_next_state_action(cur_state)
        action, _ = RL.choose_action(cur_state, next_possible_states, next_possible_actions, epsilon_greedy=True)
        cur_state_, reward = env.step(action)
        RL.store_transition(cur_state, action, reward, cur_state_)
        total_steps += 1
        cur_state = cur_state_
        if total_steps > MEMORY_SIZE_START_LEARNING:
            RL.learn()

    print('episode ', i_episode, ' finished with value', env.output(cur_state), 'cur_epsilon', RL.cur_epsilon())

    if total_steps > MEMORY_SIZE_START_LEARNING and i_episode % TEST_PERIOD == 0:
        cur_state = env.reset()
        for i_episode_test_step in range(TRIAL_SIZE):
            next_possible_states, next_possible_actions = env.all_possible_next_state_action(cur_state)
            action, q_val = RL.choose_action(cur_state, next_possible_states, next_possible_actions, epsilon_greedy=False)
            cur_state, reward = env.step(action)
            test_output = env.output(cur_state)
            print('TEST step {0}, output: {1}, at {2}, qval: {3}, reward {4}'.
                  format(i_episode_test_step, test_output, cur_state, q_val, reward))

        tb_writer.write(tags=['Prioritized={0}, gamma={1}, noisy={2}/Test Ending Output'.
                                format(use_prioritized_replay, gamma, noisy),
                              'Prioritized={0}, gamma={1}, noisy={2}/Test Ending Qvalue'.
                                format(use_prioritized_replay, gamma, noisy),
                              ],
                        values=[test_output,
                                q_val],
                        step=i_episode)





