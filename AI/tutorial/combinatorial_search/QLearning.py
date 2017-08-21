"""
Neural network approximation Q-Learning with prioritized experience replay
experience replay implementation taken from: https://github.com/Damcy/prioritized-experience-replay
"""
import time
import numpy as np
import tensorflow as tf
import numpy
from collections import deque
import prioritized_exp.rank_based as rank_based


class Memory(object):

    def __init__(self, capacity, prioritized, n_features, n_actions,
                 n_total_episode, batch_size, n_mem_size_learn_start):
        self.capacity = capacity
        self.prioritized = prioritized
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_total_episode = n_total_episode
        self.batch_size = batch_size
        self.n_mem_size_learn_start = n_mem_size_learn_start

        if self.prioritized:
            # partition_num (==batch_size) * n_mem_size_learn_start should be larger than capacity
            assert self.batch_size * self.n_mem_size_learn_start >= self.capacity
            self.memory = rank_based.Experience({'size': self.capacity, 'learn_start': self.n_mem_size_learn_start,
                                                 'partition_num': self.batch_size, 'batch_size': self.batch_size,
                                                 'total_step': self.n_total_episode})
        else:
            self.memory = deque(maxlen=self.capacity)

    def store(self, transition):
        if self.prioritized:
            self.memory.store(transition)
        else:
            self.memory.append(transition)

    def sample(self, learn_step_counter):
        if self.prioritized:
            return self._prioritized_sample(learn_step_counter)
        else:
            return self._no_prioritized_sample()

    def _prioritized_sample(self, learn_step_counter):
        samples, is_weights, sample_mem_idxs = self.memory.sample(learn_step_counter)

        qsa_feature = numpy.zeros((self.batch_size, self.n_features))
        qsa_next_feature = numpy.zeros((self.batch_size, self.n_actions, self.n_features))
        rewards = numpy.zeros(self.batch_size)
        is_weights = numpy.array(is_weights)

        for i, (state, action, reward, next_state, terminal) in enumerate(samples):
            rewards[i] = reward
            qsa_feature[i] = self.apply(state, action)
            qsa_next_feature[i] = self.all_possible_next_states(next_state)

        return qsa_feature, qsa_next_feature, rewards, is_weights, sample_mem_idxs

    def _no_prioritized_sample(self):
        assert self.batch_size <= len(self.memory)
        sample_mem_idxs = numpy.random.choice(len(self.memory), self.batch_size, replace=False)

        qsa_feature = numpy.zeros((self.batch_size, self.n_features))
        qsa_next_feature = numpy.zeros((self.batch_size, self.n_actions, self.n_features))
        rewards = numpy.zeros(self.batch_size)
        # every sample is equally important in non-prioritized sampling
        is_weights = numpy.ones(self.batch_size)

        for i, mem_idx in enumerate(sample_mem_idxs):
            state, action, reward, next_state, terminal = self.memory[mem_idx]
            rewards[i] = reward
            qsa_feature[i] = self.apply(state, action)
            qsa_next_feature[i] = self.all_possible_next_states(next_state)

        return qsa_feature, qsa_next_feature, rewards, is_weights, sample_mem_idxs

    def apply(self, state, action):
        state_copy = state.copy()
        idx_to_remove, idx_to_add = action[0], action[1]
        state_copy[idx_to_remove] = 0
        state_copy[idx_to_add] = 1
        return state_copy

    def all_possible_next_states(self, state):
        # action format (idx_to_remove, idx_to_add)
        zero_idx = numpy.where(state == 0)[0]
        one_idx = numpy.where(state == 1)[0]
        assert len(zero_idx) * len(one_idx) + 1 == self.n_actions
        # repeat a (1, self.n_features) 2d array vertically self.n_actions times
        next_states = numpy.repeat(state.reshape((1, -1)), repeats=self.n_actions, axis=0)
        action_idx = 0
        for zi in zero_idx:
            for oi in one_idx:
                next_states[action_idx, oi] = 0
                next_states[action_idx, zi] = 1
                action_idx += 1
        # the last row of next_states means don't change any card
        return next_states

    def update_priority(self, e_id, abs_error):
        assert self.prioritized
        self.memory.update_priority(e_id, abs_error)

    def rebalance(self):
        assert self.prioritized
        self.memory.rebalance()


class QLearning:
    def __init__(
            self,
            n_features,
            n_actions,
            n_total_episode,
            n_mem_size_learn_start,
            n_hidden=20,
            learning_rate=0.005,
            reward_decay=1.0,
            e_greedy=0.8,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            prioritized=True,
            sess=None,
    ):
        self.n_features = n_features
        self.n_total_episode = n_total_episode
        self.n_mem_size_learn_start = n_mem_size_learn_start
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized  # decide to use double q or not

        self.learn_step_counter = 0

        self._build_net()

        self.memory = Memory(prioritized=self.prioritized, capacity=memory_size,
                             n_features=self.n_features, n_actions=self.n_actions,
                             n_total_episode=self.n_total_episode, batch_size=self.batch_size,
                             n_mem_size_learn_start=self.n_mem_size_learn_start)

        self.sess = sess

    def _build_net(self):
        def build_eval_layers(s, c_names, w_initializer, b_initializer):
            # s is Q(s,a) feature, shape: (n_sample, n_features)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, self.n_hidden], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [self.n_hidden], initializer=b_initializer, collections=c_names)
                # l1 shape: (n_sample, n_hidden)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [self.n_hidden, 1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1], initializer=b_initializer, collections=c_names)
                # out shape: (n_sample, 1)
                out = tf.matmul(l1, w2) + b2
            return tf.squeeze(out)

        def build_target_layers(s_, c_names, w_initializer, b_initializer):
            # s_ Q(s',a') for all a' feature, shape: (n_sample, n_actions, n_features)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, self.n_hidden], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [self.n_hidden], initializer=b_initializer, collections=c_names)
                # l1 shape: shape: (n_sample, n_actions, n_hidden)
                l1 = tf.nn.relu(tf.einsum('ijk,kh->ijh', s_, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [self.n_hidden, 1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1], initializer=b_initializer, collections=c_names)
                out = tf.einsum('ijh,ho->ijo', l1, w2) + b2
            return tf.squeeze(out)

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # Q(s,a) feature
        self.s_ = tf.placeholder(tf.float32, [None, self.n_actions, self.n_features], name='s_')
        self.reward = tf.placeholder(tf.float32, [None], name='reward')  # reward

        w_initializer, b_initializer = \
            tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_eval = build_eval_layers(self.s, c_names, w_initializer, b_initializer)

        # ------------------ build target_net ------------------
        # Q(s',a') for all a' feature
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_target_layers(self.s_, c_names, w_initializer, b_initializer)

        # ------------------ loss function ----------------------
        # importance sampling weight
        self.q_target = self.reward + self.gamma * tf.reduce_max(self.q_next, axis=1)
        self.is_weights = tf.placeholder(tf.float32, [None], name='IS_weights')
        with tf.variable_scope('train'):
            self.loss = tf.reduce_mean(self.is_weights * tf.squared_difference(self.q_target, self.q_eval))
            self.abs_errors = tf.abs(self.q_target - self.q_eval)
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        # transition is a tuple (current_state, action, reward, next_state, whether_terminal)
        self.memory.store((s, a, r, s_, False))

    def choose_action(self, state, next_possible_states, next_possbile_actions, epsilon_greedy=True):
        pred_q_values = self.sess.run(self.q_eval, feed_dict={self.s: next_possible_states}).flatten()
        if not epsilon_greedy or np.random.uniform() < self.epsilon:
            action_idx = np.argmax(pred_q_values)
        else:
            action_idx = np.random.choice(numpy.arange(len(next_possbile_actions)))
        action = next_possbile_actions[action_idx]
        pred_q_value = pred_q_values[action_idx]
        return action, pred_q_value

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('target_params_replaced')
            # btw, rebalance the prioritized memory
            if self.prioritized:
                self.memory.rebalance()
            print('prioritized memory rebalanced')

        qsa_feature, qsa_next_feature, rewards, is_weights, exp_idx = self.memory.sample(self.learn_step_counter)

        # q_target, q_next, q_eval = self.sess.run(
        #     [self.q_target, self.q_next, self.q_eval],
        #     feed_dict={self.s: qsa_feature,
        #                self.s_: qsa_next_feature,
        #                self.reward: rewards})

        _, loss, abs_errors = self.sess.run([self._train_op, self.loss, self.abs_errors],
                                            feed_dict={self.s: qsa_feature,
                                                       self.s_: qsa_next_feature,
                                                       self.reward: rewards,
                                                       self.is_weights: is_weights})

        if self.prioritized:
            self.memory.update_priority(exp_idx, abs_errors)

        self.epsilon = self.cur_epsilon()
        self.learn_step_counter += 1

    def cur_epsilon(self):
        return self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max