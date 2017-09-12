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
from prioritized_exp import RL_brain
import os
import pickle


class Memory(object):

    def __init__(self, capacity, prioritized, n_features, n_actions,
                 n_total_episode, batch_size, n_mem_size_learn_start,
                 all_possible_next_states_func, step_state_func):
        self.capacity = capacity
        self.prioritized = prioritized
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_total_episode = n_total_episode
        self.batch_size = batch_size
        self.n_mem_size_learn_start = n_mem_size_learn_start
        self.all_possible_next_states = all_possible_next_states_func
        self.step_state = step_state_func

        if self.prioritized:
            self.memory = RL_brain.Memory(self.capacity)
            # # partition_num (==batch_size) * n_mem_size_learn_start should be larger than capacity
            # assert self.batch_size * self.n_mem_size_learn_start >= self.capacity
            # self.memory = rank_based.Experience({'size': self.capacity, 'learn_start': self.n_mem_size_learn_start,
            #                                      'partition_num': self.batch_size, 'batch_size': self.batch_size,
            #                                      'total_step': self.n_total_episode})
        else:
            self.memory = deque(maxlen=self.capacity)

    def store(self, transition):
        if self.prioritized:
            if transition[2] > 4407:
                print('save 4408')
            self.memory.store(transition)
        else:
            if transition[2] > 4407:
                print('save 4408')
            self.memory.append(transition)

    def sample(self, learn_step_counter):
        if self.prioritized:
            return self._prioritized_sample(learn_step_counter)
        else:
            return self._no_prioritized_sample()

    def _prioritized_sample(self, learn_step_counter):
        # samples, is_weights, sample_mem_idxs = self.memory.sample(learn_step_counter)
        tree_idx, samples, is_weights = self.memory.sample(self.batch_size)

        qsa_feature = numpy.zeros((self.batch_size, self.n_features))
        qsa_next_feature = numpy.zeros((self.batch_size, self.n_actions, self.n_features))
        rewards = numpy.zeros(self.batch_size)
        terminal_weights = numpy.ones(self.batch_size)
        # is_weights = numpy.array(is_weights)
        is_weights = numpy.squeeze(is_weights)

        for i, (state, action, reward, next_state, terminal) in enumerate(samples):
            rewards[i] = reward
            if reward > 4407:
                print('sample 4408')
            terminal_weights[i] = 0 if terminal else 1
            qsa_feature[i] = self.step_state(state, action)
            qsa_next_feature[i] = self.all_possible_next_states(next_state)

        return qsa_feature, qsa_next_feature, rewards, terminal_weights, is_weights, tree_idx

    def _no_prioritized_sample(self):
        assert self.batch_size <= len(self.memory)
        sample_mem_idxs = numpy.random.choice(len(self.memory), self.batch_size, replace=False)

        qsa_feature = numpy.zeros((self.batch_size, self.n_features))
        qsa_next_feature = numpy.zeros((self.batch_size, self.n_actions, self.n_features))
        rewards = numpy.zeros(self.batch_size)
        terminal_weights = numpy.ones(self.batch_size)
        # every sample is equally important in non-prioritized sampling
        is_weights = numpy.ones(self.batch_size)

        for i, mem_idx in enumerate(sample_mem_idxs):
            state, action, reward, next_state, terminal = self.memory[mem_idx]
            if reward > 4407:
                print('sample 4408')
            rewards[i] = reward
            terminal_weights[i] = 0 if terminal else 1
            qsa_feature[i] = self.step_state(state, action)
            qsa_next_feature[i] = self.all_possible_next_states(next_state)

        return qsa_feature, qsa_next_feature, rewards, terminal_weights, is_weights, sample_mem_idxs

    def update_priority(self, e_ids, abs_errors):
        assert self.prioritized
        self.memory.batch_update(e_ids, abs_errors)
        # self.memory.update_priority(e_ids, abs_errors)

    @property
    def size(self):
        if self.prioritized:
            return sum(map(lambda x: 1 if x else 0, self.memory.data)) # count how many not-none element
        else:
            return len(self.memory)


class QLearning:

    def __init__(
            self,
            n_features,
            n_actions,
            n_hidden,
            n_total_episode,
            n_mem_size_learn_start,
            save_and_load_path,
            all_possible_next_states_func,
            step_state_func,
            load,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.8,
            replace_target_iter=500,
            save_model_iter=3000,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=0.0001,
            prioritized=True,
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.n_total_episode = n_total_episode
        self.n_mem_size_learn_start = n_mem_size_learn_start
        self.save_and_load_path = save_and_load_path
        self.load = load

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.save_model_iter = save_model_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.prioritized = prioritized  # decide to use double q or not

        self.learn_step_counter = 0

        # create a graph for model variables and session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        if not load:
            with self.graph.as_default():
                self._build_net()
                self.sess.run(tf.global_variables_initializer())
            self.memory = Memory(prioritized=self.prioritized, capacity=memory_size,
                                 n_features=self.n_features, n_actions=self.n_actions,
                                 n_total_episode=self.n_total_episode, batch_size=self.batch_size,
                                 n_mem_size_learn_start=self.n_mem_size_learn_start,
                                 all_possible_next_states_func=all_possible_next_states_func,
                                 step_state_func=step_state_func)
        else:
            self.load_model()

    def save_model(self):
        with self.graph.as_default():
            saver = tf.train.Saver()
            path = saver.save(self.sess, self.save_and_load_path)
            print('save model to', path)
            with open(self.save_and_load_path + '_memory.pickle', 'wb') as f:
                pickle.dump(self.memory, f, protocol=-1)   # -1: highest protocol

    def load_model(self):
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(self.save_and_load_path + '.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(self.save_and_load_path)))
            # placeholders
            self.s = self.graph.get_tensor_by_name('s:0')  # Q(s,a) feature
            self.s_ = self.graph.get_tensor_by_name('s_:0')  # Q(s',a') feature
            self.rewards = self.graph.get_tensor_by_name('reward:0')   # reward
            self.terminal_weights = self.graph.get_tensor_by_name('terminal:0')  # terminal
            # variables
            self.q_eval = self.graph.get_tensor_by_name('eval_net/out:0')
            self.eval_w1 = self.graph.get_tensor_by_name('eval_net/l1/w1:0')
            self.eval_b1 = self.graph.get_tensor_by_name('eval_net/l1/b1:0')
            self.eval_w2 = self.graph.get_tensor_by_name('eval_net/l2/w2:0')
            self.eval_b2 = self.graph.get_tensor_by_name('eval_net/l2/b2:0')
            self.q_next = self.graph.get_tensor_by_name('target_net/out:0')
            self.target_w1 = self.graph.get_tensor_by_name('target_net/l1/w1:0')
            self.target_b1 = self.graph.get_tensor_by_name('target_net/l1/b1:0')
            self.target_w2 = self.graph.get_tensor_by_name('target_net/l2/w2:0')
            self.target_b2 = self.graph.get_tensor_by_name('target_net/l2/b2:0')
            self.q_target = self.graph.get_tensor_by_name("q_target:0")
            self.is_weights = self.graph.get_tensor_by_name("is_weights:0")
            self.loss = self.graph.get_tensor_by_name("loss:0")
            self.abs_errors = self.graph.get_tensor_by_name("abs_errors:0")
            # operations
            self.train_op = self.graph.get_operation_by_name('train_op')

        with open(self.save_and_load_path + '_memory.pickle', 'rb') as f:
            self.memory = pickle.load(f)  # -1: highest protocol

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
            return tf.squeeze(out, name='out'), w1, b1, w2, b2

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
            return tf.squeeze(out, name='out'), w1, b1, w2, b2

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # Q(s,a) feature
        self.s_ = tf.placeholder(tf.float32, [None, self.n_actions, self.n_features], name='s_') # Q(s',a') feature
        self.rewards = tf.placeholder(tf.float32, [None], name='reward')  # reward
        self.terminal_weights = tf.placeholder(tf.float32, [None], name='terminal') # terminal

        w_initializer, b_initializer = \
            tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_eval, self.eval_w1, self.eval_b1, self.eval_w2, self.eval_b2 \
                = build_eval_layers(self.s, c_names, w_initializer, b_initializer)

        # ------------------ build target_net ------------------
        # Q(s',a') for all a' feature
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next, self.target_w1, self.target_b1, self.target_w2, self.target_b2 \
                = build_target_layers(self.s_, c_names, w_initializer, b_initializer)

        # ------------------ loss function ----------------------
        # importance sampling weight
        self.q_target = tf.add(self.rewards,
                               self.terminal_weights * (self.gamma * tf.reduce_max(self.q_next, axis=1)),
                               name='q_target')
        # importance sampling weight
        self.is_weights = tf.placeholder(tf.float32, [None], name='is_weights')
        self.loss = tf.reduce_mean(self.is_weights * tf.squared_difference(self.q_target, self.q_eval), name='loss')
        self.abs_errors = tf.abs(self.q_target - self.q_eval, name='abs_errors')
        self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, name='train_op')

    def store_transition(self, s, a, r, s_, terminal):
        # transition is a tuple (current_state, action, reward, next_state, whether_terminal)
        self.memory.store((s, a, r, s_, terminal))

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
        with self.graph.as_default():
            t_params = tf.get_collection('target_net_params')
            e_params = tf.get_collection('eval_net_params')
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
            print('target_params_replaced')

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            # if self.prioritized:
            #     print('4408 reward samples in total:', numpy.sum([d[2] > 4407 for d in self.memory.memory.tree.data]))

        if self.learn_step_counter % self.save_model_iter == 0:
            self.save_model()

        qsa_feature, qsa_next_feature, rewards, terminal_weights, is_weights, exp_idx \
            = self.memory.sample(self.learn_step_counter)

        # q_target, q_next, q_eval = self.sess.run(
        #     [self.q_target, self.q_next, self.q_eval],
        #     feed_dict={self.s: qsa_feature,
        #                self.s_: qsa_next_feature,
        #                self.reward: rewards})

        _, loss, abs_errors = self.sess.run([self.train_op, self.loss, self.abs_errors],
                                            feed_dict={self.s: qsa_feature,
                                                       self.s_: qsa_next_feature,
                                                       self.rewards: rewards,
                                                       self.terminal_weights: terminal_weights,
                                                       self.is_weights: is_weights})

        if self.prioritized:
            self.memory.update_priority(exp_idx, abs_errors)

        self.epsilon = self.cur_epsilon()
        self.learn_step_counter += 1

    def cur_epsilon(self):
        return self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max