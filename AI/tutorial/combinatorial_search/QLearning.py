"""
Neural network approximation Q-Learning with prioritized experience replay
experience replay implementation taken from: https://github.com/Damcy/prioritized-experience-replay
"""
import time
import numpy as np
import tensorflow as tf
import numpy
import os
import glob
import pickle
from tfboard import TensorboardWriter
from prioritized_memory import Memory
from env_time import Environment
import multiprocessing
from logger import Logger


class QLearning:

    def __init__(
            self,
            k,
            d,
            n_features,
            n_actions,
            n_hidden,
            save_and_load_path,
            load,
            tensorboard_path,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.8,
            replace_target_iter=500,
            save_model_iter=5000,
            memory_capacity=10000,
            batch_size=32,
            e_greedy_increment=0.0001,
            prioritized=True,
            planning=False
    ):
        self.env = Environment(k=k, d=d)
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.save_and_load_path = os.path.abspath(save_and_load_path)
        self.load = load
        self.tensorboard_path = tensorboard_path

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.save_model_iter = save_model_iter
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.prioritized = prioritized  # decide to use double q or not
        self.planning = planning  # decide to use planning for additional learning

        # create a logger
        self.path_check(load)
        self.logger = Logger(os.path.dirname(os.path.dirname(self.save_and_load_path)) + '/logger.log')

        # create a graph for model variables and session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        if not load:
            with self.graph.as_default():
                self._build_net()
                self.sess.run(tf.global_variables_initializer())
            self.memory = Memory(prioritized=self.prioritized, capacity=self.memory_capacity,
                                 n_features=self.n_features, n_actions=self.n_actions,
                                 batch_size=self.batch_size, planning=self.planning,
                                 qsa_feature_extractor=self.env.step_state,
                                 qsa_feature_extractor_for_all_acts=self.env.all_possible_next_states)
            self.env.monte_carlo(self.logger)
            self.learn_step_counter = 0
            self.learn_time = 0
            self.collect_sample_step_counter = 0
            self.collect_sample_time = 0
        else:
            self.load_model()

        self.tb_writer = TensorboardWriter(folder_name=self.tensorboard_path, session=self.sess)
        self.memory_lock = multiprocessing.Lock()     # lock for memory modification

    def path_check(self, load):
        save_and_load_path_dir = os.path.dirname(self.save_and_load_path)
        if load:
            assert os.path.exists(save_and_load_path_dir), "model path not exist"
        else:
            os.makedirs(save_and_load_path_dir, exist_ok=True)
            # remove old existing models if any
            files = glob.glob(save_and_load_path_dir + '/*')
            for file in files:
                os.remove(file)

    def save_model(self):
        # save tensorflow
        with self.graph.as_default():
            saver = tf.train.Saver()
            path = saver.save(self.sess, self.save_and_load_path)
        # save memory
        self.memory_lock.acquire()
        with open(self.save_and_load_path + '_memory.pickle', 'wb') as f:
            pickle.dump(self.memory, f, protocol=-1)   # -1: highest protocol
        self.memory_lock.release()
        # save variables
        with open(self.save_and_load_path + '_variables.pickle', 'wb') as f:
            pickle.dump((self.learn_step_counter, self.collect_sample_step_counter,
                         self.learn_time, self.collect_sample_time), f, protocol=-1)
        print('save model to', path)

    def load_model(self):
        # load tensorflow
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(self.save_and_load_path + '.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(self.save_and_load_path)))
            # placeholders
            self.s = self.graph.get_tensor_by_name('s:0')  # Q(s,a) feature
            self.s_ = self.graph.get_tensor_by_name('s_:0')  # Q(s',a') feature
            self.rewards = self.graph.get_tensor_by_name('reward:0')   # reward
            self.terminal_weights = self.graph.get_tensor_by_name('terminal:0')  # terminal
            # variables
            self.q_eval = self.graph.get_tensor_by_name('eval_net/q_eval:0')
            self.eval_w1 = self.graph.get_tensor_by_name('eval_net/l1/w1:0')
            self.eval_b1 = self.graph.get_tensor_by_name('eval_net/l1/b1:0')
            self.eval_w2 = self.graph.get_tensor_by_name('eval_net/l2/w2:0')
            self.eval_b2 = self.graph.get_tensor_by_name('eval_net/l2/b2:0')
            self.q_next = self.graph.get_tensor_by_name('eval_net/q_next:0')
            self.q_target = self.graph.get_tensor_by_name("q_target:0")
            self.is_weights = self.graph.get_tensor_by_name("is_weights:0")
            self.loss = self.graph.get_tensor_by_name("loss:0")
            self.abs_errors = self.graph.get_tensor_by_name("abs_errors:0")
            # operations
            self.train_op = self.graph.get_operation_by_name('train_op')
        # load memory
        with open(self.save_and_load_path + '_memory.pickle', 'rb') as f:
            self.memory = pickle.load(f)  # -1: highest protocol
        # load variables
        with open(self.save_and_load_path + '_variables.pickle', 'rb') as f:
            self.learn_step_counter, self.collect_sample_step_counter, \
                self.learn_time, self.collect_sample_time = pickle.load(f)

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # Q(s,a) feature
        self.s_ = tf.placeholder(tf.float32, [None, self.n_actions, self.n_features], name='s_')  # Q(s',a') feature
        self.rewards = tf.placeholder(tf.float32, [None], name='reward')  # reward
        self.terminal_weights = tf.placeholder(tf.float32, [None], name='terminal')  # terminal

        w_initializer, b_initializer = \
            tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            # s is Q(s,a) feature, shape: (n_sample, n_features)
            # s_ Q(s',a') for all a' feature, shape: (n_sample, n_actions, n_features)
            with tf.variable_scope('l1'):
                self.eval_w1 = tf.get_variable('w1', [self.n_features, self.n_hidden], initializer=w_initializer)
                self.eval_b1 = tf.get_variable('b1', [self.n_hidden], initializer=b_initializer)
                # l1 shape: (n_sample, n_hidden)
                l1 = tf.nn.relu(tf.matmul(self.s, self.eval_w1) + self.eval_b1)
                # l1_ shape: shape: (n_sample, n_actions, n_hidden)
                l1_ = tf.nn.relu(tf.einsum('ijk,kh->ijh', self.s_, self.eval_w1) + self.eval_b1)
            with tf.variable_scope('l2'):
                self.eval_w2 = tf.get_variable('w2', [self.n_hidden, 1], initializer=w_initializer)
                self.eval_b2 = tf.get_variable('b2', [1], initializer=b_initializer)
                # out shape: (n_sample, 1)
                out = tf.matmul(l1, self.eval_w2) + self.eval_b2
                # out_ shape: (n_sample, n_actions, 1), Q(s',a') for all a' feature
                out_ = tf.einsum('ijh,ho->ijo', l1_, self.eval_w2) + self.eval_b2
            self.q_eval = tf.squeeze(out, name='q_eval')
            self.q_next = tf.squeeze(out_, name='q_next')

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
        self.memory_lock.acquire()
        # transition is a tuple (current_state, action, reward, next_state, whether_terminal)
        self.memory.store((s, a, r, s_, terminal))
        self.memory_lock.release()

    def update_memory_priority(self, exp_ids, abs_errors):
        """ update memory priority """
        self.memory_lock.acquire()
        self.memory.update_priority(exp_ids, abs_errors)
        self.memory_lock.release()

    def choose_action(self, state, next_possible_states, next_possbile_actions, epsilon_greedy=True):
        pred_q_values = self.sess.run(self.q_eval, feed_dict={self.s: next_possible_states}).flatten()
        if not epsilon_greedy or np.random.uniform() < self.epsilon:
            action_idx = np.argmax(pred_q_values)
        else:
            action_idx = np.random.choice(numpy.arange(len(next_possbile_actions)))
        action = next_possbile_actions[action_idx]
        pred_q_value = pred_q_values[action_idx]
        return action, pred_q_value

    # def _replace_target_params(self):
    #     with self.graph.as_default():
    #         t_params = tf.get_collection('target_net_params')
    #         e_params = tf.get_collection('eval_net_params')
    #         self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
    #         print('target_params_replaced')

    def planning_learn(self, qsa_next_features, qsa_features):
        """ additional learning from planning """
        pass

    def learn(self, MEMORY_CAPACITY_START_LEARNING):
        while True:
            if self.memory_size() < MEMORY_CAPACITY_START_LEARNING:
                print('wait for more samples')
                time.sleep(1)
                continue
            #
            # if self.learn_step_counter % self.replace_target_iter == 0:
            #     self._replace_target_params()

            if self.learn_step_counter % self.save_model_iter == 0:
                self.save_model()

            learn_time = time.time()
            qsa_feature, qsa_next_features, rewards, terminal_weights, is_weights, exp_ids \
                = self.memory.sample()

            _, loss, abs_errors = self.sess.run([self.train_op, self.loss, self.abs_errors],
                                                feed_dict={self.s: qsa_feature,
                                                           self.s_: qsa_next_features,
                                                           self.rewards: rewards,
                                                           self.terminal_weights: terminal_weights,
                                                           self.is_weights: is_weights})

            if self.prioritized:
                self.update_memory_priority(exp_ids, abs_errors)

            if self.planning:
                self.planning_learn()

            self.epsilon = self.cur_epsilon()

            learn_time = time.time() - learn_time
            self.learn_step_counter += 1
            self.learn_time += learn_time

            print('LEARN:{}:mem_size:{}:virtual:{}:this time:{:.2f}:total:{:.2f}'.
                  format(self.learn_step_counter, self.memory_size(), self.memory_virtual_size(),
                         learn_time, self.learn_time))

    def cur_epsilon(self):
        return self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

    def tb_write(self, tags, values, step):
        """ write to tensorboard """
        self.tb_writer.write(tags, values, step)

    def memory_size(self):
        return self.memory.size

    def memory_virtual_size(self):
        return self.memory.virtual_size

    def collect_samples(self, EPISODE_SIZE, TRIAL_SIZE, MEMORY_CAPACITY_START_LEARNING, TEST_PERIOD, RANDOM_SEED):
        """ collect samples in a process """
        for i_episode in range(self.collect_sample_step_counter, EPISODE_SIZE):
            collect_sample_time = time.time()
            cur_state = self.env.reset()
            for i_epsisode_step in range(TRIAL_SIZE):
                next_possible_states, next_possible_actions = self.env.all_possible_next_state_action(cur_state)
                action, _ = self.choose_action(cur_state, next_possible_states, next_possible_actions,
                                               epsilon_greedy=True)
                cur_state_, reward = self.env.step(action)
                terminal = True if i_epsisode_step == TRIAL_SIZE - 1 else False
                self.store_transition(cur_state, action, reward, cur_state_, terminal)
                cur_state = cur_state_

            collect_sample_time = time.time() - collect_sample_time
            self.collect_sample_step_counter += 1
            self.collect_sample_time += collect_sample_time

            print('SAMPLE:{}:finished output:{:.5f}:cur_epsilon:{:.5f}:mem_size:{}:virtual:{}:this time:{:.2f}:total:{:.2f}'.
                  format(i_episode, self.env.output(cur_state), self.cur_epsilon(),
                         self.memory_size(), self.memory_virtual_size(),
                         collect_sample_time, self.collect_sample_time))

            # test every once a while
            if self.memory_virtual_size() >= MEMORY_CAPACITY_START_LEARNING and i_episode % TEST_PERIOD == 0:
                cur_state = self.env.reset()
                for i_episode_test_step in range(TRIAL_SIZE):
                    next_possible_states, next_possible_actions = self.env.all_possible_next_state_action(cur_state)
                    action, q_val = self.choose_action(cur_state, next_possible_states, next_possible_actions,
                                                       epsilon_greedy=False)
                    cur_state, reward = self.env.step(action)
                    test_output = self.env.output(cur_state)
                    print('TEST  :{}:output: {:.5f}, at {}, qval: {:.5f}, reward {:.5f}'.
                          format(i_episode_test_step, test_output, cur_state, q_val, reward))

                self.tb_write(tags=['Prioritized={0}, gamma={1}, seed={2}/Test Ending Output'.
                                format(self.prioritized, self.gamma, RANDOM_SEED),
                                      'Prioritized={0}, gamma={1}, seed={2}/Test Ending Qvalue'.
                                format(self.prioritized, self.gamma, RANDOM_SEED),
                                      ],
                                values=[test_output,
                                        q_val],
                                step=self.learn_step_counter)


