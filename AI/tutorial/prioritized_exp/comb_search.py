import numpy
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
from my_RL_brain import DQNPrioritizedReplay
import time

sess = tf.Session()
k = 50   # total available card size
d = 30    # deck size
use_prioritized_replay = True
# the function to optimize is a linear combination of
# power of one, two and interaction
kk = k + k + k * (k-1) / 2
kk = int(kk)
MEMORY_SIZE = 5000
TRIAL_SIZE = d // 2          # how many card modification allowed
EPISODE_SIZE = 600           # the size of training episodes
MONTE_CARLO_ITERATIONS = 100 # use monte carlo samples to determine max and min
TEST_PERIOD = 10             # how many per training episodes to do training
# np.random.seed(1)
# tf.set_random_seed(1)

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL = DQNPrioritizedReplay(
        n_features=k, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=use_prioritized_replay, output_graph=True,
    )
summary_writer = tf.summary.FileWriter("comb_search_k{0}_d{1}/{2}".format(k, d, time.time()))

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
        m = []
        for i in range(MONTE_CARLO_ITERATIONS):
            feature = numpy.zeros(self.k)
            one_idx = numpy.random.choice(self.k, self.d, replace=False)
            feature[one_idx] = 1
            m.append(self.output(feature))
            if i % 1000 == 0:
                print(i)
        print('coefficient=', self.coef)
        print("monte carlo max: {0}, min: {1}".format(numpy.max(m), numpy.min(m)))

    def reset(self):
        feature = numpy.zeros(self.k)
        one_idx = numpy.random.choice(self.k, self.d, replace=False)
        feature[one_idx] = 1
        self.curr_feature = feature
        return self.curr_feature.copy()

    def output(self, feature):
        # t1 = time.time()
        trans_feature = self.poly.fit_transform(feature.reshape((1, -1))).flatten()
        out = numpy.dot(self.coef, trans_feature)
        # t2 = time.time()
        # print('step output time', t2 - t1)
        return out

    def step(self, action):
        old_out = self.output(self.curr_feature)
        idx_to_remove, idx_to_add = action[0], action[1]
        self.curr_feature[idx_to_remove] = 0
        self.curr_feature[idx_to_add] = 1
        new_out = self.output(self.curr_feature)
        # reward = new_out - old_out
        reward = old_out - new_out
        # reward = new_out
        return self.curr_feature.copy(), reward


env = Environment(k=k, d=d, kk=kk)
total_steps = 0

for i_episode in range(EPISODE_SIZE):
    episode_steps = 0
    observation = env.reset()
    for i_epsisode_step in range(TRIAL_SIZE):
        action = RL.choose_action(observation, epsilon_greedy=True)
        observation_, reward = env.step(action)
        RL.store_transition(observation, action, reward, observation_)
        total_steps += 1
        observation = observation_
        if total_steps > MEMORY_SIZE / 5:
            RL.learn()

    print('episode ', i_episode, ' finished with value', env.output(observation), 'cur_epsilon', RL.cur_epsilon())

    if total_steps > MEMORY_SIZE / 5 and i_episode % TEST_PERIOD == 0:
        observation = env.reset()
        for i_episode_test_step in range(TRIAL_SIZE):
            action = RL.choose_action(observation, epsilon_greedy=False)
            observation, reward = env.step(action)
            test_output = env.output(observation)
            print('TEST step {0}, finished with value {1} and reward {2}'.
                  format(i_episode_test_step, test_output, reward))

        summary = tf.Summary()
        summary.value.add(tag='Prioritized={0}/Last Test Value'.format(use_prioritized_replay),
                          simple_value=test_output)
        summary_writer.add_summary(summary, global_step=i_episode)
        summary_writer.flush()





