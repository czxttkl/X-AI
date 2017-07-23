import numpy
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
from my_RL_brain import DQNPrioritizedReplay
sess = tf.Session()
k = 30   # total available card size
d = 10    # deck size
# power of one, two and interaction
kk = k + k + k * (k-1) / 2
kk = int(kk)
MEMORY_SIZE = 5000
TRIAL_SIZE = d
EPISODE_SIZE = 10000
MONTE_CARLO_ITERATIONS = 10000
TEST_PERIOD = 10


with tf.variable_scope('DQN_with_prioritized_replay'):
    RL = DQNPrioritizedReplay(
        n_features=k, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
    )
sess.run(tf.global_variables_initializer())


class Environment():
    def __init__(self, k, d, kk):
        self.k = k
        self.d = d
        self.kk = kk
        self.poly = PolynomialFeatures(2, include_bias=False)
        self.coef = 10 * numpy.random.rand(self.kk)
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
        print("monte carlo max: {0}, min: {1}".format(numpy.max(m), numpy.min(m)))

    def reset(self):
        feature = numpy.zeros(self.k)
        one_idx = numpy.random.choice(self.k, self.d, replace=False)
        feature[one_idx] = 1
        self.curr_feature = feature
        return self.curr_feature.copy()

    def output(self, feature):
        trans_feature = self.poly.fit_transform(feature.reshape((1, -1))).flatten()
        out = numpy.dot(self.coef, trans_feature)
        return out

    def step(self, action):
        old_out = self.output(self.curr_feature)
        idx_to_remove, idx_to_add = action[0], action[1]
        self.curr_feature[idx_to_remove] = 0
        self.curr_feature[idx_to_add] = 1
        new_out = self.output(self.curr_feature)
        reward = new_out - old_out
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
        if total_steps > MEMORY_SIZE:
            RL.learn()

    print('episode ', i_episode, ' finished with value', env.output(observation), 'cur_epsilon', RL.cur_epsilon())

    if total_steps > MEMORY_SIZE and i_episode % TEST_PERIOD == 0:
        observation = env.reset()
        for i_episode_test_step in range(TRIAL_SIZE):
            action = RL.choose_action(observation, epsilon_greedy=False)
            observation, _ = env.step(action)
            print('TEST step {0}, finished with value {1}'.
                  format(i_episode_test_step, env.output(observation)))






