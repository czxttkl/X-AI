import numpy
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
from TD import TDLearning
import time

sess = tf.Session()
k = 100   # total available card size
d = 30    # deck size
use_prioritized_replay = False
# the function to optimize is a linear combination of
# power of one, two and interaction
kk = k + k + k * (k-1) // 2
MEMORY_SIZE = 64000
MEMORY_SIZE_START_LEARNING = 1000
TRIAL_SIZE = d // 2               # how many card modification allowed
EPISODE_SIZE = 1001           # the size of training episodes
MONTE_CARLO_ITERATIONS = 100  # use monte carlo samples to determine max and min
TEST_PERIOD = 10              # how many per training episodes to do training
coef_seed = 1234             # seed for coefficient generation
random_seed = 1111           # seed for random behavior except coefficient generation

RL = TDLearning(
    n_features=k, memory_size=MEMORY_SIZE,
    e_greedy_increment=0.0005, sess=sess, prioritized=use_prioritized_replay, output_graph=True,
)
summary_writer = tf.summary.FileWriter("comb_search_k{0}_d{1}/{2}".format(k, d, time.time()))
sess.run(tf.global_variables_initializer())


class Environment():
    def __init__(self, k, d):
        self.k = k
        self.d = d
        self.func_generate()
        self.monte_carlo()

    def func_generate(self):
        self.poly = PolynomialFeatures(2, include_bias=False)
        # generate fixed coefficient
        numpy.random.seed(random_seed)
        self.coef = 10 * numpy.random.rand(kk)
        if random_seed is None:
            numpy.random.seed()
        else:
            numpy.random.seed(random_seed)
            tf.set_random_seed(random_seed)

    def monte_carlo(self):
        """ Use monte carlo to find the max value """
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
        assert len(feature.shape) == 1
        trans_feature = self.poly.fit_transform(feature.reshape((1, -1))).flatten()
        out = numpy.dot(self.coef, trans_feature)
        # t2 = time.time()
        # print('step output time', t2 - t1)
        return out

    def outputs(self, features):
        assert len(features.shape) == 2
        trans_features = self.poly.fit_transform(features)
        outs = numpy.dot(self.coef, trans_features.T)
        return outs

    def step(self, action):
        old_out = self.output(self.curr_feature)
        idx_to_remove, idx_to_add = action[0], action[1]
        self.curr_feature[idx_to_remove] = 0
        self.curr_feature[idx_to_add] = 1
        new_out = self.output(self.curr_feature)
        reward = new_out - old_out
        # reward = old_out - new_out
        # reward = new_out
        return self.curr_feature.copy(), reward


env = Environment(k=k, d=d)
total_steps = 0

for i_episode in range(EPISODE_SIZE):
    episode_steps = 0
    observation = env.reset()
    for i_epsisode_step in range(TRIAL_SIZE):
        action = RL.choose_action(observation, state_val_eval_func=env.output,
                                  state_vals_eval_func=env.outputs, epsilon_greedy=True)
        observation_, reward = env.step(action)
        RL.store_transition(observation, action, reward, observation_)
        total_steps += 1
        observation = observation_
        if total_steps > MEMORY_SIZE_START_LEARNING:
            RL.learn()

    print('episode ', i_episode, ' finished with value', env.output(observation), 'cur_epsilon', RL.cur_epsilon())

    if total_steps > MEMORY_SIZE_START_LEARNING and i_episode % TEST_PERIOD == 0:
        observation = env.reset()
        for i_episode_test_step in range(TRIAL_SIZE):
            action = RL.choose_action(observation, state_val_eval_func=env.output,
                                      state_vals_eval_func=env.outputs, epsilon_greedy=False)
            observation, reward = env.step(action)
            test_output = env.output(observation)
            print('TEST step {0}, finished with value {1} and reward {2}'.
                  format(i_episode_test_step, test_output, reward))

        summary = tf.Summary()
        summary.value.add(tag='TD Prioritized={0}/Last Test Value'.format(use_prioritized_replay),
                          simple_value=test_output)
        summary_writer.add_summary(summary, global_step=i_episode)
        summary_writer.flush()





