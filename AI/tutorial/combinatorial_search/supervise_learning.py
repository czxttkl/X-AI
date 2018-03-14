import time
import numpy
import pickle
import optparse
import os
import psutil
from sklearn.neural_network import MLPRegressor
numpy.set_printoptions(linewidth=10000, precision=5)


class SuperviseLearning:

    def __init__(self, k, d, env_name, wall_time_limit, load, path=None):
        self.k = k
        self.d = d
        self.env_name = env_name
        self.wall_time_limit = int(wall_time_limit)

        path_prefix = 'prtr_models/sl_{}_k{}_d{}_t{}'.format(self.env_name, self.k, self.d, self.wall_time_limit)
        if path:
            assert path.startswith(path_prefix)
            self.path = path
        else:
            self.path = path_prefix
        os.makedirs(self.path, exist_ok=True)

        if self.env_name == 'env_nn':
            from environment.env_nn import Environment
        elif self.env_name == 'env_nn_noisy':
            from environment.env_nn_noisy import Environment
        elif self.env_name == 'env_greedymove':
            from environment.env_greedymove import Environment
        elif self.env_name == 'env_gamestate':
            from environment.env_gamestate import Environment

        self.env = Environment(k=self.k, d=self.d)
        assert not self.env.if_set_fixed_xo()

        if load:
            with open(os.path.join(self.path, 'data.pickle'), 'rb') as f:
                self.x, self.y, self.time_passed = pickle.load(f)
        else:
            self.x, self.y = [], []
            self.time_passed = 0

    def get_cpu_time(self):
        cpu_time = psutil.Process().cpu_times()
        return cpu_time.user + cpu_time.system + cpu_time.children_system + cpu_time.children_user

    def save_data(self):
        with open(os.path.join(self.path, 'data.pickle'), 'wb') as f:
            pickle.dump((self.x, self.y, self.time_passed), f, protocol=-1)

    def collect_samples(self):
        last_save_time = time.time()

        while self.time_passed < self.wall_time_limit:
            self.env.reset()
            win_rate = self.env.still(self.env.output(self.env.cur_state))

            self.x.append(self.env.cur_state[:-1])    # exclude step
            self.y.append(win_rate)
            print('save #{} time {} data {}, {}'.format(len(self.x),
                                                        self.time_passed + time.time() - last_save_time,
                                                        self.y[-1],
                                                        self.x[-1]))

            # save every 6 min
            if time.time() - last_save_time > 60 * 6:
                self.time_passed += time.time() - last_save_time
                self.save_data()
                last_save_time = time.time()

    def train_and_test(self, prob_env, num_trials):
        """
        Train a neural network model. Then, test on a test problem.
        This function gets called in ad-hoc or in experimenter.py
        """
        duration = time.time()
        mlp = MLPRegressor(hidden_layer_sizes=(1000, ), early_stopping=True, max_iter=2000)
        size = len(self.x)
        x, y = numpy.array(self.x[:-size//10]), numpy.array(self.y[:-size//10])
        x_test, y_test = numpy.array(self.x[-size//10:]), numpy.array(self.y[-size//10:])
        mlp.fit(x, y)
        duration = time.time() - duration

        mse = ((y - mlp.predict(x)) ** 2).mean()
        mse_test = ((y_test - mlp.predict(x_test)) ** 2).mean()

        print("Training set R^2 score: %f, MSE: %f, time: %f, size: %d" % (mlp.score(x, y), mse, duration, len(y)))
        print("Test set R^2 score: %f, MSE: %f, size: %d" % (mlp.score(x_test, y_test), mse_test, len(y_test)))

        x_o = numpy.copy(prob_env.x_o)

        max_pred_win_rate = 0
        max_state = None

        duration = time.time()
        cpu_time = self.get_cpu_time()
        for i in range(num_trials):
            random_xp = numpy.zeros(prob_env.k + 1)  # state + step
            one_idx = numpy.random.choice(prob_env.k, prob_env.d, replace=False)
            random_xp[one_idx] = 1

            random_state = numpy.hstack((x_o, random_xp))
            random_state_output = mlp.predict(random_state[:-1].reshape((1, -1)))

            if random_state_output > max_pred_win_rate:
                max_pred_win_rate = random_state_output
                max_state = random_state
        duration = time.time() - duration
        cpu_time = self.get_cpu_time() - cpu_time
        max_real_win_rate = prob_env.still(prob_env.output(max_state))
        print("Trial: {}, duration: {}, cpu time: {}, max predict win rate: {}, real win rate: {}"
              .format(num_trials, duration, cpu_time, max_pred_win_rate, max_real_win_rate))

        return max_real_win_rate, duration, max_state


if __name__ == '__main__':
    parser = optparse.OptionParser(usage="usage: %prog [options]")
    parser.add_option("--k", dest="k", type="int")
    parser.add_option("--d", dest="d", type="int")
    parser.add_option("--env_name", dest="env_name", type="string")
    parser.add_option("--wall_time_limit", dest="wtl", type="int")
    parser.add_option("--load", dest="load", type="int")
    (kwargs, args) = parser.parse_args()
    if kwargs.load == 1:
        load = True
    else:
        load = False
    sl_model = SuperviseLearning(k=kwargs.k, d=kwargs.d, env_name=kwargs.env_name, wall_time_limit=kwargs.wtl, load=load)
    sl_model.collect_samples()

