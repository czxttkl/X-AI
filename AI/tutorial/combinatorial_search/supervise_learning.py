import time
import numpy
import pickle
import optparse
import os
numpy.set_printoptions(linewidth=10000, precision=5)


class SuperviseLearning:

    def __init__(self, k, d, env_name, wall_time_limit, load):
        self.k = k
        self.d = d
        self.env_name = env_name
        self.wall_time_limit = wall_time_limit
        self.path = 'prtr_models/sl_{}_k{}_d{}_t{}'.format(env_name, k, d, wall_time_limit)
        os.makedirs(self.path, exist_ok=True)

        if env_name == 'env_nn':
            from environment.env_nn import Environment
        elif env_name == 'env_nn_noisy':
            from environment.env_nn_noisy import Environment
        elif env_name == 'env_greedymove':
            from environment.env_greedymove import Environment
        elif env_name == 'env_gamestate':
            from environment.env_gamestate import Environment

        self.env = Environment(k=k, d=d)
        assert not self.env.if_set_fixed_xo()

        if load:
            with open(os.path.join(self.path, 'data.pickle'), 'rb') as f:
                self.x, self.y, self.time_passed = pickle.load(f)
        else:
            self.x, self.y = [], []
            self.time_passed = 0

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

    def train(self, train_path):
        """ train a neural network model. This function gets called in ad-hoc """
        pass

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

