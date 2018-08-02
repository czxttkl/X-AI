"""
Supervise learning model for MC simulation
"""
import random
import time
import numpy
import pickle
import optparse
import os
import sys
import psutil
from sklearn.neural_network import MLPClassifier
from genetic_algorithm import cxTwoDeck, mutSwapCard, my_deck_creator_func, select_best_from_hof
from deap import base
from deap import creator
from deap import tools
numpy.set_printoptions(linewidth=10000, precision=5)


class MultiLabelLearning:

    def __init__(self, k, d, env_name, wall_time_limit, load, path=None):
        self.k = k
        self.d = d
        self.env_name = env_name
        self.wall_time_limit = int(wall_time_limit)

        path_prefix = 'prtr_models/ml_{}_k{}_d{}_t{}'.format(self.env_name, self.k, self.d, self.wall_time_limit)
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
            if os.path.exists(os.path.join(self.path, 'data.pickle')):
                print("not load but file exist:", os.path.join(self.path, 'data.pickle'))
                sys.exit()
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
            self.env.unset_fixed_xo()
            assert not self.env.if_set_fixed_xo()
            self.env.reset()
            self.env.set_fixed_xo(self.env.x_o)
            assert self.env.if_set_fixed_xo()
            x_o, x_p = self.ga_x_o_x_p()
            self.x.append(x_o)
            self.y.append(x_p)
            print('save #{} time {} data {}, {}'.format(len(self.x),
                                                        self.time_passed + time.time() - last_save_time,
                                                        self.y[-1],
                                                        self.x[-1]))

            # save every 6 min
            if time.time() - last_save_time > 60 * 6:
                self.time_passed += time.time() - last_save_time
                self.save_data()
                last_save_time = time.time()

    def ga_x_o_x_p(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", numpy.ndarray, typecode='b', fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        my_deck_creator = my_deck_creator_func(self.env.k, self.env.d)
        toolbox.register("my_individual_creator", my_deck_creator)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.my_individual_creator)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        start_time = time.time()
        pop_size = 10
        CXPB = 0.2  # the probability with which two individuals are crossed
        MUTPB = 0.2  # the probability for mutating an individual

        prob_env = self.env
        def ga_output(x_p):
            # barrier method to invalidate constrained violation
            # different than rbf, genetic algorithm is a maximizer
            if numpy.sum(x_p) != prob_env.d:
                out = - numpy.abs(numpy.sum(x_p) - prob_env.d)
            else:
                out = prob_env.still(
                    prob_env.output(numpy.hstack((prob_env.x_o, x_p, 0))))  # the last one is step placeholder
            return float(out),

        toolbox.register("evaluate", ga_output)
        toolbox.register("mate", cxTwoDeck)
        toolbox.register("mutate", mutSwapCard)
        toolbox.register("select", tools.selTournament, tournsize=3)
        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = toolbox.population(n=pop_size)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        gen = 0
        max_obj = -999
        max_obj_gen = 0
        fits = [0]
        while True:
            gen = gen + 1
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            print("data point {}, gen {}, Size {}, Time {}, min {}, max {} ".
                  format(len(self.x) + 1, gen, len(offspring), time.time() - start_time, min(fits), max(fits)))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in pop]

            if max(fits) > max_obj:
                max_obj_gen = gen
                max_obj = max(fits)

            # if no improvement over 10 generations, stop
            if gen - max_obj_gen > 10:
                print("STOP")
                break

        # opt_val is noiseless
        opt_x_p = numpy.array(tools.selBest(pop, 1)[0])
        x_o = prob_env.fixed_xo
        print('x_o:', x_o)
        print('opt_x_p:', opt_x_p)
        print('type opt_x_p', type(opt_x_p))
        print()
        return x_o, opt_x_p

    def train_and_test(self, prob_env):
        """
        Train a neural network model. Then, test on a test problem.
        This function gets called in ad-hoc or in experimenter.py
        """
        duration = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=(1000, ), early_stopping=True, max_iter=2000, verbose=True)
        size = len(self.x)
        x, y = numpy.array(self.x[:-size//10]), numpy.array(self.y[:-size//10])
        x_test, y_test = numpy.array(self.x[-size//10:]), numpy.array(self.y[-size//10:])
        print('training size:', len(y))
        mlp.fit(x, y)
        duration = time.time() - duration

        mse = ((y - mlp.predict(x)) ** 2).mean()
        mse_test = ((y_test - mlp.predict(x_test)) ** 2).mean()

        print("Training set R^2 score: %f, MSE: %f, time: %f, size: %d" % (mlp.score(x, y), mse, duration, len(y)))
        print("Test set R^2 score: %f, MSE: %f, size: %d" % (mlp.score(x_test, y_test), mse_test, len(y_test)))

        x_o = numpy.copy(prob_env.x_o).reshape((1, -1))

        duration = time.time()
        pred_probs = mlp.predict_proba(x_o)[0]
        cards = numpy.argsort(pred_probs)[::-1][:prob_env.d]
        opt_x_p = numpy.zeros(prob_env.k)  # state + step
        opt_x_p[cards] = 1

        duration = time.time() - duration
        state = numpy.hstack((prob_env.x_o, opt_x_p, [0]))
        real_win_rate = prob_env.still(prob_env.output(state))
        print("duration: {}, real win rate: {}".format(duration, real_win_rate))

        return real_win_rate, duration, opt_x_p


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
    sl_model = MultiLabelLearning(k=kwargs.k, d=kwargs.d, env_name=kwargs.env_name, wall_time_limit=kwargs.wtl, load=load)
    sl_model.collect_samples()

