"""
This is a copy from experimenter.py.
This file removes the functionality to record experiment results.
We will test CPU time usage using this file
"""
import optparse
import numpy
from random_search import RandomSearch
from QLearning import QLearning
from supervise_learning import SuperviseLearning
import time
from multiprocessing import Process, freeze_support
from multiprocessing.managers import BaseManager
import os
import rbfopt

import random
from genetic_algorithm import cxTwoDeck, mutSwapCard, my_deck_creator_func, select_best_from_hof
from deap import base
from deap import creator
from deap import tools
import psutil


def get_prob_env_name_class(env_dir):
    env_name = env_dir.split('prob_')[1].split('_pv')[0]
    env_cls = None
    if env_name == 'env_nn':
        from environment.env_nn import Environment
        env_cls = Environment
    elif env_name == 'env_nn_noisy':
        from environment.env_nn_noisy import Environment
        env_cls = Environment
    elif env_name == 'env_greedymove':
        from environment.env_greedymove import Environment
        env_cls = Environment
    elif env_name == 'env_gamestate':
        from environment.env_gamestate import Environment
        env_cls = Environment
    return env_name, env_cls


def get_model_env_name(model_dir):
    env_name = model_dir.split('rl_prtr_')[1].split('_k')[0]
    return env_name


def rl_collect_samples(RL, EPISODE_SIZE, TEST_PERIOD):
    RL.collect_samples(EPISODE_SIZE, TEST_PERIOD)


def rl_learn(RL):
    RL.learn()


def get_cpu_time():
    cpu_time = psutil.Process().cpu_times()
    return cpu_time.user + cpu_time.system + cpu_time.children_system + cpu_time.children_user


def extract_rl_exp_result(opt, prob_env):
    _, _, opt_val, opt_state, duration, if_set_fixed_xo, start_state = opt.exp_test(debug=False)
    opt_x_o = opt_state[:prob_env.k]
    opt_x_p = opt_state[prob_env.k:-1]
    start_x_o = start_state[:prob_env.k]
    start_x_p = start_state[prob_env.k:-1]
    assert if_set_fixed_xo
    assert (numpy.array_equal(opt_x_o, prob_env.x_o) and numpy.array_equal(start_x_o, opt_x_o))
    assert (len(opt_x_p) == prob_env.k)
    return opt_val, start_x_o, opt_x_p, start_x_p, duration


if __name__ == '__main__':
    # for multiprocessing on windows
    freeze_support()

    parser = optparse.OptionParser(usage="usage: %prog [options]")
    parser.add_option("--wall_time_limit", dest="wall_time_limit",
                      help="wall time limit", type="float", default=0)
    parser.add_option("--prob_env_dir", dest="prob_env_dir",
                      help="environment directory", type="string", default="test_probs/prob_env_nn_0")
    parser.add_option("--prtr_model_dir", dest="prtr_model_dir",
                      help="model directory of rlprtr or sl", type="string", default="rl_prtr")
    parser.add_option("--rl_load", dest="rl_load",
                      help="whether to load rl model to continue learning", type="int", default=0)
    parser.add_option("--sl_num_trial", dest="sl_num_trial",
                      help="number of trials in supervise learning test", type="int", default=0)
    # method:
    # 1. rl_prtr (pretraining)
    # 2. rl (no-pretraining)
    # 3. rbf
    # 4. random
    # 5. ga (genetic algorithm)
    parser.add_option("--method", dest='method',
                      help="method to test", type="string", default="rl")
    (kwargs, args) = parser.parse_args()

    prob_env_name, prob_env_class = get_prob_env_name_class(kwargs.prob_env_dir)
    prob_env = prob_env_class.load(kwargs.prob_env_dir)

    if kwargs.method == 'random':
        # problem environment has not fixed x_o.
        # however, we want to fix x_o for monto carlo random search
        assert not prob_env.if_set_fixed_xo()
        prob_env.set_fixed_xo(prob_env.x_o)
        assert prob_env.if_set_fixed_xo()
        opt = RandomSearch(prob_env)
        cpu_time = get_cpu_time()
        print("before test {}, cpu time: {}".format(kwargs.method, cpu_time))
        _, opt_state, _, _, duration, call_counts = \
            opt.random_search(
                iteration_limit=int(9e30),  # never stop until wall_time_limt
                wall_time_limit=kwargs.wall_time_limit,
            )
        print("after test {}, cpu time: {}, diff: {}".format(kwargs.method, get_cpu_time(), get_cpu_time() - cpu_time))
    elif kwargs.method == 'rl_prtr':
        model_env_name = get_model_env_name(kwargs.prtr_model_dir)
        assert model_env_name == prob_env_name
        opt = QLearning(
            k=prob_env.k,
            d=prob_env.d,
            env_name=prob_env_name,
            env_dir=kwargs.prob_env_dir,     # env_dir will load environment
            env_fixed_xo=False,
            n_hidden=0,
            save_and_load_path=kwargs.prtr_model_dir,  # model dir will load model
            load=True,
            tensorboard_path=None,
            logger_path=None,
            memory_capacity=None,
            memory_capacity_start_learning=None,
            learn_wall_time_limit=None,
            prioritized=None,
            save_model_iter=None,
            trial_size=0,
        )
        # opt is learned from non fixed xo environment
        # but we will test it under fixed xo environment
        assert not opt.get_env_if_set_fixed_xo()
        opt.set_env_fixed_xo(prob_env.x_o)
        assert opt.get_env_if_set_fixed_xo()
        cpu_time = get_cpu_time()
        print("before test {}, cpu time: {}".format(kwargs.method, cpu_time))
        opt_val, start_x_o, opt_x_p, start_x_p, duration = extract_rl_exp_result(opt, prob_env)
        print("after test {}, cpu time: {}, diff: {}".format(kwargs.method, get_cpu_time(), get_cpu_time() - cpu_time))
    elif kwargs.method == 'ga':
        print("cpu time after load:", get_cpu_time())
        # problem environment has not fixed x_o.
        # however, we want to fix x_o for rbf method
        assert not prob_env.if_set_fixed_xo()
        prob_env.set_fixed_xo(prob_env.x_o)
        assert prob_env.if_set_fixed_xo()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", numpy.ndarray, typecode='b', fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        my_deck_creator = my_deck_creator_func(prob_env.k, prob_env.d)
        toolbox.register("my_individual_creator", my_deck_creator)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.my_individual_creator)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        call_counts = 0
        start_time = time.time()
        pop_size = 10
        hof_size = 1
        CXPB = 0.2  # the probability with which two individuals are crossed
        MUTPB = 0.2  # the probability for mutating an individual

        def ga_output(x_p):
            # barrier method to invalidate constrained violation
            # different than rbf, genetic algorithm is a maximizer
            if numpy.sum(x_p) != prob_env.d:
                out = - numpy.abs(numpy.sum(x_p) - prob_env.d)
            else:
                out = prob_env.still(prob_env.output(numpy.hstack((prob_env.x_o, x_p, 0))))   # the last one is step placeholder
                global call_counts
                call_counts += 1
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
        last_gen_print_time = time.time()

        cpu_time = get_cpu_time()
        print("before test {}, cpu time: {}".format(kwargs.method, cpu_time))
        while True:
            gen = gen + 1
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            if time.time() - last_gen_print_time > 15:
                last_gen_print_time = time.time()
                print("-- Generation {}, Size {}, Time {}, Call Counts {} --".
                      format(gen, len(offspring), time.time() - start_time, call_counts))
                print("  Min %s" % min(fits))
                print("  Max %s" % max(fits))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if time.time() - start_time > kwargs.wall_time_limit:
                    break
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if time.time() - start_time > kwargs.wall_time_limit:
                    break
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            if time.time() - start_time > kwargs.wall_time_limit:
                break
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in pop]

        # opt_val is noiseless
        opt_val, opt_x_p = select_best_from_hof(tools.selBest(pop, hof_size), prob_env)
        print("after test {}, cpu time: {}, diff: {}".format(kwargs.method, get_cpu_time(), get_cpu_time() - cpu_time))
    elif kwargs.method == 'sl':
        # a monte carlo method + a supervised learning model
        # problem environment has not fixed x_o.
        # however, we want to fix x_o for rbf method
        assert not prob_env.if_set_fixed_xo()
        prob_env.set_fixed_xo(prob_env.x_o)
        assert prob_env.if_set_fixed_xo()
        sl_model = SuperviseLearning(k=prob_env.k,
                                     d=prob_env.d,
                                     env_name=prob_env_name,
                                     wall_time_limit=kwargs.wall_time_limit,
                                     load=True,
                                     path=kwargs.prtr_model_dir)
        # cpu time is already recorded in train_and_test function
        opt_val, duration, opt_state = sl_model.train_and_test(prob_env, kwargs.sl_num_trial)




