"""
Command-line parameterized experiment tool
"""
import optparse
import numpy
import tensorflow as tf
from QLearning import QLearning
import time
from multiprocessing import Process, freeze_support
from multiprocessing.managers import BaseManager
import os
import rbfopt

import array
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def get_prob_env_name_class(env_dir):
    env_name = env_dir.split('prob_')[1].split('_pv')[0]
    env_cls = None
    if env_name == 'env_nn':
        from environment.env_nn import Environment
        env_cls = Environment
    return env_name, env_cls


def get_model_env_name(model_dir):
    env_name = model_dir.split('rl_prtr_')[1].split('_k')[0]
    return env_name


def rl_collect_samples(RL, EPISODE_SIZE, TEST_PERIOD):
    RL.collect_samples(EPISODE_SIZE,
                       TEST_PERIOD)


def rl_learn(RL):
    RL.learn()


def rl_opt(RL):
    RL.exp_test()


def extract_exp_result(opt, prob_env):
    opt_val, opt_state, _, _, duration, if_set_fixed_xo, start_state = opt.exp_test()
    opt_x_o = opt_state[:prob_env.k]
    opt_x_p = opt_state[prob_env.k:]
    start_x_o = start_state[:prob_env.k]
    start_x_p = start_state[prob_env.k:]
    assert if_set_fixed_xo
    assert (numpy.array_equal(opt_x_o, prob_env.x_o) and numpy.array_equal(start_x_o, opt_x_o))
    assert (len(opt_x_p) == prob_env.k + 1)
    return opt_val, start_x_o, opt_x_p, start_x_p


if __name__ == '__main__':
    # for multiprocessing on windows
    freeze_support()

    parser = optparse.OptionParser(usage="usage: %prog [options]")
    parser.add_option("--wall_time_limit", dest="wall_time_limit",
                      help="wall time limit", type="int", default=0)
    parser.add_option("--prob_env_dir", dest="prob_env_dir",
                      help="environment directory", type="string", default="test_probs/prob_env_nn_0")
    parser.add_option("--prtr_model_dir", dest="prtr_model_dir",
                      help="model directory", type="string", default="rl_prtr")
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
        opt_val, opt_state, _, _, duration = \
            prob_env.monte_carlo(
                MONTE_CARLO_ITERATIONS=int(9e30),  # never stop until wall_time_limt
                WALL_TIME_LIMIT=kwargs.wall_time_limit)
        start_x_o = prob_env.fixed_xo
        start_x_p = None   # meaningless in random method
        opt_x_p = opt_state[prob_env.k:-1]   # exclude the step
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
            learn_interval=None,
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
        opt_val, start_x_o, opt_x_p, start_x_p = extract_exp_result(opt, prob_env)
    elif kwargs.method == 'rl':
        parent_path = os.path.abspath(os.path.join(
            kwargs.prob_env_dir,
            'rl_{}_k{}_d{}_t{}'.
                format(prob_env_name, prob_env.k, prob_env.d, kwargs.wall_time_limit)))
        tensorboard_path = os.path.join(parent_path, str(time.time()))
        model_save_load_path = os.path.join(parent_path, 'optimizer_model_fixedxoTrue', 'qlearning')
        logger_path = os.path.join(parent_path, 'logger_fixedxoTrue.log')
        # initialize critical components
        BaseManager.register('QLearning', QLearning)
        manager = BaseManager()
        manager.start()
        opt = manager.QLearning(
                k=0,    # will load from environment dir
                d=0,    # will load from environment dir
                env_name=prob_env_name,
                env_dir=kwargs.prob_env_dir,
                env_fixed_xo=None,    # will load from environment dir
                n_hidden=400,
                save_and_load_path=model_save_load_path,
                load=False,
                tensorboard_path=tensorboard_path,
                logger_path=logger_path,
                learn_interval=1,
                memory_capacity=300000,
                memory_capacity_start_learning=10000,
                learn_wall_time_limit=kwargs.wall_time_limit,
                prioritized=True,
                save_model_iter=1000,
                trial_size=prob_env.d,
        )
        # opt loads the environment which has not fixed x_o.
        # But it will learn based on fixed x_o.
        assert not opt.get_env_if_set_fixed_xo()
        opt.set_env_fixed_xo(prob_env.x_o)
        assert opt.get_env_if_set_fixed_xo()

        EPISODE_SIZE = int(9e30)  # run until hitting wall time limit
        TEST_PERIOD = int(9e30)   # never test
        p1 = Process(target=rl_collect_samples,
                     args=[opt, EPISODE_SIZE, TEST_PERIOD])
        p1.start()
        p2 = Process(target=rl_learn,
                     args=[opt])
        p2.start()
        p1.join()
        p2.join()
        opt_val, start_x_o, opt_x_p, start_x_p = extract_exp_result(opt, prob_env)
    elif kwargs.method == 'rbf':
        # problem environment has not fixed x_o.
        # however, we want to fix x_o for rbf method
        assert not prob_env.if_set_fixed_xo()
        prob_env.set_fixed_xo(prob_env.x_o)
        assert prob_env.if_set_fixed_xo()

        def rbf_output(x_p):
            # barrier method to invalidate constrained violation
            if numpy.sum(x_p) != prob_env.d:
                out = numpy.abs(numpy.sum(x_p) - prob_env.d)
            else:
                # since rbf-opt is a minimizer, we need to make the output negative
                out = - prob_env.output(numpy.hstack((prob_env.x_o, x_p, 0)))  # the last one is step placeholder
            print("x: {}, out: {}".format(x_p, out))
            return out

        rbf_bb = rbfopt.RbfoptUserBlackBox(prob_env.k,  # dimension
                                           numpy.array([0] * prob_env.k),  # lower bound
                                           numpy.array([1] * prob_env.k),  # upper bound
                                           numpy.array(['I'] * prob_env.k),  # type: integer
                                           rbf_output)
        # since evaluating f(x) would require parallelism for the deck recommendation problem,
        # we don't increase num_cpus here
        settings = rbfopt.RbfoptSettings(max_evaluations=1e30, max_iterations=1e30,
                                         max_clock_time=kwargs.wall_time_limit, num_cpus=1)
        opt = rbfopt.RbfoptAlgorithm(settings, rbf_bb)
        # minimize
        opt_val, opt_x_p, _, _, _ = opt.optimize()
        opt_val = prob_env.still(-opt_val)
        start_x_o = prob_env.fixed_xo
        start_x_p = None    # meaningless in rbf method
    elif kwargs.method == 'ga':
        # problem environment has not fixed x_o.
        # however, we want to fix x_o for rbf method
        assert not prob_env.if_set_fixed_xo()
        prob_env.set_fixed_xo(prob_env.x_o)
        assert prob_env.if_set_fixed_xo()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, prob_env.k)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def ga_output(x_p):
            # barrier method to invalidate constrained violation
            # different than rbf, genetic algorithm is a maximizer
            if numpy.sum(x_p) != prob_env.d:
                out = - numpy.abs(numpy.sum(x_p) - prob_env.d)
            else:
                out = prob_env.output(numpy.hstack((prob_env.x_o, x_p, 0)))  # the last one is step placeholder
            return float(out),

        toolbox.register("evaluate", ga_output)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = toolbox.population(n=300)
        CXPB, MUTPB = 0.5, 0.2
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        gen = 0
        start_time = time.time()
        last_gen_print_time = time.time()
        while True:
            gen = gen + 1
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            if time.time() - last_gen_print_time > 15:
                last_gen_print_time = time.time()
                print("-- Generation {}, Size {}, Time {} --".format(gen, len(offspring), time.time() - start_time))
                print("  Min %s" % min(fits))
                print("  Max %s" % max(fits))
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
            if time.time() - start_time > kwargs.wall_time_limit:
                break

        best_ind = tools.selBest(pop, 1)[0]
        opt_val = prob_env.still(best_ind.fitness.values[0])
        opt_x_p = numpy.array(best_ind, dtype=numpy.float)
        start_x_o = prob_env.fixed_xo
        start_x_p = None   # meaningless in genetic algorithm

    # output test results
    numpy.set_printoptions(linewidth=10000)
    test_result_path = os.path.join(kwargs.prob_env_dir, 'test_result.csv')
    # write header
    if not os.path.exists(test_result_path):
        with open(test_result_path, 'w') as f:
            line = "method, wall_time_limit, opt_val, start_x_o, opt_x_p, start_x_p \n"
            f.write(line)
    # write data
    with open(test_result_path, 'a') as f:
        line = "{}, {}, {}, {}, {}, {}\n".\
            format(kwargs.method, kwargs.wall_time_limit, opt_val, start_x_o, opt_x_p, start_x_p)
        f.write(line)


