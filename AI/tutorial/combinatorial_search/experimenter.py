"""
Command-line parameterized experiment tool
"""
import optparse
import numpy
from random_search import RandomSearch
from QLearning import QLearning
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
                      help="model directory", type="string", default="rl_prtr")
    parser.add_option("--rl_load", dest="rl_load",
                      help="whether to load rl model to continue learning", type="int", default=0)
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
        _, opt_state, _, _, duration, call_counts = \
            opt.random_search(
                iteration_limit=int(9e30),  # never stop until wall_time_limt
                wall_time_limit=kwargs.wall_time_limit,
            )
        start_x_o = prob_env.fixed_xo
        start_x_p = None   # meaningless in random method
        opt_x_p = opt_state[prob_env.k:-1]   # exclude the step
        # use noiseless output
        opt_val = prob_env.still(
            prob_env.output_noiseless(opt_state)
        )
        wall_time_limit = kwargs.wall_time_limit
        generation = call_counts   # for random search, generation means call counts
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
        opt_val, start_x_o, opt_x_p, start_x_p, duration = extract_rl_exp_result(opt, prob_env)
        call_counts = opt.function_call_counts_training()
        wall_time_limit = opt.learn_wall_time_limit
        # for RL, generation means learning iteration
        generation = opt.get_learn_iteration()
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
                load=bool(kwargs.rl_load),
                tensorboard_path=tensorboard_path,
                logger_path=logger_path,
                memory_capacity=300000,
                memory_capacity_start_learning=1000,
                learn_wall_time_limit=kwargs.wall_time_limit,
                prioritized=True,
                trial_size=prob_env.d,
        )
        # opt loads the environment which has not fixed x_o.
        # But it will learn based on fixed x_o.
        assert not opt.get_env_if_set_fixed_xo()
        opt.set_env_fixed_xo(prob_env.x_o)
        assert opt.get_env_if_set_fixed_xo()

        EPISODE_SIZE = int(9e30)  # run until hitting wall time limit
        TEST_PERIOD = 100
        p1 = Process(target=rl_collect_samples,
                     args=[opt, EPISODE_SIZE, TEST_PERIOD])
        p1.start()
        p2 = Process(target=rl_learn,
                     args=[opt])
        p2.start()
        p1.join()
        p2.join()
        opt_val, start_x_o, opt_x_p, start_x_p, duration = extract_rl_exp_result(opt, prob_env)
        # for rl method, duration should be training time
        duration = kwargs.wall_time_limit
        wall_time_limit = kwargs.wall_time_limi
        call_counts = opt.function_call_counts_training()
        # for RL, generation means learning iteration
        generation = opt.get_learn_iteration()
    elif kwargs.method == 'rbf':
        # problem environment has not fixed x_o.
        # however, we want to fix x_o for rbf method
        assert not prob_env.if_set_fixed_xo()
        prob_env.set_fixed_xo(prob_env.x_o)
        assert prob_env.if_set_fixed_xo()

        call_counts = 0

        def rbf_output(x_p):
            # barrier method to invalidate constrained violation
            if numpy.sum(x_p) != prob_env.d:
                out = numpy.abs(numpy.sum(x_p) - prob_env.d)
            else:
                # since rbf-opt is a minimizer, we need to make the output negative
                out = - prob_env.output(numpy.hstack((prob_env.x_o, x_p, 0)))  # the last one is step placeholder
                global call_counts
                call_counts += 1
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
        _, opt_x_p, _, _, _ = opt.optimize()
        start_x_o = prob_env.fixed_xo
        start_x_p = None    # meaningless in rbf method
        # noiseless opt_val
        opt_val = prob_env.still(
            prob_env.output_noiseless(numpy.hstack((start_x_o, opt_x_p, 0)))
        )
        duration = kwargs.wall_time_limit
        wall_time_limit = kwargs.wall_time_limit
        generation = opt.itercount
    elif kwargs.method == 'ga':
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
        hof_size = 10
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
        opt_x_p = numpy.array(opt_x_p, dtype=numpy.float)
        start_x_o = prob_env.fixed_xo
        start_x_p = None   # meaningless in genetic algorithm
        duration = time.time() - start_time
        wall_time_limit = kwargs.wall_time_limit
        generation = gen

    # also output opt_x_p non-zero idx
    opt_x_p_idx = numpy.nonzero(opt_x_p)[0]
    assert len(opt_x_p_idx) == prob_env.d    

    # output test results
    numpy.set_printoptions(linewidth=10000)
    test_result_path = os.path.join(kwargs.prob_env_dir, 'test_result.csv')
    # write header
    if not os.path.exists(test_result_path):
        with open(test_result_path, 'w') as f:
            line = "method, wall_time_limit, duration, generation, call_counts, opt_val, start_x_o, opt_x_p, start_x_p, opt_x_p_idx \n"
            f.write(line)
    # write data
    with open(test_result_path, 'a') as f:
        line = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".\
            format(kwargs.method, wall_time_limit, duration, generation, call_counts, opt_val, start_x_o, opt_x_p, start_x_p, opt_x_p_idx)
        f.write(line)


