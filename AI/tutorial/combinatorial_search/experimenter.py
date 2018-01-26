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
                      help="wall time limit", type="int", default=999)
    parser.add_option("--prob_env_dir", dest="prob_env_dir",
                      help="environment directory", type="string", default="test_probs/prob_env_nn_0")
    parser.add_option("--prtr_model_dir", dest="prtr_model_dir",
                      help="model directory", type="string", default="rl_prtr")
    # method:
    # 1. rl_prtr (pretraining)
    # 2. rl (no-pretraining)
    # 3. rbf
    # 4. random
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
        opt_x_p = opt_state[prob_env.k:]
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
            save_and_load_path=kwargs.prtr_model_dir,
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


