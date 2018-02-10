"""
Use Q-learning to maximize a function, which is feed-forward neural network
"""

import numpy
import tensorflow as tf
from QLearning import QLearning
import time
from multiprocessing import Process, freeze_support
from multiprocessing.managers import BaseManager
import os
import optparse


def read_args(k, d, upr, test_period, random_seed, load, msi, lwtl, env_name, env_dir, fixed_xo, root_dir):
    parser = optparse.OptionParser(usage="usage: %prog [options]")
    parser.add_option("--k", dest="k", type="int")
    parser.add_option("--d", dest="d", type="int")
    parser.add_option("--use_prioritized_replay", dest="upr", type="int")
    parser.add_option("--test_period", dest="test_period", type="int")
    parser.add_option("--random_seed", dest="random_seed", type="int")
    parser.add_option("--load", dest="load", type="int")
    parser.add_option("--model_save_iteration", dest="msi", type="int")
    parser.add_option("--learn_wall_time_limit", dest="lwtl", type="int")
    parser.add_option("--env_name", dest="env_name", type="string")
    parser.add_option("--env_dir", dest="env_dir", type="string")   # only used if we want to load an environment
    parser.add_option("--fixed_xo", dest="fixed_xo", type="int")
    parser.add_option("--root_dir", dest="root_dir", type="string")
    (kwargs, args) = parser.parse_args()
    if kwargs.k:
        k = kwargs.k
    if kwargs.d:
        d = kwargs.d
    if kwargs.upr:
        upr = bool(kwargs.upr)
    if kwargs.test_period:
        test_period = kwargs.test_period
    if kwargs.random_seed:
        random_seed = kwargs.random_seed
    else:
        random_seed = None
    if kwargs.load:
        load = bool(kwargs.load)
    if kwargs.msi:
        msi = kwargs.msi
    if kwargs.lwtl:
        lwtl = kwargs.lwtl
    if kwargs.env_name:
        env_name = kwargs.env_name
    if kwargs.env_dir:
        assert kwargs.k and kwargs.d, "if loading from env_dir, you need to specify k and d in advance"
        env_dir = kwargs.env_dir
    if kwargs.fixed_xo:
        fixed_xo = bool(kwargs.fixed_xo)
    if kwargs.root_dir:
        root_dir = kwargs.root_dir

    return k, d, upr, test_period, random_seed, load, msi, lwtl, env_name, env_dir, fixed_xo, root_dir


def collect_samples(RL):
    RL.collect_samples(EPISODE_SIZE, TEST_PERIOD)


def learn(RL):
    RL.learn()


if __name__ == '__main__':
    # for multiprocessing on windows
    freeze_support()

    wall_time = time.time()

    # Raw parameters
    k = 312  # total available card size
    d = 30  # deck size
    USE_PRIORITIZED_REPLAY = True
    n_hidden_ql = 400  # number of hidden units in Qlearning NN
    MEMORY_CAPACITY = 300000
    MEMORY_CAPACITY_START_LEARNING = 1000
    EPISODE_SIZE = 10000001  # the size of training episodes
    TEST_PERIOD = 100  # how many per training episodes to do testing
    RANDOM_SEED = 208  # seed for random behavior except coefficient generation
    LOAD = False  # whether to load existing model
    MODEL_SAVE_ITERATION = 100
    LEARN_WALL_TIME_LIMIT = 50000  # seconds of limit of wall time the algorithm can learn
    env_name = 'env_nn'
    env_dir = ''                   # directory to load the environment
    fixed_xo = False               # whether to set x_o fixed in the environment
    root_dir = ''                  # the root directory to save model, tensorflow, logger, etc.
    # PLANNING = False             # whether to use planning

    # Derived parameters
    k, d, USE_PRIORITIZED_REPLAY, TEST_PERIOD, RANDOM_SEED, LOAD, \
        MODEL_SAVE_ITERATION, LEARN_WALL_TIME_LIMIT, env_name, env_dir, fixed_xo, root_dir = \
        read_args(k,
                  d,
                  USE_PRIORITIZED_REPLAY,
                  TEST_PERIOD,
                  RANDOM_SEED,
                  LOAD,
                  MODEL_SAVE_ITERATION,
                  LEARN_WALL_TIME_LIMIT,
                  env_name,
                  env_dir,
                  fixed_xo,
                  root_dir)
    numpy.set_printoptions(linewidth=10000)
    TRIAL_SIZE = d  # how many card modification allowed
    parent_path = os.path.join(root_dir,
                               'rl_prtr_{}_k{}_d{}_t{}'.format(env_name, k, d, LEARN_WALL_TIME_LIMIT))
    tensorboard_path = os.path.join(parent_path, str(time.time()))
    model_save_load_path = os.path.join(parent_path, 'optimizer_model_fixedxo{}'.format(fixed_xo), 'qlearning')
    logger_path = os.path.join(parent_path, 'logger_fixedxo{}.log'.format(fixed_xo))
    if fixed_xo:
        fixed_xo = numpy.zeros(k)
        fixed_xo[:d] = 1
    else:
        fixed_xo = None

    # initialize critical components
    BaseManager.register('QLearning', QLearning)
    manager = BaseManager()
    manager.start()
    RL = manager.QLearning(
        k=k,
        d=d,
        env_name=env_name,
        env_dir=env_dir,
        env_fixed_xo=fixed_xo,
        n_hidden=n_hidden_ql,
        save_and_load_path=model_save_load_path,
        load=LOAD,
        tensorboard_path=tensorboard_path,
        logger_path=logger_path,
        memory_capacity=MEMORY_CAPACITY,
        memory_capacity_start_learning=MEMORY_CAPACITY_START_LEARNING,
        learn_wall_time_limit=LEARN_WALL_TIME_LIMIT,
        prioritized=USE_PRIORITIZED_REPLAY,
        save_model_iter=MODEL_SAVE_ITERATION,
        trial_size=TRIAL_SIZE,
        random_seed=RANDOM_SEED,
    )

    print('original process', os.getpid())
    # collect samples and learning run in two separate processes
    p1 = Process(target=collect_samples, args=[RL])
    p1.start()
    p2 = Process(target=learn, args=[RL])
    p2.start()
    p1.join()
    p2.join()

    # log wall time
    wall_time = time.time() - wall_time
    RL.get_logger().log_wall_time(wall_time)


