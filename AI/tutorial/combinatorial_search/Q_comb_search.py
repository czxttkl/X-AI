"""
Use Q-learning to maximize a function, which is feed-forward neural network
"""

import numpy
import tensorflow as tf
from QLearning import QLearning
import time
from multiprocessing import Process, freeze_support
from multiprocessing.managers import BaseManager


# Raw parameters
k = 20   # total available card size
d = 6    # deck size
USE_PRIORITIZED_REPLAY = False
gamma = 0.9
n_hidden_ql = 400                 # number of hidden units in Qlearning NN
BATCH_SIZE = 64
MEMORY_CAPACITY = 300000
MEMORY_CAPACITY_START_LEARNING = 10000
EPISODE_SIZE = 10000001          # the size of training episodes
TEST_PERIOD = 2000                 # how many per training episodes to do testing
RANDOM_SEED = 206               # seed for random behavior except coefficient generation
LOAD = False                     # whether to load existing model
PLANNING = False                 # whether to use planning
MODEL_SAVE_ITERATION = 5000
LEARN_CPU_TIME_LIMIT = 3600000      # seconds of limit of cpu time the algorithm can learn
env_name = 'env_nn_xo'
# Read parameters
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--timed', dest='timed', action='store_true')
# parser.set_defaults(timed=timed)
# args = parser.parse_args()
# timed = args.timed

# Derived parameters
TRIAL_SIZE = d                # how many card modification allowed
tensorboard_path = 'comb_search_k{}_d{}_t{}/{}'.format(k, d, LEARN_CPU_TIME_LIMIT, time.time())
model_save_load_path = 'comb_search_k{}_d{}_t{}/optimizer_model/qlearning'.format(k, d, LEARN_CPU_TIME_LIMIT)
numpy.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


if __name__ == '__main__':
    # for multiprocessing on windows
    freeze_support()

    wall_time = time.time()

    # initialize critical components
    BaseManager.register('QLearning', QLearning)
    manager = BaseManager()
    manager.start()
    RL = manager.QLearning(
        k=k, d=d, env_name=env_name, n_hidden=n_hidden_ql, load=LOAD,
        memory_capacity=MEMORY_CAPACITY, prioritized=USE_PRIORITIZED_REPLAY, planning=PLANNING, batch_size=BATCH_SIZE,
        save_and_load_path=model_save_load_path, reward_decay=gamma, tensorboard_path=tensorboard_path,
        save_model_iter=MODEL_SAVE_ITERATION,
    )


    def collect_samples(RL):
        RL.collect_samples(EPISODE_SIZE, TRIAL_SIZE, MEMORY_CAPACITY_START_LEARNING, TEST_PERIOD, RANDOM_SEED,
                           LEARN_CPU_TIME_LIMIT)


    def learn(RL):
        RL.learn(MEMORY_CAPACITY_START_LEARNING, LEARN_CPU_TIME_LIMIT)


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


