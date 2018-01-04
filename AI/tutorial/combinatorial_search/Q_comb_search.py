"""
Use Q-learning to maximize a function, which is feed-forward neural network
"""

import numpy
import tensorflow as tf
from QLearning import QLearning
import time
import argparse
from multiprocessing import Process, Array, Value
from multiprocessing.managers import BaseManager


# Raw parameters
k = 50   # total available card size
d = 30    # deck size
USE_PRIORITIZED_REPLAY = False
gamma = 0.9
n_hidden_ql = 200                 # number of hidden units in Qlearning NN
BATCH_SIZE = 64
MEMORY_CAPACITY = 300000
MEMORY_CAPACITY_START_LEARNING = 10000
EPISODE_SIZE = 10000001          # the size of training episodes
TEST_PERIOD = 200                 # how many per training episodes to do testing
RANDOM_SEED = 204               # seed for random behavior except coefficient generation
LOAD = False                     # whether to load existing model
PLANNING = False                 # whether to use planning
MODEL_SAVE_ITERATION = 500

# Read parameters
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--timed', dest='timed', action='store_true')
# parser.set_defaults(timed=timed)
# args = parser.parse_args()
# timed = args.timed

# Derived parameters
n_actions = d * (k-d) + 1    # number of one-card modification
TRIAL_SIZE = d                # how many card modification allowed
n_input_ql = k+1   # input dimension to qlearning network (k plus time step as a feature)
tensorboard_path = 'comb_search_k{0}_d{1}/{2}'.format(k, d, time.time())
model_save_load_path = 'comb_search_k{0}_d{1}/optimizer_model/qlearning'.format(k, d)
numpy.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


# initialize critical components
BaseManager.register('QLearning', QLearning)
manager = BaseManager()
manager.start()
RL = manager.QLearning(
    k=k, d=d, n_features=n_input_ql, n_actions=n_actions, n_hidden=n_hidden_ql, load=LOAD,
    memory_capacity=MEMORY_CAPACITY, prioritized=USE_PRIORITIZED_REPLAY, planning=PLANNING, batch_size=BATCH_SIZE,
    save_and_load_path=model_save_load_path, reward_decay=gamma, tensorboard_path=tensorboard_path,
    save_model_iter=MODEL_SAVE_ITERATION,
)


def collect_samples(RL):
    RL.collect_samples(EPISODE_SIZE, TRIAL_SIZE, MEMORY_CAPACITY_START_LEARNING, TEST_PERIOD, RANDOM_SEED)


def learn(RL):
    RL.learn(MEMORY_CAPACITY_START_LEARNING)


# collect samples and learning run in two separate processes
p1 = Process(target=collect_samples, args=[RL])
p1.start()
p2 = Process(target=learn, args=[RL])
p2.start()
p1.join()
p2.join()



