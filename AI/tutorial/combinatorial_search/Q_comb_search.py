"""
Use Q-learning to maximize a function, which is feed-forward neural network
"""

import numpy
import tensorflow as tf
from QLearning import QLearning
import time
from tfboard import TensorboardWriter
import argparse

# Raw parameters
k = 20   # total available card size
d = 6    # deck size
use_prioritized_replay = True
gamma = 0.9
n_hidden_ql = 200                 # number of hidden units in Qlearning NN
BATCH_SIZE = 64
MEMORY_SIZE = 64000
MEMORY_SIZE_START_LEARNING = 64000
EPISODE_SIZE = 10000001          # the size of training episodes
TEST_PERIOD = 10                 # how many per training episodes to do testing
timed = True                     # whether including step as one feature
RANDOM_SEED = 2214               # seed for random behavior except coefficient generation
load = False                     # whether to load existing model

# Read parameters
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--timed', dest='timed', action='store_true')
parser.set_defaults(timed=timed)
args = parser.parse_args()
timed = args.timed

# Derived parameters
n_actions = d * (k-d) + 1    # number of one-card modification
TRIAL_SIZE = d                # how many card modification allowed
n_input_ql = k if not timed else k+1   # input dimension to qlearning network
numpy.random.seed(RANDOM_SEED)
if timed:
    from env_time import Environment
else:
    from env import Environment

# initialize critical components
env = Environment(k=k, d=d)
RL = QLearning(
    n_features=n_input_ql, n_actions=n_actions, n_hidden=n_hidden_ql, memory_size=MEMORY_SIZE, load=load,
    prioritized=use_prioritized_replay, batch_size=BATCH_SIZE, save_and_load_path='optimizer_model/qlearning',
    reward_decay=gamma, n_total_episode=EPISODE_SIZE, n_mem_size_learn_start=MEMORY_SIZE_START_LEARNING,
    all_possible_next_states_func=env.all_possible_next_states, step_state_func=env.step_state
)
tb_writer = TensorboardWriter(folder_name="comb_search_k{0}_d{1}/{2}".format(k, d, time.time()), session=RL.sess)


total_steps = RL.memory.size
for i_episode in range(EPISODE_SIZE):
    episode_steps = 0
    cur_state = env.reset()
    for i_epsisode_step in range(TRIAL_SIZE):
        next_possible_states, next_possible_actions = env.all_possible_next_state_action(cur_state)
        action, _ = RL.choose_action(cur_state, next_possible_states, next_possible_actions, epsilon_greedy=True)
        cur_state_, reward = env.step(action)
        terminal = True if i_epsisode_step == TRIAL_SIZE - 1 else False
        RL.store_transition(cur_state, action, reward, cur_state_, terminal)
        total_steps += 1
        cur_state = cur_state_
        if total_steps > MEMORY_SIZE_START_LEARNING:
            RL.learn()

    print('episode ', i_episode, ' finished with value', env.output(cur_state), 'cur_epsilon', RL.cur_epsilon())

    if total_steps > MEMORY_SIZE_START_LEARNING and i_episode % TEST_PERIOD == 0:
        cur_state = env.reset()
        for i_episode_test_step in range(TRIAL_SIZE):
            next_possible_states, next_possible_actions = env.all_possible_next_state_action(cur_state)
            action, q_val = RL.choose_action(cur_state, next_possible_states, next_possible_actions, epsilon_greedy=False)
            cur_state, reward = env.step(action)
            test_output = env.output(cur_state)
            print('TEST step {0}, output: {1}, at {2}, qval: {3}, reward {4}'.
                  format(i_episode_test_step, test_output, cur_state, q_val, reward))

        tb_writer.write(tags=['Prioritized={0}, gamma={1}, include_step={2}, seed={3}/Test Ending Output'.
                                format(use_prioritized_replay, gamma, timed, RANDOM_SEED),
                              'Prioritized={0}, gamma={1}, include_step={2}, seed={3}/Test Ending Qvalue'.
                                format(use_prioritized_replay, gamma, timed, RANDOM_SEED),
                              ],
                        values=[test_output,
                                q_val],
                        step=i_episode)





