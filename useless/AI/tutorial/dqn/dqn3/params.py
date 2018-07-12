import os
import argparse

DEFAULT_EPISODES = 2000
DEFAULT_STEPS = 1000000
DEFAULT_STEPS_TO_UPDATE = 2000
DEFAULT_ENVIRONMENT = 'AirRaid-ram-v0'
DEFAULT_SKIPPING_CONST = 4

DEFAULT_MEMORY_CAPACITY = 10000
DEFAULT_EPSILON = 0.1
DEFAULT_GAMMA = 0.9
DEFAULT_MINI_BATCH_SIZE = 16

DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_REGULARIZATION = 0.001
DEFAULT_NUM_HIDDEN = 3
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_NUM_OBSERVATIONS = 4

DEFAULT_ID = 0

def parse_args():
  """
  Parses the command line input.

  """
  parser = argparse.ArgumentParser()
  parser.add_argument('-episodes', default = DEFAULT_EPISODES, help = 'number of episodes', type=int)
  parser.add_argument('-steps', default = DEFAULT_STEPS, help = 'number of steps', type=int)
  parser.add_argument('-update_every', default = DEFAULT_STEPS_TO_UPDATE, help = 'number of steps before updating the target network', type=int)
  parser.add_argument('-env', default = DEFAULT_ENVIRONMENT, help = 'environment name', type=str)
  parser.add_argument('-id', default = DEFAULT_ID, help = 'id number of run to append to output file name', type=str)
  parser.add_argument('-skipping', default = DEFAULT_SKIPPING_CONST, help = 'the number of frames to skip', type=int)


  parser.add_argument('-capacity', default = DEFAULT_MEMORY_CAPACITY, help = 'memory capacity', type=int)
  parser.add_argument('-epsilon', default = DEFAULT_EPSILON, help = 'epsilon value for the probability of taking a random action', type=float)
  parser.add_argument('-gamma', default = DEFAULT_GAMMA, help = 'gamma value for the contribution of the Q function in learning', type=float)
  parser.add_argument('-minibatch_size', default = DEFAULT_MINI_BATCH_SIZE, help = 'mini batch size for training', type=int)

  parser.add_argument('-l', default = DEFAULT_LEARNING_RATE, help = 'learning rate', type=float)
  parser.add_argument('-r', default = DEFAULT_REGULARIZATION, help = 'regularization', type=float)
  parser.add_argument('-num_hidden', default = DEFAULT_NUM_HIDDEN, help = 'the number of hidden layers in the deep network', type=int)
  parser.add_argument('-hidden_size', default = DEFAULT_HIDDEN_SIZE, help = 'the hidden size of all layers in the network', type=int)
  parser.add_argument('-num_observations', default = DEFAULT_NUM_OBSERVATIONS, help = 'the number of observations to pass to the network', type=int)


  args = parser.parse_args()

  run_id = "lr_" + str(args.l) + "_reg_" + str(args.r) + "_h_" + str(args.hidden_size) + "_m_" + str(args.minibatch_size) + "_c_" + str(args.capacity) + "_id_" + str(args.id)

  agent_params = {'episodes': args.episodes, 'steps': args.steps, 'steps_to_update': args.update_every, 'environment': args.env, 'run_id': run_id, 'skipping': args.skipping}
  dqn_params = {'memory_capacity': args.capacity, 'epsilon': args.epsilon, 'gamma': args.gamma, 'mini_batch_size': args.minibatch_size}
  cnn_params = {'lr': args.l, 'reg': args.r, 'num_hidden': args.num_hidden, 'hidden_size': args.hidden_size, 'mini_batch_size': args.minibatch_size, 'num_observations': args.num_observations}

  return agent_params, dqn_params, cnn_params
