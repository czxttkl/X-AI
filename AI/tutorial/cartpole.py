"""
https://gym.openai.com/evaluations/eval_PgCVjqrTCObX2gHpEQdeQ
https://github.com/ankitdhall/CartPole/blob/master/Qlearning-linear.py
https://github.com/dennybritz/reinforcement-learning/blob/master/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb
https://github.com/yenchenlin/DeepLearningFlappyBird
"""

####################################################################
#  Reinforcement learning agent with linear approimate Q-function  #
####################################################################
import gym
import numpy as np
import random
import math
import tensorflow as tf

NUM_EPISODES = 1000
MAX_T = 1000
GAMMA = 1

EXPLORATION_RATE = 0.5
EXPLORATION_RATE_DECAY = 0.5

env = gym.make('CartPole-v0')

NUM_ACTIONS = env.action_space.n
NUM_OBS = 4

#################### TensorFlow for linear model #####################

session = tf.Session()
state_ = tf.placeholder("float", [None, NUM_OBS])
targets = tf.placeholder("float", [None, NUM_ACTIONS])

hidden_weights = tf.Variable(
    tf.random_uniform(shape=[NUM_OBS, NUM_ACTIONS], minval=-0.0001, maxval=0.0001, dtype=tf.float32))

output = tf.matmul(state_, hidden_weights)

loss = tf.reduce_mean(tf.square(output - targets))
train_operation = tf.train.AdamOptimizer(1).minimize(loss)

session.run(tf.initialize_all_variables())

######################################################################

STOP_TRAIN = False
DEBUG_MODE = False

avg = 0

# to store samples for training
X = []
y = []

score_100 = [0 for i in range(100)]
score_ptr = 0
accumulated_sample = 0


def get_action(state, on=True):
    p = np.random.uniform(0, 1)
    # epsilon greedy action
    if p < EXPLORATION_RATE and on == True:
        return np.random.uniform(0, 1, (1, 2))

    # action according to policy
    reward = session.run(output, feed_dict={state_: [state]})
    return reward


# prepare for update given s, s' and r
def update(state, state_prime, r):
    reward = get_action(state, on=False)[0]
    reward_prime = get_action(state_prime, on=False)[0]

    q_prime = max(reward_prime)
    action_prime = np.argmax(reward_prime)

    retval = []
    for i in range(NUM_ACTIONS):
        if i == action_prime:
            retval.append(r + GAMMA * q_prime)
        else:
            retval.append(reward[i])

    X.append(state)
    y.append(retval)


# begin RL
for episode in range(NUM_EPISODES):
    obs = env.reset()
    state = obs

    for t in range(MAX_T):
        # get reward accoding to policy
        rewards_from_action = get_action(state)[0]
        # argmax{a} among rewards_from_action
        action = np.argmax(rewards_from_action)

        # take step with action
        obs, r, done, _ = env.step(action)

        state_prime = obs

        if not STOP_TRAIN:
            update(state, state_prime, r)

        state = state_prime

        accumulated_samples += 1

        if done or t == MAX_T - 1:
            if DEBUG_MODE:
                print("Episode %d completed in %d" % (episode, t))
            avg += t

            score_100[score_ptr] = t
            score_ptr = (score_ptr + 1) % 100

    if not STOP_TRAIN:
        session.run(train_operation, feed_dict={state_: X, targets: y})
        X, y = [], []

    if episode % 50 == 0:
        EXPLORATION_RATE *= EXPLORATION_RATE_DECAY

    if episode % 100 == 0:
        print("At %d episodes" % (episode))
        print
        "EXPLORATION_RATE:", EXPLORATION_RATE
        print
        "Average of 100:", avg / 100.0
        print
        session.run(hidden_weights)
        print
        "\n"
        avg = 0

    if sum(score_100) / 100.0 >= 195:
        print("Completed in %d episodes with score of %f" % (episode, sum(score_100) / 100.0))
        break
