"""
Supervise learning model for MC simulation.
This file is only used for CPU time test purpose
"""

import os
import numpy
import time
import psutil
import tensorflow as tf
from sklearn.neural_network import MLPRegressor

numpy.set_printoptions(linewidth=1000, threshold=100)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

k = 312
d = 15
n_hidden = 1000
num_trials = 300 * 15 * 15
x_o = numpy.zeros(k)
one_idx = numpy.random.choice(k, d, replace=False)
x_o[one_idx] = 1
random_states = numpy.zeros((num_trials, 2*k))

def get_cpu_time():
    cpu_time = psutil.Process().cpu_times()
    return cpu_time.user + cpu_time.system + cpu_time.children_system + cpu_time.children_user


duration = time.time()
cpu_time = get_cpu_time()
for i in range(num_trials):
    random_xp = numpy.zeros(k)
    one_idx = numpy.random.choice(k, d, replace=False)
    random_xp[one_idx] = 1
    random_state = numpy.hstack((x_o, random_xp))
    random_states[i, :] = random_state
duration = time.time() - duration
cpu_time = get_cpu_time() - cpu_time
print("collect {} samples need {} time and {} cpu time".format(num_trials, duration, cpu_time))


mlp = MLPRegressor(hidden_layer_sizes=(n_hidden, ), early_stopping=True, max_iter=1)
mlp.fit(random_states[:2, :], numpy.array([0, 1]))

duration = time.time()
cpu_time = get_cpu_time()
random_states_output = mlp.predict(random_states)
duration = time.time() - duration
cpu_time = get_cpu_time() - cpu_time
print("scikit predict {} samples need {} time and {} cpu time".format(num_trials, duration, cpu_time))
print("scikit output:", random_states_output)



graph = tf.Graph()
sess = tf.Session(graph=graph)
w_initializer, b_initializer = \
    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
with graph.as_default():
    with tf.variable_scope('eval_net'):
        s = tf.placeholder(tf.float32, [None, 2 * k], name='s')
        with tf.variable_scope('l1'):
            eval_w1 = tf.get_variable('w1', [2*k, n_hidden], initializer=w_initializer)
            eval_b1 = tf.get_variable('b1', [n_hidden], initializer=b_initializer)
            l1 = tf.nn.relu(tf.matmul(s, eval_w1) + eval_b1)
        with tf.variable_scope('l2'):
            eval_w2 = tf.get_variable('w2', [n_hidden, 1], initializer=w_initializer)
            eval_b2 = tf.get_variable('b2', [1], initializer=b_initializer)
            out = tf.matmul(l1, eval_w2) + eval_b2
    sess.run(tf.global_variables_initializer())

duration = time.time()
cpu_time = get_cpu_time()
random_states_output = sess.run(out, feed_dict={s: random_states}).flatten()
duration = time.time() - duration
cpu_time = get_cpu_time() - cpu_time
print("tf predict {} samples need {} time and {} cpu time".format(num_trials, duration, cpu_time))
print("tf output:", random_states_output)

#
#
# if random_state_output > max_pred_win_rate:
#     max_pred_win_rate = random_state_output
#     max_state = random_state
# duration = time.time() - duration
# cpu_time = self.get_cpu_time() - cpu_time
# max_real_win_rate = prob_env.still(prob_env.output(max_state))
# print("Trial: {}, duration: {}, cpu time: {}, max predict win rate: {}, real win rate: {}"
#       .format(num_trials, duration, cpu_time, max_pred_win_rate, max_real_win_rate))
