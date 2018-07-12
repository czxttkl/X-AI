import tensorflow as tf
import numpy


def t_mul():
    c_names = ['test_mul', tf.GraphKeys.GLOBAL_VARIABLES]
    w_initializer, b_initializer = \
        tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
    n_features = 4
    n_hidden = 5
    n_actions = 3

    s_ = tf.placeholder(tf.float32, [None, n_actions, n_features], name='s_')
    with tf.variable_scope('l1'):
        w1 = tf.get_variable('w1', [n_features, n_hidden], initializer=w_initializer, collections=c_names)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_initializer, collections=c_names)
        l1 = tf.einsum('ijk,kh->ijh', s_, w1) + b1
    with tf.variable_scope('l2'):
        w2 = tf.get_variable('w2', [n_hidden, 1], initializer=w_initializer, collections=c_names)
        b2 = tf.get_variable('b2', [1], initializer=b_initializer, collections=c_names)
        l2 = tf.einsum('ijh,ho->ijo', l1, w2) + b2

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    l1_out, l2_out = sess.run([l1, l2],
                              feed_dict={s_: numpy.ones((2, n_actions, n_features))})

    print(l1_out)
    print(l2_out)


if __name__ == '__main__':
    t_mul()

