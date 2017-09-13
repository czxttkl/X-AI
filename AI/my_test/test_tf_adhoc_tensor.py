"""
Check whether I can replace Tensorflow tensor adhoc
"""
import tensorflow as tf
import numpy
tf.set_random_seed(1234)
numpy.random.seed(1234)

n_features = 20

sess = tf.Session()

s = tf.placeholder(tf.float32, [n_features], name='s')
b = tf.get_variable('b1', [n_features], initializer=tf.constant_initializer(2.))
out = s + b

sess.run(tf.global_variables_initializer())

real_s = numpy.random.randn(n_features)

print('s with default b')
print(sess.run(out, feed_dict={s: real_s}))

print('')
print('s with adhoc b')
print(sess.run(out, feed_dict={s: real_s,
                               b: numpy.ones(n_features)}))

