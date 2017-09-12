"""
Tensorflow model save&restore tutorial

create a graph and put variables only related to this model onto the graph
saver and session only work on the session
"""
import tensorflow as tf
import os


class Model:
    def __init__(self, save_and_load_path, load=False):
        self.save_and_load_path = save_and_load_path

        # create a graph for model variables and session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        if not load:
            # create variables & initialize
            with self.graph.as_default():
                w_initializer = tf.random_normal_initializer(0., 0.3)
                self.w1 = tf.get_variable('w1', [2, 3], initializer=w_initializer)
                self.w2 = tf.get_variable('w2', [2, 3], initializer=w_initializer)
                self.w3 = tf.add(self.w1, self.w2, 'w3')
                self.sess.run(tf.global_variables_initializer())
        else:
            self.load()

    def operate_plus(self):
        print('w1:')
        print(self.sess.run(self.w1))
        print('w2:')
        print(self.sess.run(self.w2))
        print('w3:')
        print(self.sess.run(self.w3))

    def save(self):
        with self.graph.as_default():
            saver = tf.train.Saver()
            path = saver.save(self.sess, self.save_and_load_path)
            print('save model to', path)

    def load(self):
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(self.save_and_load_path + '.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(self.save_and_load_path)))
            self.w1 = self.graph.get_tensor_by_name("w1:0")
            self.w2 = self.graph.get_tensor_by_name("w2:0")
            self.w3 = self.graph.get_tensor_by_name("w3:0")


if __name__ == '__main__':
    m1 = Model(save_and_load_path='optimizer_model/qlearning', load=False)
    m1.operate_plus()
    m1.save()

    m2 = Model(save_and_load_path='optimizer_model/qlearning', load=True)
    m2.operate_plus()
    m2.save()

    m3 = Model(save_and_load_path='optimizer_model/qlearning', load=True)
    m3.operate_plus()


