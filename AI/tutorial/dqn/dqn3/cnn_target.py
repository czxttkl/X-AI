import tensorflow as tf
import numpy as np
import logging


class CNNtarget:
    """
    Convolutional Neural Network model with trainable and target networks.

    """

    def __init__(self, num_actions, observation_shape, params={}, verbose=False):
        """
        Initialize the CNN model with a set of parameters.

        Args:
          params: a dictionary containing values of the models' parameters.

        """
        self.verbose = verbose
        self.num_actions = num_actions

        # observation shape will be a tuple
        self.observation_shape = observation_shape[0]
        logging.info('Initialized with params: {}'.format(params))

        self.lr = params['lr']
        self.reg = params['reg']
        self.num_hidden = params['num_hidden']
        self.hidden_size = params['hidden_size']
        self.num_observations = params['num_observations']

        self.input_placeholder, self.labels_placeholder, self.actions_placeholder = self.add_placeholders()

        self.session = self.start_session()

    def add_placeholders(self):
        """ Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of shape
                           (None, observation_shape), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape
                            (None,), type tf.float32
        actions_placeholder: Actions mask placeholder (None, self.num_actions),
                             type tf.float32

        """
        input_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_shape * self.num_observations))
        labels_placeholder = tf.placeholder(tf.float32, shape=(None,))
        actions_placeholder = tf.placeholder(tf.float32, shape=(None, self.num_actions))

        return input_placeholder, labels_placeholder, actions_placeholder

    def nn(self, input_obs):
        # Input layer
        W_input_shape = [self.observation_shape * self.num_observations, self.hidden_size]
        W_input = tf.get_variable("W_input", shape=W_input_shape)
        b_shape = [1, self.hidden_size]
        b_input = tf.get_variable("b_input", shape=b_shape, initializer=tf.constant_initializer(0.0))

        xW = tf.matmul(input_obs, W_input)
        h = tf.tanh(tf.add(xW, b_input))

        reg = tf.reduce_sum(tf.square(W_input))

        W_shape = [self.hidden_size, self.hidden_size]

        # Hidden layers
        for i in range(self.num_hidden):
            W = tf.get_variable("W" + str(i), shape=W_shape)
            b = tf.get_variable("b" + str(i), shape=b_shape, initializer=tf.constant_initializer(0.0))

            xW = tf.matmul(h, W)
            h = tf.tanh(tf.add(xW, b))

            reg += tf.reduce_sum(tf.square(W))

        # Output layer
        W_output_shape = [self.hidden_size, self.num_actions]
        W_output = tf.get_variable("W_output", shape=W_output_shape)
        b_output_shape = [1, self.num_actions]
        b_output = tf.get_variable("b_output", shape=b_output_shape, initializer=tf.constant_initializer(0.0))

        hU = tf.matmul(h, W_output)
        out = tf.add(hU, b_output)

        reg += tf.reduce_sum(tf.square(W_output))
        reg = self.reg * reg

        return out, reg

    # def conv2d(self, x, W, stride):
    #   return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    # def cnn(self, input_obs):
    #   # Input placeholer of shape: [batch, in_height, in_width, in_channels]

    #   # W shape is [filter_height, filter_width, in_channels, output_channels]
    #   W1shape = [2, 1, self.num_observations, self.num_actions]
    #   W1 = tf.get_variable("W1", shape=W1shape)
    #   bshape = [1, self.num_actions]
    #   b1 = tf.get_variable("b1", shape=bshape, initializer = tf.constant_initializer(0.0))

    #   out = self.conv2d(input_obs, W1, 1) + b1

    #   reg = self.reg * (tf.reduce_sum(tf.square(W1)))
    #   return out, reg


    def create_model_trainable(self):
        """
        The model definition.

        """
        outputs, reg = self.nn(self.input_placeholder)

        self.predictions = outputs

        self.q_vals = tf.reduce_sum(tf.multiply(self.predictions, self.actions_placeholder), 1)
        self.loss = tf.reduce_sum(tf.square(self.labels_placeholder - self.q_vals)) + reg

        # optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate = self.lr)

        self.train_op = optimizer.minimize(self.loss)

    def create_model_target(self):
        """
        The model definition for the target network.

        """
        outputs, reg = self.nn(self.input_placeholder)

        self.predictions_target = outputs

    def start_session(self):
        """
        Creates the session.

        """
        self.input_layer_mats = ["W_input", "b_input"]
        self.hidden_layer_mats = []
        for i in range(self.num_hidden):
            self.hidden_layer_mats.append("W" + str(i))
            self.hidden_layer_mats.append("b" + str(i))
        self.output_layer_mats = ["W_output", "b_output"]

        self.weight_mats = self.input_layer_mats + self.hidden_layer_mats + self.output_layer_mats

        with tf.variable_scope("network") as scope:
            self.create_model_trainable()

        with tf.variable_scope("target") as scope:
            self.create_model_target()

        init = tf.initialize_all_variables()

        session = tf.Session()
        session.run(init)

        return session

    def predict(self, observation):
        """
        Predicts the rewards for an input observation state from the target network.

        Args:
          observation: a numpy array of a single observation state

        """

        loss, prediction_probs = self.session.run(
            [self.loss, self.predictions],
            feed_dict={self.input_placeholder: observation,
                       self.labels_placeholder: np.zeros(len(observation)),
                       self.actions_placeholder: np.zeros((len(observation), self.num_actions))
                       })

        return prediction_probs

    def predict_target(self, observation):
        """
        Predicts the rewards for an input observation state from the target network.

        Args:
          observation: a numpy array of a single observation state

        """

        prediction_probs = self.session.run(
            [self.predictions_target],
            feed_dict={self.input_placeholder: observation,
                       self.labels_placeholder: np.zeros(len(observation)),
                       self.actions_placeholder: np.zeros((len(observation), self.num_actions))
                       })

        return prediction_probs

    def train_step(self, Xs, ys, actions):
        """
        Updates the CNN model with a mini batch of training examples.

        """

        loss, _, prediction_probs, q_values = self.session.run(
            [self.loss, self.train_op, self.predictions, self.q_vals],
            feed_dict={self.input_placeholder: Xs,
                       self.labels_placeholder: ys,
                       self.actions_placeholder: actions
                       })

    def target_update_weights(self):
        """
        Loads in the new network weights from a file.

        """

        for mat_name in self.weight_mats:
            with tf.variable_scope("network", reuse=True):
                network_mat = tf.get_variable(mat_name)

            with tf.variable_scope("target", reuse=True):
                target_mat = tf.get_variable(mat_name)

            assign_op = target_mat.assign(network_mat)

            net_mat, tar_mat, _ = self.session.run([network_mat, target_mat, assign_op])
