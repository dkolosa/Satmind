from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf

class Actor:

    def __init__(self, features, n_actions, layer_1_nodes, layer_2_nodes, action_bound, tau, learning_rate, batch_size, name):

        self.tau = tau
        self.action_bound = action_bound
        self.features = features
        self.n_actions = n_actions
        self.layer_1_nodes = layer_1_nodes
        self.layer_2_nodes = layer_2_nodes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.name = name

        # create the actor network and target network
        # self.input, self.output, self.scaled_output = self.build_network(name)
        self.input, self.output, self.scaled_output = self.build_network_keras()
        self.network_parameters = tf.trainable_variables()
        self.target_input, self.target_output, self.target_scaled_output = self.build_network_keras()
        self.target_network_parameters = tf.trainable_variables()[len(self.network_parameters):]

        # This is retrieved from the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, n_actions])

        self.unnorm_actor_grad = tf.gradients(self.scaled_output, self.network_parameters, -self.action_gradient)
        self.actor_gradient = list(map(lambda x: tf.div(x, batch_size), self.unnorm_actor_grad))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.actor_gradient, self.network_parameters))

        self.update_target_network_parameters = [self.target_network_parameters[i].assign(tf.multiply(self.network_parameters[i], self.tau) +
                                                 tf.multiply(self.target_network_parameters[i], 1. - self.tau))
                                                 for i in range(len(self.target_network_parameters))]

        self.trainable_variables = len(self.network_parameters) + len(self.target_network_parameters)

    def build_network_keras(self):

        input = tf.keras.Input(shape=(self.features,))

        x = tf.keras.layers.Dense(self.layer_1_nodes,
                                  kernel_initializer=tf.random_uniform_initializer(-1/np.sqrt(self.layer_1_nodes),
                                                                                   1/np.sqrt(self.layer_1_nodes)),
                                  bias_initializer=tf.random_uniform_initializer(-1 / np.sqrt(self.layer_1_nodes),
                                                                                 1 / np.sqrt(self.layer_1_nodes)))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.LayerNormalization()(x)
        x = tf.nn.relu(x)

        x = tf.keras.layers.Dense(self.layer_2_nodes,
                                  kernel_initializer=tf.random_uniform_initializer(-1/np.sqrt(self.layer_2_nodes),
                                                                                   1/np.sqrt(self.layer_2_nodes)),
                                  bias_initializer=tf.random_uniform_initializer(-1/np.sqrt(self.layer_2_nodes),
                                                                                 1/np.sqrt(self.layer_2_nodes)))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.LayerNormalization()(x)
        x = tf.nn.relu(x)

        x = tf.keras.layers.Dense(128,
                                  kernel_initializer=tf.random_uniform_initializer(-1 / np.sqrt(128),
                                                                                   1 / np.sqrt(128)),
                                  bias_initializer=tf.random_uniform_initializer(-1 / np.sqrt(128),
                                                                                 1 / np.sqrt(128)))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.LayerNormalization()(x)
        x = tf.nn.relu(x)

        output = tf.keras.layers.Dense(self.n_actions, activation='tanh',  kernel_initializer=tf.random_uniform_initializer(-0.003,0.003))(x)
        scaled_output = tf.multiply(output, self.action_bound)

        return input, output, scaled_output

    def predict(self, state, sess):
        action = sess.run(self.scaled_output, {self.input: state})
        return action

    def predict_target(self, state, sess):
        action = sess.run(self.target_scaled_output, {self.target_input: state})
        return action

    def train(self, state, action_gradient, sess):
        sess.run(self.train_op, {self.input: state, self.action_gradient: action_gradient})

    def update_target_network(self, sess):
        """
        theta^u' <- tau*theta^u + (1-tau)theta^u'
        :return:
        """
        sess.run(self.update_target_network_parameters)


    def __str__(self):
        return (f'Actor neural Network:\n'
                f'Inputs: {self.features} \t Actions: {self.n_actions} \t Action bound: {self.action_bound}\n'
                f'Layer 1 nodes: {self.layer_1_nodes} \t layer 2 nodes: {self.layer_2_nodes}\n'
                f'learning rate: {self.learning_rate} \t target network update (tau): {self.tau}\n')

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{self.features!r}, {self.n_actions!r},{self.layer_1_nodes!r}, {self.layer_2_nodes!r},'
                f'{self.action_bound!r}, {self.tau!r}, {self.learning_rate!r}, {self.batch_size!r}, {self.name!r})')


class Critic:

    def __init__(self, n_features, n_actions, layer_1_nodes, layer_2_nodes, learning_rate, tau, name, actor_trainable_variables):

        self.tau = tau
        self.n_features = n_features
        self.n_actions = n_actions
        self.layer_1_nodes = layer_1_nodes
        self.layer_2_nodes = layer_2_nodes
        self.learning_rate = learning_rate
        self.name = name
        self.actor_trainable_variables = actor_trainable_variables

        # self.input, self.action, self.output  = self.build_network(name)
        self.input, self.action, self.output = self.build_network_keras()

        self.network_parameters = tf.trainable_variables()[actor_trainable_variables:]

        self.input_target, self.action_target, self.output_target = self.build_network_keras()
        self.target_network_parameters = tf.trainable_variables()[(len(self.network_parameters) + actor_trainable_variables):]

        self.q_value = tf.placeholder(tf.float32, shape=[None, 1])
        self.importance = tf.placeholder(tf.float32, shape=[None, 1])

        # self.target_q = tf.multiply(self.output_target, self.importance)

        self.update_target_network_parameters = \
            [self.target_network_parameters[i].assign(tf.multiply(self.network_parameters[i], self.tau) \
                                                  + tf.multiply(self.target_network_parameters[i], 1. - self.tau))
             for i in range(len(self.target_network_parameters))]

        # self.loss = tf.losses.mean_squared_error(self.output, self.q_value)

        self.error = self.output - self.q_value
        self.loss = tf.reduce_mean(tf.multiply(tf.square(self.error), self.importance))

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # the action-value gradient to be used be the actor network
        self.action_grad = tf.gradients(self.output, self.action)

    def build_network_keras(self):

        input = tf.keras.Input(shape=(self.n_features,))
        action = tf.keras.Input(shape=(self.n_actions,))

        x = tf.keras.layers.Dense(self.layer_1_nodes,
                                  kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(input)
        # x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = tf.keras.layers.concatenate([tf.keras.layers.Flatten()(x), action])
        x = tf.keras.layers.Dense(self.layer_2_nodes, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)

        output = tf.keras.layers.Dense(1,activation='linear', kernel_initializer=tf.random_uniform_initializer(-0.003,0.003))(x)

        return input, action, output

    def predict(self, state, action, sess):
        return sess.run(self.output, {self.input: state,
                                      self.action: action})

    def predict_target(self, state, action, sess):
        return sess.run(self.output_target, {self.input_target: state,
                                             self.action_target: action})

    def train(self, state, action, q_value, importance, sess):
        return sess.run([self.error, self.output, self.train_op], {self.input: state,
                                                       self.action: action,
                                                       self.q_value: q_value,
                                                       self.importance: importance})

    def action_gradient(self, state, action, sess):
        return sess.run(self.action_grad, {self.input: state,
                                           self.action: action})

    def update_target_network(self, sess):
        """
        theta^Q' <- tau*theta^Q + (1-tau)theta^Q'
        where theta are all of the weights of the target network
        :return:
        """
        sess.run(self.update_target_network_parameters)

    def __str__(self):
        return (f'Critic Neural Network:\n'
                f'Inputs: {self.n_features} \t Actions: {self.n_actions}\n'
                f'Layer 1 nodes: {self.layer_1_nodes} \t layer 2 nodes: {self.layer_2_nodes}\n'
                f'learning rate: {self.learning_rate} \t target network update (tau): {self.tau}')

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{self.n_features!r}, {self.n_actions!r},{self.layer_1_nodes!r}, {self.layer_2_nodes!r},'
                f'{self.learning_rate!r}, {self.tau!r}, {self.name!r}, {self.actor_trainable_variables!r})')



