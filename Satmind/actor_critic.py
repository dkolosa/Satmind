import numpy as np
import tensorflow as tf
import tflearn

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
        # self.is_train = tf.placeholder(tf.bool, name="is_train")


        # create the actor network and target network
        self.input, self.output, self.scaled_output = self.build_network(name)
        self.network_parameters = tf.trainable_variables()
        self.target_input, self.target_output, self.target_scaled_output = self.build_network(name='target_actor_')
        self.target_network_parameters = tf.trainable_variables()[len(self.network_parameters):]

        # This is retrieved from the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, n_actions])

        self.unnorm_actor_grad = tf.gradients(self.scaled_output, self.network_parameters, -self.action_gradient)
        self.actor_gradient = list(map(lambda x: tf.math.divide(x, batch_size), self.unnorm_actor_grad))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.actor_gradient, self.network_parameters))

        self.update_target_network_parameters = [self.target_network_parameters[i].assign(tf.multiply(self.network_parameters[i], self.tau) +
                                                 tf.multiply(self.target_network_parameters[i], 1. - self.tau))
         for i in range(len(self.target_network_parameters))]

        self.trainable_variables = len(self.network_parameters) + len(self.target_network_parameters)

    def build_network(self, name):
        input = tf.placeholder(tf.float32, shape=[None, self.features])
        with tf.variable_scope(str(name) + '_layer_1'):
            layer_1 = tf.layers.dense(inputs=input,
                                      units=self.layer_1_nodes,
                                      activation=tf.nn.relu,
                                      )
            # l1_batch = tf.layers.batch_normalization(layer_1, training=self.is_train)
            # l1_batch = tf.contrib.layers.layer_norm(layer_1)
            # l1_noise = self.gaussian_noise(l1_batch,stddev=0.2)

            # l1_act = tf.nn.relu(l1_batch)

        with tf.variable_scope(str(name) + '_layer_2'):
            layer_2 = tf.layers.dense(inputs=layer_1,
                                      units=self.layer_2_nodes,
                                      activation=tf.nn.relu,
                                      )
            # l2_batch = tf.layers.batch_normalization(layer_2, training=True)
            # l2_batch = tf.contrib.layers.layer_norm(layer_2)
            # l2_noise = self.gaussian_noise(l2_batch, stddev=0.2)
            # l2_act = tf.nn.relu(l2_batch)

        with tf.variable_scope(str(name) + '_output'):
            output = tf.layers.dense(inputs=layer_2,
                                          units=self.n_actions,
                                          activation=tf.nn.tanh,
                                          kernel_initializer=tf.random_uniform_initializer(-0.003,0.003),
                                     )

        scaled_output = tf.multiply(output, self.action_bound)

        return input, output, scaled_output

    def gaussian_noise(self, input, mean=0.0, stddev=0.2):
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=0.2, dtype=tf.float32)
        return input + noise

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

        self.input, self.action, self.output  = self.build_network(name)
        self.network_parameters = tf.trainable_variables()[actor_trainable_variables:]

        self.input_target, self.action_target, self.output_target = self.build_network(name='target_critic_')
        self.target_network_parameters = tf.trainable_variables()[(len(self.network_parameters) + actor_trainable_variables):]

        self.q_value = tf.placeholder(tf.float32, shape=[None, 1])
        self.importance = tf.placeholder(tf.float32, shape=[None, 1])

        self.update_target_network_parameters = \
            [self.target_network_parameters[i].assign(tf.multiply(self.network_parameters[i], self.tau) \
                                                  + tf.multiply(self.target_network_parameters[i], 1. - self.tau))
             for i in range(len(self.target_network_parameters))]

        # self.loss = tf.losses.mean_squared_error(self.output, self.q_value)

        self.error = self.output - self.q_value
        self.loss = tf.reduce_mean(tf.multiply(tf.square(self.error), self.importance))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # the action-value gradient to be used be the actor network
        self.action_grad = tf.gradients(self.output, self.action)

    def build_network(self, name):


        input = tf.placeholder(tf.float32, shape=[None, self.n_features])
        action = tf.placeholder(tf.float32, shape=[None, self.n_actions])

        # layer_1 = tf.contrib.layers.fully_connected(input, self.layer_1_nodes, activation_fn=tf.nn.relu)
        # l1_batch = tf.contrib.layers.layer_norm(layer_1)
        # l1_batch = tf.layers.batch_normalization(layer_1, training=True)
        # l1_act = tf.nn.relu(l1_batch)

        # layer_1 = tf.contrib.layers.fully_connected(input, self.layer_1_nodes)

        layer_1 = tf.layers.dense(inputs=input,
                                  units=self.layer_1_nodes,
                                  activation=tf.nn.relu,
                                  )
        # l1_batch = tf.contrib.layers.layer_norm(layer_1)
        # l1_act = tf.nn.relu(layer_1)

        t1 = tflearn.fully_connected(layer_1, self.layer_2_nodes)
        t2 = tflearn.fully_connected(action, self.layer_2_nodes)

        layer_2 = tf.nn.relu(tf.matmul(layer_1, t1.W) + tf.matmul(action, t2.W) + t2.b)
        # layer_2 = tf.contrib.layers.fully_connected(tf.concat((l1_batch, action), axis=1), self.layer_2_nodes)

        with tf.variable_scope(str(name) + '_output'):
            output = tf.layers.dense(inputs=layer_2,
                                     units=1,
                                     activation=None,
                                     use_bias=False,
                                     kernel_initializer=tf.random_uniform_initializer(-0.003,0.003)
                                     )

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
