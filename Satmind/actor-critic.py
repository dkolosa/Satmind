import numpy as np
import tensorflow as tf


class Actor:

    def __init__(self, features, n_actions, layer_1_nodes):

        self.input = tf.placeholder(tf.float32, shape=[1, features])
        self.action = tf.placeholder(tf.float32, None, "action")
        self.td_error = tf.placeholder(tf.float32, shape=None)

        with tf.variable_scope('input'):
            layer_1 = tf.layers.dense(inputs=self.input,
                                      units=layer_1_nodes,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(0., .1),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      name='layer_1'
                                      )

        with tf.variable_scope('output'):
            self.action_prob = tf.layers.dense(inputs=layer_1,
                                               units=n_actions,
                                               activation=tf.nn.softmax,
                                               kernel_initializer=tf.random_normal_initializer(0.,.1),
                                               bias_initializer=tf.constant_initializer(0.1),
                                               name='action_probability'
                                               )

        with tf.variable_scope('exp_v'):
            log_probiblity = tf.log(self.action_prob)
            self.exp_v = tf.reduce_mean(log_probiblity * self.td_error)

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(-self.exp_v)

    def train(self, state, action, td, sess):
        state = state[np.newaxis, :]

        _, exp_v = sess.run([self.train_op, self.exp_v], {self.input: state,
                                                               self.action: action,
                                                               self.td_error: td})
        return exp_v

    def choose_action(self, state, sess):
        s = state[np.newaxis, :]
        probibility = sess.run(self.action_prob, {self.input:s})
        return np.random.choice(np.arange(probibility.shape[1]), p=probibility.ravel())


class Critic:

    def __init__(self, n_features, layer_1_nodes):

        self.state = tf.placeholder(tf.float32, shape=[1, n_features])
        self.v_prime = tf.placeholder(tf.float32, [1,1])
        self.reward = tf.placeholder(tf.float32, None)

        with tf.variable_scope('layer_1'):
            layer_1 = tf.layers.dense(inputs=self.state,
                                      units=layer_1_nodes,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(0., .1),
                                      bias_initializer=tf.constant_initializer(0.1)
                                      )

        with tf.variable_scope('output'):
            self.value = tf.layers.dense(inputs=layer_1,
                                         units=1,
                                         kernel_initializer=tf.random_normal_initializer(0., .1),
                                         bias_initializer=tf.constant_initializer(0.1)
                                         )

        with tf.variable_scope('squared_td_error'):
            self.td_error = self.reward + GAMMA * self.v_prime-self.value
            self.loss = tf.square(self.td_error)

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

    def train(self, state, reward, next_state, sess):

        v_prime = sess.run(self.value, {self.state: next_state})
        td_error, _ = sess.run([self.td_error, self.train_op],
                               feed_dict={self.state: state, self.v_prime: v_prime,
                                          self.reward: reward}
                               )
        return td_error
