import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
import numpy as np
import os


class Critic(tf.keras.Model):
    def __init__(self, layer_1, layer_2, model_name='Critic',
                 checkpoint_dir=''):
        super(Critic, self).__init__()
        self.layer_1 = Dense(layer_1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
        self.layer_2 = Dense(layer_2, activation=None,  kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
        self.bnl1 = tf.keras.layers.BatchNormalization()
        self.bnl2 = tf.keras.layers.BatchNormalization()

        self.q = Dense(1, activation=None)
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

    @tf.function
    def call(self, state, action):
        x = self.layer_1(tf.concat([state, action], axis=1))
        x = tf.keras.activations.relu(x)
        x = self.layer_2(x)
        x = tf.keras.activations.relu(x)
        q = self.q(x)
        return q


class Actor(tf.keras.Model):
    def __init__(self, n_actions, action_bound, layer_1=64, layer_2=32, model_name='Actor',
                 checkpoint_dir=''):
        super(Actor, self).__init__()
        self.layer_1 = Dense(layer_1, activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(-1 / np.sqrt(layer_1), 1 / np.sqrt(layer_1)),
            bias_initializer=tf.keras.initializers.RandomUniform(-1 / np.sqrt(layer_1), 1 / np.sqrt(layer_1))
                             )
        self.bnl1 = tf.keras.layers.BatchNormalization()
        # self.layer_1 = tf.keras.layers.Dense(layer_1, activation=None,
        #                           kernel_initializer=tf.random.uniform(-1 / np.sqrt(layer_1),
        #                                                                            1 / np.sqrt(layer_1)),
        #                           bias_initializer=tf.random.uniform(-1 / np.sqrt(layer_1), tf.float32,
        #                                                                          1 / np.sqrt(layer_1)))
        self.layer_2 = Dense(layer_2, activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(-1 / np.sqrt(layer_2), 1 / np.sqrt(layer_2)),
            bias_initializer=tf.keras.initializers.RandomUniform(-1 / np.sqrt(layer_2), 1 / np.sqrt(layer_2))
                             )
        # self.layer_2 = Dense(layer_2, activation=None)
        self.bnl2 = tf.keras.layers.BatchNormalization()
        self.action = Dense(n_actions, activation='tanh')
        self.action_bound = action_bound
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

    @tf.function
    def call(self, x):
        x = self.layer_1(x)
        x = tf.keras.activations.relu(x)
        x = self.bnl1(x)
        x = self.layer_2(x)
        x = tf.keras.activations.relu(x)
        x = self.bnl2(x)
        x = self.action(x) * self.action_bound
        return x
