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
        self.q = Dense(1, activation=None)
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

    def call(self, state, action):
        x = self.layer_1(state)
        x = tf.keras.activations.relu(x)
        x = self.layer_2(tf.concat([action, x], axis=1))
        x = tf.keras.activations.relu(x)
        q = self.q(x)
        return q


class Actor(tf.keras.Model):
    def __init__(self, n_actions, layer_1=64, layer_2=32, model_name='Actor',
                 checkpoint_dir=''):
        super(Actor, self).__init__()
        self.layer_1 = Dense(layer_1, activation=None)
        self.layer_2 = Dense(layer_2, activation=None)
        self.action = Dense(n_actions, activation='tanh')
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

    def call(self, x):
        x = self.layer_1(x)
        x = tf.keras.activations.relu(x)
        x = self.layer_2(x)
        x = tf.keras.activations.relu(x)
        x = self.action(x)
        return x
