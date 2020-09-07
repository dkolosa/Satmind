import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
import numpy as np
import os


class Critic(tf.keras.Model):
    def __init__(self, layer_1, layer_2, name='Critic',
                 checkpoint_dir=''):
        super(Critic, self).__init__()
        self.layer_1 = Dense(layer_1, activation='relu')
        self.layer_2 = Dense(layer_2, activation='relu')
        self.q = Dense(1, activation=None)
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = os.path.join(self.checkpoint_dir, name+ '_ddpg.h5')

    def call(self, state, action):
        action = self.layer_1(tf.concat([state, action], axis=1))
        action = self.layers_2(action)
        q = self.q(action)
        return q


class Actor(tf.keras.Model):
    def __init__(self, n_actions, layer_1=64, layer_2=32, name='Actor',
                 checkpoint_dir=''):
        super(Actor, self).__init__()
        self.layer_1 = Dense(layer_1, activation='relu')
        self.layer_2 = Dense(layer_2, activation='relu')
        self.action = Dense(n_actions, activation='tanh')
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = os.path.join(self.checkpoint_dir, name+ '_ddpg.h5')

    def call(self, x):
        x = self.layer_1(x)
        x = tf.keras.layers.BatchNormalization(x)
        x = self.layer_2(x)
        x = tf.keras.layers.BatchNormalization(x)
        act = self.action(x)
        return act
