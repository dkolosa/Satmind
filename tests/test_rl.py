import tensorflow as tf
import numpy as np
import pytest
import gym

from Satmind.actor_critic import Actor, Critic


def test_training():
    """Test if training has taken place (update of weights)"""

    ENV = 'Pendulum-v0'
    env = gym.make(ENV)
    features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bound = env.action_space.high

    actor = Actor(features, n_actions, 128, 128, action_bound, 0.0001, .001,1, 'actor')
    critic = Critic(features, n_actions, 64, 64,0.001, 0.001,'critic', actor.trainable_variables)

    s = env.reset()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        before = actor.trainable_variables

        a = actor.predict(np.reshape(s, (1, features)), sess)
        s1, r, done, _ = env.step(a[0])

        grad = critic.action_gradient(s, a, sess)
        actor.train(s, grad[0], sess)

        after = actor.trainable_variables
        for b, a, n in zip(before, after):
        # Make sure something changed
            assert (b != a).any()


# def test_loss():
#   in_tensor = tf.placeholder(tf.float32, (None, 3))
#   labels = tf.placeholder(tf.int32, None, 1))
#   model = Model(in_tensor, labels)
#   sess = tf.Session()
#   loss = sess.run(model.loss, feed_dict={
#     in_tensor:np.ones(1, 3),
#     labels:[[1]]
#   })
# assert loss != 0

if __name__ == '__main__':
    test_training()
