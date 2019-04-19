import tensorflow as tf
import numpy as np
import gym

from Satmind.actor_critic import Actor, Critic
from Satmind.utils import OrnsteinUhlenbeck
from Satmind.replay_memory import Experience


def test_training():
    """Test if training has taken place (update of weights)"""

    ENV = 'Pendulum-v0'
    env = gym.make(ENV)
    features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bound = env.action_space.high

    actor = Actor(features, n_actions, 128, 128, action_bound, 0.0001, .001,1, 'actor')
    critic = Critic(features, n_actions, 128, 128, 0.001, 0.001,'critic', actor.trainable_variables)

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


def test_rl():
    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v2')

    ENV = ENVS[0]
    env = gym.make(ENV)
    iter_per_episode = 200
    features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bound = env.action_space.high

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 800
    batch_size = 64

    layer_1_nodes, layer_2_nodes = 300, 200
    tau = 0.001
    actor_lr, critic_lr = 0.0001, 0.001
    GAMMA = 0.99

    # Initialize actor and critic network and targets
    actor = Actor(features, n_actions, layer_1_nodes, layer_2_nodes, action_bound, tau, actor_lr, batch_size, 'actor')
    actor_noise = OrnsteinUhlenbeck(np.zeros(n_actions))
    critic = Critic(features, n_actions, layer_1_nodes, layer_2_nodes, critic_lr, tau, 'critic', actor.trainable_variables)

    # Replay memory buffer
    replay = Experience(buffer_size=500)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        actor.update_target_network(sess)
        critic.update_target_network(sess)

        for i in range(num_episodes):
            s = env.reset()
            sum_reward = 0
            sum_q = 0

            for j in range(iter_per_episode):

                env.render()

                a = actor.predict(np.reshape(s, (1, features)), sess) + actor_noise()
                s1, r, done, _ = env.step(a[0])

                # Store in replay memory
                replay.add((np.reshape(s, (features,)), np.reshape(a, (n_actions,)), r, np.reshape(s1,(features,)), done))
                # sample from random memory
                if batch_size < replay.get_count:
                    mem = replay.experience_replay(batch_size)
                    s_rep = np.array([_[0] for _ in mem])
                    a_rep = np.array([_[1] for _ in mem])
                    r_rep = np.array([_[2] for _ in mem])
                    s1_rep = np.array([_[3] for _ in mem])
                    d_rep = np.array([_[4] for _ in mem])

                    # Get q-value from the critic target
                    act_target = actor.predict_target(s1_rep, sess)
                    target_q = critic.predict_target(s1_rep, act_target, sess)

                    y_i = []
                    for x in range(batch_size):
                        if d_rep[x]:
                            y_i.append(r_rep[x])
                        else:
                            y_i.append(r_rep[x] + GAMMA * target_q[x])

                    # update the critic network
                    predicted_q, _ = critic.train(s_rep, a_rep, np.reshape(y_i, (batch_size,1)), sess)
                    sum_q += np.amax(predicted_q)
                    # update actor policy
                    a_output = actor.predict(s_rep, sess)
                    grad = critic.action_gradient(s_rep, a_output, sess)
                    actor.train(s_rep, grad[0], sess)

                    # update target networks
                    actor.update_target_network(sess)
                    critic.update_target_network(sess)

                sum_reward += r
                s = s1
                if done:
                    print('Episode: {}, reward: {}, Q_max: {}'.format(i, int(sum_reward), sum_q/float(j)))
                    print('===========')
                    break


if __name__ == '__main__':
    test_rl()