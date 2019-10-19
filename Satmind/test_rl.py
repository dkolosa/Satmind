import tensorflow as tf
import numpy as np
import gym
# import matplotlib.pyplot as plt

from actor_critic import Actor, Critic
from utils import OrnsteinUhlenbeck
from replay_memory import Per_Memory, Experience


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


def upadte_perturbed_actor(self,actor, perturbed_actor, param_noise):
    updates =[]
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            updates.append(perturbed_var.assign(var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise)))
        else:
            updates.append(perturbed_var.assign(var))
    return tf.group(*updates)


def setup_noise(actor, obs, sess):
    # Make a copy of the acotr policy (no noise)
    param_noise_actor = copy(actor)
    self.perturb_policy_op = self.upadte_perturbed_actor(actor, param_noise_actor, self.param_noise_stddev)

    # Configure separate copy for stddev adoption.
    adaptive_param_noise_actor = copy(actor)
    adaptive_actor_tf = adaptive_param_noise_actor(obs)
    self.perturb_adaptive_policy_ops = self.upadte_perturbed_actor(actor, adaptive_param_noise_actor,
                                                                   self.param_noise_stddev)

    self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(actor - adaptive_actor_tf)))

def test_rl():
    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v2', 'LunarLanderContinuous-v2')

    ENV = ENVS[0]
    env = gym.make(ENV)
    iter_per_episode = 200
    features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bound = env.action_space.high

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 5000
    batch_size = 64

    layer_1_nodes, layer_2_nodes = 400, 350
    tau = 0.001
    actor_lr, critic_lr = 0.0001, 0.001
    GAMMA = 0.99

    # Initialize actor and critic network and targets
    actor = Actor(features, n_actions, layer_1_nodes, layer_2_nodes, action_bound, tau, actor_lr, batch_size, 'actor')
    actor_noise = OrnsteinUhlenbeck(np.zeros(n_actions))
    critic = Critic(features, n_actions, layer_1_nodes, layer_2_nodes, critic_lr, tau, 'critic', actor.trainable_variables)
    PER = True

    # Replay memory buffer
    if PER:
        per_mem = Per_Memory(capacity=100000)
    else:
        replay = Experience(buffer_size=1000)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        actor.update_target_network(sess)
        critic.update_target_network(sess)

        for i in range(num_episodes):
            s = env.reset()
            sum_reward = 0
            sum_q = 0
            rewards = []
            j = 0

            #for j in range(iter_per_episode):
            while True:

                env.render()

                a = actor.predict(np.reshape(s, (1, features)), sess)
                s1, r, done, _ = env.step(a[0])

                rewards.append(r)
                # Store in replay memory
                if PER:
                    error = abs(r)  # D_i = max D
                    per_mem.add(error, (np.reshape(s, (features,)), np.reshape(a[0], (n_actions,)), r, np.reshape(s1, (features,)), done))
                else:
                    replay.add((np.reshape(s, (features,)), np.reshape(a[0], (n_actions,)), r, np.reshape(s1,(features,)), done))

                # sample from memory
                # if batch_size < replay.get_count:
                    # mem = replay.experience_replay(batch_size)
                    # s_rep = np.array([_[0] for _ in mem])
                    # a_rep = np.array([_[1] for _ in mem])
                    # r_rep = np.array([_[2] for _ in mem])
                    # s1_rep = np.array([_[3] for _ in mem])
                    # d_rep = np.array([_[4] for _ in mem])

                if batch_size < per_mem.count:
                    mem, idxs, isweight = per_mem.sample(batch_size)
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
                    error, predicted_q, _ = critic.train(s_rep, a_rep, np.reshape(y_i, (batch_size,1)), np.reshape(isweight, (batch_size,1)), sess)

                    for n in range(batch_size):
                        idx = idxs[n]
                        per_mem.update(idx, abs(error[n][0]))

                    sum_q += np.amax(predicted_q)
                    # update actor policy
                    a_output = actor.predict(s_rep, sess)
                    grad = critic.action_gradient(s_rep, a_output, sess)
                    actor.train(s_rep, grad[0], sess)

                    # update target networks
                    actor.update_target_network(sess)
                    critic.update_target_network(sess)
                # else:
                #     per_mem.add(error,(np.reshape(s, (features,)), np.reshape(a[0], (n_actions,)), r, np.reshape(s1, (features,)), done))

                sum_reward += r
                s = s1
                j += 1
                if done:
                    print('Episode: {}, reward: {}, Q_max: {}'.format(i, int(sum_reward), sum_q/float(j)))
                    print('===========')
                    break



if __name__ == '__main__':
    test_rl()
