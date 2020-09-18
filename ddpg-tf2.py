import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import gym
import gym.spaces
# import matplotlib.pyplot as plt
from Satmind.tf2model import Critic, Actor
from Satmind.utils import OrnsteinUhlenbeck, AdaptiveParamNoiseSpec
from Satmind.replay_memory import Per_Memory, Uniform_Memory


def test_rl():
    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v3', 'LunarLanderContinuous-v2',
            'BipedalWalkerHardcore-v3')

    ENV = ENVS[2]
    env = gym.make(ENV)
    iter_per_episode = 200
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]
    action_bound = env.action_space.high

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 1001

    batch_size = 128
    #Pendulum
    layer_1_nodes, layer_2_nodes = 512, 256
    #lander

    tau = 0.01
    actor_lr, critic_lr = 0.0001, 0.001
    GAMMA = 0.99
    ep = 0.001

    actor = Actor(n_action, action_bound, layer_1_nodes, layer_2_nodes)
    critic = Critic(layer_1_nodes, layer_2_nodes)

    actor_target = Actor(n_action, action_bound, layer_1_nodes, layer_2_nodes)
    critic_target = Critic(layer_1_nodes, layer_2_nodes)

    actor_noise = OrnsteinUhlenbeck(np.zeros(n_action))

    actor.compile(optimizer=Adam(learning_rate=actor_lr))
    critic.compile(optimizer=Adam(learning_rate=critic_lr))

    PER = False
    if PER:
        memory = Per_Memory(capacity=100000)
    else:
        memory = Uniform_Memory(buffer_size=100000)

    update_target_network(actor, actor_target, tau)
    update_target_network(critic, critic_target, tau)

    for i in range(num_episodes):
        s = env.reset()
        sum_reward = 0
        sum_q = 0
        j = 0

        while True:
            env.render()

            a = actor(tf.convert_to_tensor([s], dtype=tf.float32))[0] + actor_noise()
            s1, r, done, _ = env.step(a)

            # Store in replay memory
            if PER:
                error = abs(r + ep)  # D_i = max D
                memory.add(error, (
                np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
            else:
                memory.add(
                    (np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))

            # sample from memory
            if batch_size < memory.get_count:
                mem = memory.sample(batch_size)
                s_rep = tf.convert_to_tensor(np.array([_[0] for _ in mem]), dtype=tf.float32)
                a_rep = tf.convert_to_tensor(np.array([_[1] for _ in mem]), dtype=tf.float32)
                r_rep = tf.convert_to_tensor(np.array([_[2] for _ in mem]), dtype=tf.float32)
                s1_rep = tf.convert_to_tensor(np.array([_[3] for _ in mem]), dtype=tf.float32)
                d_rep = tf.convert_to_tensor(np.array([_[4] for _ in mem]), dtype=tf.float32)

                loss_critic(GAMMA, a_rep, actor_target, critic, critic_target, d_rep, r_rep, s1_rep, s_rep)

                loss_actor(actor, critic, s_rep)

                # update target network
                update_target_network(actor, actor_target, tau)
                update_target_network(critic, critic_target, tau)

            sum_reward += r
            s = s1
            j += 1
            if done:
                print('Episode: {}, reward: {}, q_max: {}'.format(i, int(sum_reward), sum_q))
                # rewards.append(sum_reward)
                print('===========')
                break


@tf.function
def loss_actor(actor, critic, s_rep):
    with tf.GradientTape() as tape:
        actions = actor(s_rep)
        actor_loss = -tf.reduce_mean(critic(s_rep, actions))
    actor_grad = tape.gradient(actor_loss, actor.trainable_variables)  # compute actor gradient
    actor.optimizer.apply_gradients(zip(actor_grad, actor.trainable_variables))


@tf.function
def loss_critic(GAMMA, a_rep, actor_target, critic, critic_target, d_rep, r_rep, s1_rep, s_rep):
    targ_actions = actor_target(s1_rep)
    target_q = tf.squeeze(critic_target(s1_rep, targ_actions), 1)
    y_i = r_rep + GAMMA * target_q * (1 - d_rep)
    with tf.GradientTape() as tape:
        q = tf.squeeze(critic(s_rep, a_rep), 1)
        td_error = q - y_i
        critic_loss = tf.math.reduce_mean(tf.math.square(td_error))
    critic_gradient = tape.gradient(critic_loss, critic.trainable_variables)
    critic.optimizer.apply_gradients(zip(critic_gradient, critic.trainable_variables))


def update_target_network(network_params, target_network_params, tau=.001):
    weights = network_params.get_weights()
    target_weights = target_network_params.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_network_params.set_weights(target_weights)

if __name__ == '__main__':
    test_rl()