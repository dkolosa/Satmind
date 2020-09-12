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
    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v2', 'LunarLanderContinuous-v2',
            'BipedalWalkerHardcore-v2')

    ENV = ENVS[0]
    env = gym.make(ENV)
    iter_per_episode = 200
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]
    action_bound = env.action_space.high

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 1001

    batch_size = 64
    #Pendulum
    layer_1_nodes, layer_2_nodes = 32, 64
    #lander

    tau = 0.001
    actor_lr, critic_lr = 0.0001, 0.001
    GAMMA = 0.99
    ep = 0.001

    actor = Actor(n_action, layer_1_nodes, layer_2_nodes)
    critic = Critic(layer_1_nodes, layer_2_nodes)

    actor_target = Actor(n_action, layer_1_nodes, layer_2_nodes)
    critic_target = Critic(layer_1_nodes, layer_2_nodes)

    actor_noise = OrnsteinUhlenbeck(np.zeros(n_action))

    actor.compile(optimizer=Adam(learning_rate=actor_lr))
    critic.compile(optimizer=Adam(learning_rate=critic_lr))

    PER = False
    if PER:
        memory = Per_Memory(capacity=100000)
    else:
        memory = Uniform_Memory(buffer_size=100000)

    for i in range(num_episodes):
        s = env.reset()
        sum_reward = 0
        sum_q = 0
        j = 0

        while True:
            env.render()

            a = actor(tf.convert_to_tensor([s], dtype=tf.float32))
            s1, r, done, _ = env.step(tf.squeeze(a, 1) + actor_noise())

            # Store in replay memory
            if PER:
                error = abs(r + ep)  # D_i = max D
                memory.add(error, (
                np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
            else:
                memory.add(
                    (np.reshape(s, (n_state,)), np.reshape(a[0], (n_action,)), r, np.reshape(s1, (n_state,)), done))

            # sample from memory
            if batch_size < memory.get_count:
                mem = memory.sample(batch_size)
                s_rep = tf.convert_to_tensor(np.array([_[0] for _ in mem]), dtype=tf.float32)
                a_rep = tf.convert_to_tensor(np.array([_[1] for _ in mem]), dtype=tf.float32)
                r_rep = tf.convert_to_tensor(np.array([_[2] for _ in mem]), dtype=tf.float32)
                s1_rep = tf.convert_to_tensor(np.array([_[3] for _ in mem]), dtype=tf.float32)
                d_rep = tf.convert_to_tensor(np.array([_[4] for _ in mem]), dtype=tf.float32)

            # if batch_size < memory.count:
            #     mem, idxs, isweight = memory.sample(batch_size)
            #     s_rep = np.array([_[0] for _ in mem])
            #     a_rep = np.array([_[1] for _ in mem])
            #     r_rep = np.array([_[2] for _ in mem])
            #     s1_rep = np.array([_[3] for _ in mem])
            #     d_rep = np.array([_[4] for _ in mem])

                with tf.GradientTape() as crit_tape:
                    targ_actions = actor_target(s1_rep)
                    target_q = tf.squeeze(critic_target(s1_rep, targ_actions), 1)
                    q = tf.squeeze(critic(s_rep, a_rep), 1)
                    sum_q += np.amax(q)
                    y_i = r_rep + GAMMA * target_q*(1-d_rep)
                    critic_loss = tf.keras.losses.MSE(y_i, q)

                critic_gradient = crit_tape.gradient(critic_loss, critic.trainable_variables)
                critic.optimizer.apply_gradients(zip(critic_gradient, critic.trainable_variables))

                with tf.GradientTape() as act_tape:
                    actor_policy = actor(s_rep)
                    actor_loss = -critic(s_rep, actor_policy)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                actor_gradient = act_tape.gradient(actor_loss, actor.trainable_variables)
                actor.optimizer.apply_gradients(zip(actor_gradient, actor.trainable_variables))

                # update target network
                update_target_network(actor, actor_target, tau)
                update_target_network(critic, critic_target, tau)

                # # update the critic network
                # error, predicted_q, _ = critic.fit(s_rep, a_rep, np.reshape(y_i, (batch_size, 1)),
                #                                      np.reshape(isweight, (batch_size, 1)), sess)
                # loss = mean_squared_error(y_i, critic.predict(s_rep, a_rep))
                # opt = Adam(loss)
                # for n in range(batch_size):
                #     idx = idxs[n]
                #     memory.update(idx, abs(error[n][0]))

                # sum_q += np.amax(predicted_q)


                # update actor policy
                # a_output = actor.predict(s_rep, sess)
                # grad = critic.action_gradient(s_rep, a_output, sess)
                # actor.train(s_rep, grad[0], sess)

            sum_reward += r
            s = s1
            j += 1
            if done:
                print('Episode: {}, reward: {}, q_max: {}'.format(i, int(sum_reward), sum_q))
                # rewards.append(sum_reward)
                print('===========')
                break


def loss_fn(y_true, y_pred, importance):
    error = tf.math.square(y_true, y_pred)
    return tf.reduce_mean(tf.math.multiply(error, importance))


def update_target_network(network_params, target_network_params, tau=.001):
    update = []
    targets = target_network_params.weights
    for i, weight in enumerate(network_params.weights):
        update.append(weight*tau + targets[i]*(1-tau))
    network_params.set_weights(update)



if __name__ == '__main__':
    test_rl()