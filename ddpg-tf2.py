import tensorflow as tf
from tensorflow.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
import numpy as np
import gym
import gym.spaces
import matplotlib.pyplot as plt
from Satmind.tf2model import Critic, Actor
from Satmind.utils import OrnsteinUhlenbeck, AdaptiveParamNoiseSpec
from Satmind.replay_memory import Per_Memory, Uniform_Memory



def test_rl():
    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v2', 'LunarLanderContinuous-v2',
            'BipedalWalkerHardcore-v2')

    ENV = ENVS[0]
    env = gym.make(ENV)
    iter_per_episode = 200
    features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bound = env.action_space.high

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 1001

    batch_size = 64
    #Pendulum
    layer_1_nodes, layer_2_nodes = 120, 64
    #lander

    tau = 0.001
    actor_lr, critic_lr = 0.0001, 0.001
    GAMMA = 0.99
    ep = 0.001

    actor = Actor(n_actions)
    critic = Critic()

    actor_target = Actor(n_actions)
    critic_target = Critic()

    actor_noise = OrnsteinUhlenbeck()

    actor.compile(optimizer=Adam(learning_rate=actor_lr))
    critic.compile(optimizer=Adam(learning_rate=critic_lr))

    PER = True
    if PER:
        memory = Per_Memory(capacity=100000)
    else:
        memory = Uniform_Memory(buffer_size=1000)

    for i in range(num_episodes):
        s = env.reset()
        sum_reward = 0
        sum_q = 0
        j = 0

        while True:
            env.render()

            a = actor.predict(s + actor_noise())
            s1, r, done, _ = env.step(a[0])

            # Store in replay memory
            if PER:
                error = abs(r + ep)  # D_i = max D
                memory.add(error, (
                np.reshape(s, (features,)), np.reshape(a[0], (n_actions,)), r, np.reshape(s1, (features,)), done))
            else:
                memory.add(
                    (np.reshape(s, (features,)), np.reshape(a[0], (n_actions,)), r, np.reshape(s1, (features,)), done))

            # sample from memory
            # if batch_size < memory.get_count:
            # mem = memory.sample(batch_size)
            # s_rep = np.array([_[0] for _ in mem])
            # a_rep = np.array([_[1] for _ in mem])
            # r_rep = np.array([_[2] for _ in mem])
            # s1_rep = np.array([_[3] for _ in mem])
            # d_rep = np.array([_[4] for _ in mem])

            if batch_size < memory.count:
                mem, idxs, isweight = memory.sample(batch_size)
                s_rep = np.array([_[0] for _ in mem])
                a_rep = np.array([_[1] for _ in mem])
                r_rep = np.array([_[2] for _ in mem])
                s1_rep = np.array([_[3] for _ in mem])
                d_rep = np.array([_[4] for _ in mem])

                # Get q-value from the critic target
                act_target = actor.predict(s1_rep)

                target_q = critic.predict(s1_rep, act_target)

                y_i = []
                for x in range(batch_size):
                    if d_rep[x]:
                        y_i.append(r_rep[x])
                    else:
                        y_i.append(r_rep[x] + GAMMA * target_q[x])

                # update the critic network
                error, predicted_q, _ = critic.fit(s_rep, a_rep, np.reshape(y_i, (batch_size, 1)),
                                                     np.reshape(isweight, (batch_size, 1)), sess)
                loss = mean_squared_error(y_i, critic.predict(s_rep, a_rep))
                opt = Adam(loss)
                for n in range(batch_size):
                    idx = idxs[n]
                    memory.update(idx, abs(error[n][0]))

                sum_q += np.amax(predicted_q)


                # update actor policy
                # a_output = actor.predict(s_rep, sess)
                # grad = critic.action_gradient(s_rep, a_output, sess)
                # actor.train(s_rep, grad[0], sess)

                # update target network
                update_actor = update_target_network(actor, actor_target)
                actor_target.set_weights(update_actor)
                update_critic = update_target_network(critic, critic_target)
                critic_target.set_weights(update_critic)

            sum_reward += r

            s = s1
            j += 1
            if done:
                print('Episode: {}, reward: {}, Q_max: {}'.format(i, int(sum_reward), sum_q / float(j)))
                # rewards.append(sum_reward)
                print('===========')
                break

def loss_fn(y_true, y_pred, importance):
    error = tf.math.square(y_true, y_pred)
    return tf.reduce_mean(tf.math.multiply(error, importance))


def update_target_network(network_params, target_network_params, tau=.01):

    update = []
    for layer, target_layer in zip(network_params.layers, target_network_params.layers):
        target_weights = target_layer.get_weights()
        network_weights = layer.get_weights()
        update.append(network_weights*tau + target_weights*(1-tau))
    return np.array(update)



if __name__ == '__main__':
    test_rl()