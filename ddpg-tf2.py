import tensorflow as tf
import numpy as np
import gym
import gym.spaces
import os, datetime
from Satmind.utils import OrnsteinUhlenbeck
from Satmind.DDPG import DDPG


def test_rl():
    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v3', 'LunarLanderContinuous-v2',
            'BipedalWalkerHardcore-v3')

    ENV = ENVS[0]

    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV), exist_ok=True)
    save_dir = os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV)

    env = gym.make(ENV)
    iter_per_episode = 200
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]
    action_bound = env.action_space.high

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 1001
    PER = False

    batch_size = 64
    #Pendulum
    layer_1_nodes, layer_2_nodes = 256, 128

    tau = 0.01
    actor_lr, critic_lr = 0.0001, 0.001
    GAMMA = 0.99
    ep = 0.001

    actor_noise = OrnsteinUhlenbeck(np.zeros(n_action))

    agent = DDPG(n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, PER, GAMMA,
                 tau, batch_size, save_dir)

    agent.update_target_network(agent.actor, agent.actor_target, agent.tau)
    agent.update_target_network(agent.critic, agent.critic_target, agent.tau)

    load_models = False
    save = False
    # If loading model, a gradient update must be called once before loading weights
    if load_models:
        load_model(PER, agent, batch_size, env, ep, n_action, n_state)

    for i in range(num_episodes):
        s = env.reset()
        sum_reward = 0
        agent.sum_q = 0
        agent.actor_loss = 0
        agent.critic_loss = 0
        j = 0

        while True:
            env.render()

            a = np.clip(agent.actor(tf.convert_to_tensor([s], dtype=tf.float32))[0] + actor_noise(), a_max=action_bound,
                        a_min=-action_bound)
            s1, r, done, _ = env.step(a)
            # Store in replay memory
            if PER:
                error = 1 # D_i = max D
                agent.memory.add(error, (
                np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
            else:
                agent.memory.add(
                    (np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
            agent.train()

            sum_reward += r
            s = s1
            j += 1
            if done:
                print(f'Episode: {i}, reward: {int(sum_reward)}, q_max: {agent.sum_q / float(j)},\nactor loss:{agent.actor_loss / float(j)}, critic loss:{agent.critic_loss/ float(j)}')
                # rewards.append(sum_reward)
                print('===========')
                if save:
                    agent.save_model()
                break


def load_model(PER, agent, batch_size, env, ep, n_action, n_state):
    for i in range(batch_size + 1):
        s = env.reset()
        a = agent.actor(tf.convert_to_tensor([s], dtype=tf.float32))[0]
        s1, r, done, _ = env.step(a)
        # Store in replay memory
        if PER:
            error = abs(r + ep)  # D_i = max D
            agent.memory.add(error, (
                np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
        else:
            agent.memory.add(
                (np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
    agent.train()
    agent.load_model()


if __name__ == '__main__':
    test_rl()