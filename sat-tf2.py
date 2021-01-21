import tensorflow as tf
import numpy as np
from Satmind.utils import OrnsteinUhlenbeck
from Satmind.utils import orekit_setup
from Satmind.DDPG import DDPG
import os
import datetime
from Satmind.env_orekit import OrekitEnv
stepT = 500.0


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


def main():
    ENVS = ('Orbit_Raising', 'inclination_change', 'sma_change', 'meo_geo')
    ENV = ENVS[2]
    
    env, duration = orekit_setup(ENV, stepT)
    
    iter_per_episode = int(duration / stepT)
    
    # Network inputs and outputs
    n_state = env.observation_space
    n_action = env.action_space
    action_bound = env.action_bound

    np.random.seed(1234)

    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV), exist_ok=True)
    save_dir = os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV)

    num_episodes = 1001
    PER = False

    batch_size = 64
    # Pendulum
    layer_1_nodes, layer_2_nodes = 256, 128

    tau = 0.01
    actor_lr, critic_lr = 0.0001, 0.001
    GAMMA = 0.99
    ep = 0.001

    # Render target
    env.render_target()
    env.randomize = False

    actor_noise = OrnsteinUhlenbeck(np.zeros(n_action))

    agent = DDPG(n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, PER, GAMMA,
                 tau, batch_size, save_dir)

    agent.update_target_network(agent.actor, agent.actor_target, agent.tau)
    agent.update_target_network(agent.critic, agent.critic_target, agent.tau)

    #TODO:
    # Write the parameters file
    load_models = False
    save = False
    # If loading model, a gradient update must be called once before loading weights
    if load_models:
        load_model(PER, agent, batch_size, env, ep, n_actions, n_state)

    for i in range(num_episodes):
        s = env.reset()
        sum_reward = 0
        agent.sum_q = 0
        agent.actor_loss = 0
        agent.critic_loss = 0
        j = 0

        for j in range(iter_per_episode):

            a = np.clip(agent.actor(tf.convert_to_tensor([s], dtype=tf.float32))[0] + actor_noise(), a_max=action_bound,
                        a_min=-action_bound)
            s1, r, done = env.step(a)
            # Store in replay memory
            if PER:
                error = 1  # D_i = max D
                agent.memory.add(error, (
                    np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
            else:
                agent.memory.add(
                    (np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
            agent.train()

            sum_reward += r
            s = s1
            j += 1
            if done or (j >= iter_per_episode - 1):
                print('Episode: {}, reward: {}, Q_max: {}'.format(i, int(sum_reward), agent.sum_q / float(j)))
                print(f'actor loss: {agent.actor_loss / float(j)}, critic loss{agent.critic_loss / float(j)}')
                print(f'diff:   a (km): {((env._targetOrbit.getA() - env.currentOrbit.getA()) / 1e3):.4f},\n'
                      f'ex: {(env.r_target_state[1] - env._currentOrbit.getEquinoctialEx()):.3f},\t'
                      f'ey: {(env.r_target_state[2] - env._currentOrbit.getEquinoctialEy()):.3f},\n'
                      f'hx: {(env.r_target_state[3] - env._currentOrbit.getHx()):.3f},\t'
                      f'hy: {(env.r_target_state[4] - env._currentOrbit.getHy()):.4f}\n'
                      f'Fuel Mass: {(env.cuf_fuel_mass):.3f}\n'
                      f'Initial Orbit:{env._orbit}\n'
                      f'Final Orbit:{env._currentOrbit}\n'
                      f'Target Orbit:{env._targetOrbit}')
                if save:
                    agent.save_model()
                break

if __name__ == '__main__':
    main()
