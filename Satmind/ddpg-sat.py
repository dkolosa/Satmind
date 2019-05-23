import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import argparse
import datetime
import json
from math import degrees
import Satmind.actor_critic as models
from Satmind.env_orekit import OrekitEnv
import Satmind.utils
from Satmind.replay_memory import Experience, Per_Memory


stepT = 500.0

def orekit_setup():

    input_file = 'input.json'
    with open(input_file) as input:
        data = json.load(input)
        mission = data['inclination_change']
        state = list(mission['initial_orbit'].values())
        state_targ = list(mission['target_orbit'].values())
        date = list(mission['initial_date'].values())
        dry_mass = mission['spacecraft_parameters']['dry_mass']
        fuel_mass = mission['spacecraft_parameters']['fuel_mass']
        duration = mission['duration']
    mass = [dry_mass, fuel_mass]
    duration = 24.0 * 60.0 ** 2 * 6

    env = OrekitEnv(state, state_targ, date, duration,mass, stepT)
    return env, duration


def main(args):
    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v2', 'OrekitEnv-v0')
    ENV = ENVS[3]

    env, duration = orekit_setup()
    iter_per_episode = int(duration / stepT)
    # Network inputs and outputs
    features = env.observation_space
    n_actions = 3
    action_bound = 0.7

    np.random.seed(1234)

    num_episodes = 5000
    batch_size = 64

    layer_1_nodes, layer_2_nodes = 128, 100
    tau = 0.01
    actor_lr, critic_lr = 0.0001, 0.0001
    GAMMA = 0.99

    # Initialize actor and critic network and targets
    actor = models.Actor(features, n_actions, layer_1_nodes, layer_2_nodes, action_bound, tau, actor_lr, batch_size, 'actor')
    actor_noise = Satmind.utils.OrnsteinUhlenbeck(np.zeros(n_actions))
    critic = models.Critic(features, n_actions, layer_1_nodes, layer_2_nodes, critic_lr, tau, 'critic', actor.trainable_variables)

    # Replay memory buffer
    # replay = Experience(buffer_size=1000000)
    per_mem = Per_Memory(capacity=100000)
    thrust_values = np.array([0.00, 0.0, -0.4])
    per_mem.pre_populate(env, features, n_actions, thrust_values)

    # replay = Experience(buffer_size=1000000)
    # thrust_values = np.array([0.00, 0.0, -0.7])
    # replay.populate_memory(env, features, n_actions, thrust_values)

    saver = tf.train.Saver()

    # Save model directory
    LOAD = False
    if args['model'] is not None:
        checkpoint_path = args['model'] + '/'
        os.makedirs(checkpoint_path,exist_ok=True)
        if args['test']:
            TRAIN = False
        else:
            TRAIN = True
            LOAD = True
    else:
        TRAIN = True
        today = datetime.date.today()
        path = '/tmp/ddpg_models/'
        checkpoint_path =path+str(today)+'-'+ENV+'/'
        os.makedirs(checkpoint_path, exist_ok=True)
        print(f'Model will be saved in: {checkpoint_path}')

    # Save the model parameters (for reproducibility)
    params = checkpoint_path + 'model_params.txt'
    with open(params, 'w+') as text_file:
        text_file.write("enviornment params:\n")
        text_file.write("enviornment: " + ENV + "\n")
        text_file.write("episodes: {}, iterations per episode {}\n".format(num_episodes, iter_per_episode))
        text_file.write("model parameters:\n")
        text_file.write(actor.__str__())
        text_file.write(critic.__str__() + "\n")

    # Render target
    env.render_target()
    env.randomize = True

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if TRAIN:
            if LOAD:
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            actor.update_target_network(sess)
            critic.update_target_network(sess)

            rewards = []

            for i in range(num_episodes):
                s = env.reset()
                sum_reward = 0
                sum_q = 0
                actions = []
                for j in range(iter_per_episode):

                    # Select an action
                    a = actor.predict(np.reshape(s, (1, features)), sess) + actor_noise()*.1

                    # Observe state and reward
                    s1, r, done = env.step(a[0])

                    actions.append(a[0])
                    # Store in replay memory
                    # replay.add((np.reshape(s, (features,)), np.reshape(a, (n_actions,)), r, np.reshape(s1,(features,)), done))
                    error = abs(r)  # D_i = max D
                    per_mem.add(error, (np.reshape(s, (features,)), np.reshape(a[0], (n_actions,)), r, np.reshape(s1, (features,)), done))
                    # sample from random memory
                    # if batch_size < replay.get_count:
                    #     mem = replay.experience_replay(batch_size)
                    #     s_rep = np.array([_[0] for _ in mem])
                    #     a_rep = np.array([_[1] for _ in mem])
                    #     r_rep = np.array([_[2] for _ in mem])
                    #     s1_rep = np.array([_[3] for _ in mem])
                    #     d_rep = np.array([_[4] for _ in mem])

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
                        error, predicted_q, _ = critic.train(s_rep, a_rep, np.reshape(y_i, (batch_size, 1)),
                                                             np.reshape(isweight, (batch_size, 1)), sess)

                        for n in range(batch_size):
                            idx = idxs[n]
                            per_mem.update(idx, abs(error[n][0]))

                        # update the critic network
                        # predicted_q, _ = critic.train(s_rep, a_rep, np.reshape(y_i, (batch_size,1)), sess)
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
                    if done or j >= iter_per_episode - 1:
                        rewards.append(sum_reward)
                        print(f'I: {degrees(env._currentOrbit.getI())}')
                        print('Episode: {}, reward: {}, Q_max: {}'.format(i, int(sum_reward), sum_q/float(j)))
                        print(f'diff:   a: {(env.r_target_state[0] - env._currentOrbit.getA())/1e3},\n'
                              f'ex: {env.r_target_state[1] - env._currentOrbit.getEquinoctialEx()},\t'
                              f'ey: {env.r_target_state[2] - env._currentOrbit.getEquinoctialEy()},\n'
                              f'hx: {env.r_target_state[3] - env._currentOrbit.getHx()},\t'
                              f'hy: {env.r_target_state[4] - env._currentOrbit.getHy()}\n'
                              f'Fuel Mass: {env.cuf_fuel_mass}\n'
                              f'Final Orbit:{env._currentOrbit}\n'
                              f'Initial Orbit:{env._orbit}')
                        print('=========================')
                        saver.save(sess, checkpoint_path)
                        if env.target_hit:
                            n = range(j + 1)
                            save_fig = True
                            show = True
                            env.render_target()
                            if 0 <= i < 10:
                                episode = '00' + str(i)
                            elif 10 <= i < 100:
                                episode = '0' + str(i)
                            elif i >= 100:
                                episode = str(i)
                            env.render_plots(i, save=save_fig, show=show)
                            thrust_mag = np.linalg.norm(np.asarray(actions), axis=1)
                            plt.subplot(2, 1, 1)
                            plt.plot(thrust_mag)
                            plt.title('Thrust Magnitude (N)')
                            plt.subplot(2, 1, 2)
                            plt.plot(n, np.asarray(actions)[:, 0], n, np.asarray(actions)[:, 1], n,
                                     np.asarray(actions)[:, 2])
                            plt.xlabel('Mission Step ' + str(stepT) + ' sec per step')
                            plt.title('Thrust (N)')
                            plt.legend(('R', 'S', 'W'))
                            plt.tight_layout()
                            if save_fig: plt.savefig('results/' + episode + '/thrust.pdf')
                            if show:
                                plt.show()
                        break
                if i % 10 == 0:
                    n = range(j+1)
                    if i % 10 == 0:
                        save_fig = True
                    else:
                        save_fig = False
                    if i % 50 == 0:
                        show = True
                    else:
                        show = False
                    env.render_target()
                    env.render_plots(i, save=save_fig, show=show)
                    thrust_mag = np.linalg.norm(np.asarray(actions), axis=1)

                    if 0 <= i < 10:
                        episode = '00' + str(i)
                    elif 10 <= i < 100:
                        episode = '0' + str(i)
                    elif i >= 100:
                        episode = str(i)

                    plt.subplot(2,1,1)
                    plt.plot(thrust_mag)
                    plt.title('Thrust Magnitude (N)')
                    plt.subplot(2,1,2)
                    plt.plot(n,np.asarray(actions)[:,0], n, np.asarray(actions)[:,1], n, np.asarray(actions)[:,2])
                    plt.xlabel('Mission Step ' + str(stepT) + ' sec per step')
                    plt.title('Thrust (N)')
                    plt.legend(('R', 'S', 'W'))
                    plt.tight_layout()
                    if save_fig: plt.savefig('results/' + episode + '/thrust.pdf')
                    if show:
                        plt.show()

        else:
            if args['model'] is not None:
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            env.render_target()
            for i in range(num_episodes):
                s = env.reset()
                sum_reward = 0
                actions = []
                # while True:
                for j in range(iter_per_episode):
                    # env.render()
                    a = actor.predict(np.reshape(s, (1, features)), sess)
                    s1, r, done = env.step(a[0])
                    s = s1
                    sum_reward += r
                    # if done:
                    actions.append(a[0])
                    if done or j >= iter_per_episode - 1:
                        print(f'Episode: {i}, reward: {int(sum_reward)}')
                        plt.plot(actions)
                        plt.show()
                        env.render_plots()
                        break
        plt.plot(rewards)
        plt.tight_layout()
        plt.savefig('results/rewards.pdf')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="path of a trained tensorlfow model (str: path)", type=str)
    parser.add_argument('--test', help="pass if testing a model", action='store_true')
    args = vars(parser.parse_args())
    main(args)
