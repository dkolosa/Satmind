import numpy as np

import tensorflow as tf

from env_orekit import OrekitEnv
import actor-critic


if __name__ == '__main__':

    # Orekit env
    save = False
    env = OrekitEnv()

    year, month, day, hr, minute, sec = 2018, 8, 1, 9, 30, 00.00
    date = [year, month, day, hr, minute, sec]
    env.set_date(date)

    mass = 1000.0
    fuel_mass = 500.0
    duration = 2 * 24.0 * 60.0 ** 2

    sma = 40_000.0e3
    e = 0.001
    i = 0.0
    omega = 0.1
    rann = 0.01
    lv = 0.01
    state = [sma, e, i, omega, rann, lv]

    # target state
    a_targ = 45_000_000.0
    e_targ = e
    i_targ = i
    omega_targ = omega
    raan_targ = rann
    lM_targ = lv
    state_targ = [a_targ, e_targ, i_targ, omega_targ, raan_targ, lM_targ]

    env.create_orbit(state, env._initial_date, target=False)
    env.set_spacecraft(mass, fuel_mass)
    env.create_Propagator()
    env.setForceModel()

    final_date = env._initial_date.shiftedBy(duration)

    env.create_orbit(state_targ, final_date, target=True)

    env._extrap_Date = env._initial_date
    stepT = 100.0

    # thrust_mag = 0.0

    thrust_values = [0.0, 0.25, 0.50, 0.75, 1.0]
    # learning parameters
    y = .95
    e = 0.10
    num_episodes = 100
    # steps and rewards per episode (respectively)
    j_list = []
    r_list = []

    # Network Model
    num_inputs = 2
    num_outputs = 5

    layer_1_nodes = 128

    actor = actor-critic.Actor(num_inputs,num_outputs, layer_1_nodes)
    critic = actor-critic.Critic(num_inputs, layer_1_nodes)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(num_episodes):

            s = env.reset()

            rall=0
            r=0
            d=False
            j=0
            reward = []

            while j < 5000:

                a = actor.choose_action(s, sess)

                s1, r, done, _ = env.step(a, stepT)

                reward.append(r)

                # Train the critic and actor
                td_error = critic.train(s, r, s1, sess)
                exp_v = actor.train(s, a, td_error, sess)

                s = s1

                j+=1

                if done:
                    reward_sun = sum(reward)

                print('episode: ', i, "reward: ", r)

            




















