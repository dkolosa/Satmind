import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import gym
import gym.spaces

from env_orekit import OrekitEnv


class Experience:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def experience_replay(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)


if __name__ == '__main__':
    # Create the enviornment
    # env = gym.make('FrozenLake-v0')
    # env.reset()

    # Orekit env

    env = OrekitEnv()

    year, month, day, hr, minute, sec = 2018, 8, 1, 9, 30, 00.00
    date = [year, month, day, hr, minute, sec]
    env.set_date(date)

    mass = 1000.0
    fuel_mass = 500.0
    duration =3 * 24.0 * 60.0 ** 2

    sma = 41_000.0e3
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

    thrust_mag = [0.0, 0.1, 0.2, 0.5, 1.0, 3.0]

    # learning parameters
    y = .99
    e = 0.01
    num_episodes = 21
    # steps and rewards per episode (respectively)
    j_list = []
    r_list = []

    # experience replay
    experience = Experience(buffer_size=50)

    # Network Model
    num_inputs = 2
    num_outputs = 6
    layer_1_nodes = 20
    layer_2_nodes = 20

    # Establish feed-forward network
    inputs = tf.placeholder(shape=[1, num_inputs], dtype=tf.float32)

    # w1 = tf.Variable(tf.zeros[16,100])
    # b1 = tf.variable(tf.zeros[100])

    with tf.variable_scope('layer-1'):
        weights = tf.get_variable(name='weights-1', shape=(num_inputs, layer_1_nodes),
                                  initializer=tf.contrib.layers.xavier_initializer())
        # bias = tf.get_variable(name='bias1', shape=([layer_1_nodes]), initializer=tf.zeros_initializer())
        layer_1_output = tf.nn.tanh(tf.matmul(inputs, weights))

    with tf.variable_scope('layer-2'):
        weights = tf.get_variable(name='weights-2', shape=(layer_1_nodes, layer_2_nodes))
        layer_2_output = tf.nn.tanh(tf.matmul(layer_1_output, weights))

    # weights2 = tf.Variable(tf.random_uniform([16,4]
    with tf.variable_scope('output'):
        weights = tf.get_variable(name='weight-out', shape=(layer_2_nodes, num_outputs))
        # bias = tf.get_variable(name='bias_out', shape=([num_outputs]), initializer=tf.zeros_initializer())

        # Q_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + bias)
        Q_output = tf.matmul(layer_2_output, weights)

    predict = tf.argmax(Q_output, 1)

    # actions = tf.placeholder(shape=[None], dtype=tf.int32)
    # actions_onehot = tf.one_hot(actions, num_outputs, dtype=tf.float32)

    # Q = tf.reduce_sum(tf.multiply(Q_output, actions_onehot), reduction_indices=1)

    # Sum of squares loss between target and predicted Q

    next_Q = tf.placeholder(shape=[1, num_outputs], dtype=tf.float32)
    # next_Q = tf.placeholder(shape=[None], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(next_Q - Q_output))
    trainer = tf.train.AdamOptimizer(learning_rate=0.001)
    update = trainer.minimize(loss)

    # with tf.variable_scope('logging'):
    #     tf.summary.scalar('current_cost', update)
    #     summary = tf.summary.merge_all()

    # saver = tf.train.saver()

    # Initialize network nodes
    init = tf.global_variables_initializer()

    # Network Training
    # Start tensorflow session
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            # reset enviornment to get first observation
            s = env.reset()
            rall = 0
            d = False
            j = 0
            # episonde_eperience = Experience(buffer_size=50)

            # Q-network
            # while j < 99:
            while env._extrap_Date.compareTo(final_date) <= 0:
                actions = []
                j += 1
                # choose an action
                # This value is thrust
                a, allQ = sess.run([predict, Q_output], feed_dict={inputs: [s]})
                # print(Q_output)

                # print('action: ', a)
                # print('allQ: ', allQ)
                # a, allQ = sess.run([predict, Q_output],
                #                    feed_dict={inputs:np.identity(num_inputs)[s:s+1]})
                if np.random.rand(1) < e:
                    # a[0] = env.action_space.sample()
                    a[0] = random.randint(0, len(thrust_mag)-1)
                    # print("Random Hit!")
                # print("a[0]: ", a[0])
                # Get a new state and reward
                # The state is the x-y coordinates, r =0 if not reached
                # s1, r, d, _ = env.step(thrust_mag[a[0]], stepT)

                if a == 0:
                    action = thrust_mag[0]
                elif a == 1:
                    action = thrust_mag[1]
                elif a == 2:
                    action = thrust_mag[2]
                elif a == 3:
                    action = thrust_mag[3]
                elif a == 4:
                    action = thrust_mag[4]
                elif a == 5:
                    action = thrust_mag[5]
                else:
                    action = thrust_mag[0]

                s1, r, done, _ = env.step(action, stepT)
                actions.append(action)
                # env.shift_date(stepT)

                # s1 = env.step(thrust_mag, stepT)

                # episonde_eperience.experience_replay(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

                # Obtain the Q value
                Q1 = sess.run(Q_output, feed_dict={inputs: [s1]})
                # Get maxQ and set target value for chosen action
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + y * maxQ1

                # Train the NN using target and predicted Q values
                # _, W1 = sess.run([update, weights],
                # feed_dict={inputs:np.identity(num_inputs)[s:s+1],next_Q:targetQ})

                _, W1 = sess.run([update, weights], feed_dict={inputs: [s], next_Q: targetQ})
                rall += r
                s = s1
                actions.append(action)



                if done:
                    # Random action
                    e = 1.0 / ((i / 50) + 10)
                    plt.title('completed episode')
                    plt.subplot(2, 1, 1)
                    plt.plot(np.asarray(env._px) / 1e3, np.asarray(env._py) / 1e3)
                    plt.xlabel('km')
                    plt.ylabel('km')
                    plt.subplot(2, 1, 2)
                    plt.plot(r_list)
                    plt.show()
                    break

            j_list.append(j)
            r_list.append(rall)


            # experience.experience_replay(episonde_eperience)

            if i % 5 == 0:
            # plt.plot(env._px, env._py)
            # env.render_plots()
            # env.render()
                plt.subplot(2, 2, 1)
                plt.plot(np.asarray(env._px)/1e3, np.asarray(env._py)/1e3)
                plt.xlabel('km')
                plt.ylabel('km')
                plt.subplot(2, 2, 2)
                plt.plot(r_list)
                plt.subplot(2,2,3)
                plt.plot(range(0,len(actions)), actions)
                plt.show()
            print("episode {} of {}".format(i, num_episodes))

        # print('KEY:\nSFFF(S: starting point, safe)\n',
        #       'FHFH(F: frozen surface, safe)\n',
        #       'FFFH(H: hole, fall to your doom)\n',
        #       'HFFG(G: goal, where the frisbee is located)')
        # print("successful episodes " + str(sum(r_list)/num_episodes*100) + "%")
        # print("Score: " + str(sum(r_list)))
        print('orbit:{}'.format(env._currentOrbit.getA()))
        # plt.show()
        # print(r_list)
