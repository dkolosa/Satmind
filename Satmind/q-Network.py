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
    stepT = 1000.0

    # thrust_mag = 0.0

    thrust_values = [0.0, 0.25, 0.50, 1.0, 5.0, 10.0]
    thrust_mag = [0, 1, 2, 3, 4, 5]
    # learning parameters
    y = .95
    e = 0.01
    num_episodes = 5
    # steps and rewards per episode (respectively)
    j_list = []
    r_list = []

    # experience replay
    experience = Experience(buffer_size=50)

    # Network Model
    num_inputs = 2
    num_outputs = 6
    # TODO: one-hot encode output acitons
        # [[1,0,0,0,0,0],[0,1,0,0,0,0],...]

    layer_1_nodes = 512
    layer_2_nodes = 512

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

    with tf.variable_scope('output'):
        weights = tf.get_variable(name='weight-out', shape=(layer_2_nodes, num_outputs))
        # bias = tf.get_variable(name='bias_out', shape=([num_outputs]), initializer=tf.zeros_initializer())
        Q_output = tf.matmul(layer_2_output, weights)

    next_Q = tf.placeholder(shape=[1, num_outputs], dtype=tf.float32)

    predict = tf.argmax(Q_output, 1)

    # Sum of squares loss between target and predicted Q
    loss = tf.reduce_sum(tf.square(next_Q - Q_output))
    trainer = tf.train.AdamOptimizer(learning_rate=0.001)
    update = trainer.minimize(loss)


    # writer = tf.summary.scalar("Loss", loss)
    with tf.variable_scope('logging'):
        writer = tf.summary.FileWriter("log/dq")
        tf.summary.scalar('loss', loss)
        summary = tf.summary.merge_all()

    # saver = tf.train.saver()

    # Initialize network nodes
    init = tf.global_variables_initializer()

    # Network Training
    # Start tensorflow session
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1,num_episodes):
            # reset enviornment to get first observation
            s = env.reset()
            rall = 0
            r = 0
            d = False
            j = 0
            # episonde_eperience = Experience(buffer_size=50)
            # Q-network
            # while j < 99:
            actions = []
            loss_tot = []
            while env._extrap_Date.compareTo(final_date) <= 0:

                j += 1
                # choose an action
                # This value is thrust
                a, allQ = sess.run([predict, Q_output], feed_dict={inputs: [s]})

                if np.random.rand(1) < e:
                    a[0] = random.randint(0, len(thrust_mag)-1)
                    # print("Random Hit!")

                # Get a new state and reward
                # The state is the x-y coordinates, r =0 if not reached
                if a == 0:
                    action = thrust_values[0]
                elif a == 1:
                    action = thrust_values[1]
                elif a == 2:
                    action = thrust_values[2]
                elif a == 3:
                    action = thrust_values[3]
                elif a == 4:
                    action = thrust_values[4]
                elif a == 5:
                    action = thrust_values[5]
                else:
                    action = thrust_mag[0]

                s1, r, done, _ = env.step(action, stepT)
                actions.append(action)

                # Obtain the Q value
                Q1 = sess.run(Q_output, feed_dict={inputs: [s1]})

                # Get maxQ and set target value for chosen action
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + y * maxQ1

                # Train the NN using target and predicted Q values
                _, W1 = sess.run([update, weights], feed_dict={inputs: [s], next_Q: targetQ})
                rall += r
                s = s1

                opt = sess.run(loss, feed_dict={inputs: [s], next_Q: targetQ})
                # print("==============")
                # print("loss: ", opt)
                loss_tot.append(opt)
                if done:
                    # Random action
                    e = 1.0 / ((i / 50) + 10)
                    plt.title('completed episode')
                    plt.subplot(2, 1, 1)
                    plt.plot(np.asarray(env._px) / 1e3, np.asarray(env._py) / 1e3)
                    plt.xlabel('km')
                    plt.ylabel('km')
                    plt.subplot(2, 1, 2)
                    plt.plot(action)
                    plt.show()
                    break

            j_list.append(j)
            r_list.append(rall)

            print('a final {}'.format(env._currentOrbit.getA()/1e3))

            # experience.experience_replay(episonde_eperience)
            # plt.plot(j_list, opt)
            # plt.plot(actions)

            plt.show()
            if i % 5 == 0:

                plt.title('iteration {}'.format(i))
                plt.subplot(2, 1, 1)
                plt.plot(np.asarray(env._px)/1e3, np.asarray(env._py)/1e3)
                plt.xlabel('km')
                plt.ylabel('km')
                plt.subplot(2, 1, 2)
                plt.plot(r_list)
                # plt.subplot(2,2,3)
                # plt.plot(actions)
                plt.show()
            print("episode {} of {}".format(i, num_episodes))
        plt.plot(loss_tot)
        plt.show()
        print('orbit:{}'.format(env._currentOrbit.getA()/1e3))

