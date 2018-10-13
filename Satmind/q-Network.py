import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from collections import deque
import gym
import gym.spaces

from env_orekit import OrekitEnv


class Experience:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)


    def experience_replay(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)


class Q_Network:

    def __init__(self, num_inputs, num_outputs, layer_1_nodes, layer_2_nodes):
        # Establish feed-forward network
        self.inputs = tf.placeholder(shape=[1, num_inputs], dtype=tf.float32)
        self.next_Q = tf.placeholder(shape=[1, num_outputs], dtype=tf.float32)

        # self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
        # one_hot_action = tf.one_hot(self.actions, num_outputs)

        # w1 = tf.Variable(tf.zeros[16,100])
        # b1 = tf.variable(tf.zeros[100])

        # with tf.variable_scope('layer-1'):
        #     weights = tf.get_variable(name='weights-1', shape=(num_inputs, layer_1_nodes),
        #                               initializer=tf.contrib.layers.xavier_initializer())
        #     # bias = tf.get_variable(name='bias1', shape=([layer_1_nodes]), initializer=tf.zeros_initializer())
        #     layer_1_output = tf.nn.tanh(tf.matmul(inputs, weights))

        self.fc1 = tf.contrib.layers.fully_connected(self.inputs, layer_1_nodes)

        self.fc2 = tf.contrib.layers.fully_connected(self.fc1, layer_2_nodes)

        # self.fc3 = tf.contrib.layers.fully_connected(self.fc2, layer_2_nodes)

        self.Q_output = tf.contrib.layers.fully_connected(self.fc2, num_outputs,activation_fn=None)

        self.predict = tf.argmax(self.Q_output, 1)
        

        # self.Q = tf.reduce_sum(tf.multiply(self.Q_output, one_hot_action), axis=1)

        # Sum of squares loss between target and predicted Q

        self.loss = tf.reduce_mean(tf.square(self.next_Q - self.Q_output), axis=1)

        # self.loss = tf.reduce_sum(tf.square(self.next_Q - self.Q_output))
        trainer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.update = trainer.minimize(self.loss)


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
    duration =2 * 24.0 * 60.0 ** 2

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

    thrust_values = [0.0, 0.25, 0.50, 0.75, 1.0, 2.0]
    # learning parameters
    y = .95
    e = 0.05
    num_episodes = 500
    # steps and rewards per episode (respectively)
    j_list = []
    r_list = []

    # experience replay
    experience = Experience(buffer_size=100)

    # Network Model
    num_inputs = 2
    num_outputs = 6
    # TODO: one-hot encode output acitons
        # [[1,0,0,0,0,0],[0,1,0,0,0,0],...]

    layer_1_nodes = 512
    layer_2_nodes = 512

    deep_q = Q_Network(num_inputs, num_outputs, layer_1_nodes, layer_2_nodes)


    # writer = tf.summary.scalar("Loss", loss)
    # with tf.variable_scope('logging'):
    #     writer = tf.summary.FileWriter("log/dq")
    #     tf.summary.scalar('loss', loss)
    #     summary = tf.summary.merge_all()

    # saver = tf.train.saver()

    # Initialize network nodes
    init = tf.global_variables_initializer()

    # Network Training
    # Start tensorflow session
    hit = 0
    with tf.Session() as sess:
        sess.run(init)
        track_a =[]
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
                a, allQ = sess.run([deep_q.predict, deep_q.Q_output], feed_dict={deep_q.inputs: [s]})

                if np.random.rand(1) < e:
                    a[0] = random.randint(0, len(thrust_values)-1)
                    # print("Random Hit!")

                # Get a new state and reward
                # The state is the x-y coordinates, r =0 if not reached
                # if a == 0:
                #     action = thrust_values[0]
                # elif a == 1:
                #     action = thrust_values[1]
                # elif a == 2:
                #     action = thrust_values[2]
                # elif a == 3:
                #     action = thrust_values[3]
                # elif a == 4:
                #     action = thrust_values[4]
                # elif a == 5:
                #     action = thrust_values[5]
                # else:
                #     action = thrust_values[0]

                action = thrust_values[int(a)]

                s1, r, done, _ = env.step(action, stepT)
                actions.append(action)

                experience.add((s, action, r, s1))

                # Obtain the Q value
                Q1 = sess.run(deep_q.Q_output, feed_dict={deep_q.inputs: [s1]})

                # Get maxQ and set target value for chosen action
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + y * maxQ1

                # Train the NN using target and predicted Q values
                _, W1 = sess.run([deep_q.update, deep_q.fc1], feed_dict={deep_q.inputs: [s], deep_q.next_Q: targetQ})
                _, W2 = sess.run([deep_q.update, deep_q.fc2], feed_dict={deep_q.inputs: [s], deep_q.next_Q: targetQ})

                loss, _ = sess.run([deep_q.loss, deep_q.update], feed_dict={deep_q.inputs: [s], deep_q.next_Q: targetQ})

                rall += r
                s = s1

                # print("==============")
                # print("loss: ", opt)
                loss_tot.append(loss)
                if done:
                    hit +=1
                    # Random action
                    e = 1.0 / ((i / 50) + 10)
                    plt.title('completed episode')
                    plt.subplot(2, 1, 1)
                    plt.plot(np.asarray(env._px) / 1e3, np.asarray(env._py) / 1e3)
                    plt.xlabel('km')
                    plt.ylabel('km')
                    plt.subplot(2, 1, 2)
                    plt.plot(actions)
                    plt.xlabel('Mission Step ' + str(stepT) + 'sec per step')
                    plt.ylabel('Thrust (N)')
                    plt.show()
                    break

            j_list.append(j)
            r_list.append(rall)

            # print('a final {}'.format(env._currentOrbit.getA()/1e3))
            track_a.append(env._currentOrbit.getA()/1e3)

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
            print("episode {} of {}, orbit:{}".format(i, num_episodes, env._currentOrbit.getA()/1e3))
        plt.subplot(2,1,1)
        plt.plot(track_a)
        plt.title('final sma per episode')
        plt.subplot(2,1,2)
        plt.plot(loss_tot)
        plt.title('Total Loss')
        plt.show()
        print("Target hit: {} of {} episodes".format(hit, num_episodes))
