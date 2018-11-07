import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from collections import deque
import gym
import gym.spaces
import time

from env_orekit import OrekitEnv


class Experience:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        """
        Add an experience to the buffer
        :param experience: (state, action, reward, next state)
        :return:
        """
        self.buffer.append(experience)

    def experience_replay(self):
        """
        Get a random experience from the deque
        :return:  experience: (state, action, reward, next state)
        """
        index = np.random.choice(np.arange(len(self.buffer)), replace=False)
        return self.buffer[index]

    def populate_memory(self, env, thrust_values, stepT):
        """
        Populate with experiences by taking random actions
        :param env: Agent enviornment object
        :param thrust_values: Given list of possible thrust levels
        :param stepT: Thrust step values
        :return:
        """
        state = env.reset()
        for e in self.buffer:
            act = np.random.choice(thrust_values)
            state_1, reward, done_mem, _ = env.state(act, stepT)
            e = [state, act, reward, state_1]
            self.add(e)
            state = state_1

    def print_buffer(self):
        '''
        Prints all of the experience data stored in the buffer

        :return: Printed list of the experience in the buffer
        '''
        for e in self.buffer: print(e)


class Q_Network:
    """ A Q-learning based neural network"""

    def __init__(self, num_inputs, num_outputs, layer_1_nodes, layer_2_nodes, name):
        """

        :param num_inputs: number of inputs nodes (integer)
        :param num_outputs: number of output nodes (integer)
        :param layer_1_nodes: Number of nodes in the first hidden layer
        :param layer_2_nodes: number of nodes in the second hidden layer
        :param name: name of network
        """

        # Establish feed-forward network

        with tf.variable_scope(str(name) + '-inputs'):
            self.inputs = tf.placeholder(shape=[1, num_inputs], dtype=tf.float32)

        self.next_Q = tf.placeholder(shape=[1, num_outputs], dtype=tf.float32)


        # w1 = tf.Variable(tf.zeros[16,100])
        # b1 = tf.variable(tf.zeros[100])

        # with tf.variable_scope('layer-1'):
        #     weights = tf.get_variable(name='weights-1', shape=(num_inputs, layer_1_nodes),
        #                               initializer=tf.contrib.layers.xavier_initializer())
        #     # bias = tf.get_variable(name='bias1', shape=([layer_1_nodes]), initializer=tf.zeros_initializer())
        #     layer_1_output = tf.nn.tanh(tf.matmul(inputs, weights))

        with tf.variable_scope(str(name) + '-layer-1'):
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs, layer_1_nodes)

        with tf.variable_scope(str(name) + '-layer-2'):
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, layer_2_nodes)

        # with tf.variable_scope('layer-3'):
        #     self.fc3 = tf.contrib.layers.fully_connected(self.fc2, layer_2_nodes)
        #
        # with tf.variable_scope('layer-4'):
        #     self.fc4 = tf.contrib.layers.fully_connected(self.fc3, layer_2_nodes)
        #
        # with tf.variable_scope('layer-5'):
        #     self.fc5 = tf.contrib.layers.fully_connected(self.fc4, layer_2_nodes)

        with tf.variable_scope(str(name) + '-output'):
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
    e = 0.05
    num_episodes = 500
    # steps and rewards per episode (respectively)
    j_list = []
    r_list = []

    # experience replay
    experience = Experience(buffer_size=500)
    experience.populate_memory(env, thrust_values, stepT)

    experience.print_buffer()
    # Network Model
    num_inputs = 2
    num_outputs = 5
    # TODO: one-hot encode output acitons
        # [[1,0,0,0,0,0],[0,1,0,0,0,0],...]

    layer_1_nodes = 512
    layer_2_nodes = 512

    deep_q = Q_Network(num_inputs, num_outputs, layer_1_nodes, layer_2_nodes, name='action')

    target_q = Q_Network(num_inputs, num_outputs, 128, 128, name='target')

    # Initialize network nodes
    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    # Network Training
    # Start tensorflow session
    hit = 0

    # Initialize the saver
    saver = tf.train.Saver()
    # experience.populate_experience(thrust=thrust_values, env=env, stepT=stepT, final_date=final_date)


    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter('log/graph',sess.graph)
        loss_writer = tf.summary.FileWriter('log/loss')

        sess.run(init)
        track_a =[]
        for i in range(1,num_episodes):
            # start = time.time()
            # reset enviornment to get first observation
            s = env.reset()
            rall = 0
            r = 0
            d = False
            j = 0
            # Q-network
            # while j < 99:
            actions = []
            loss_tot = []
            reward = []
            # while env._extrap_Date.compareTo(final_date) <= 0:
            while j < 5000:
                j += 1
                # choose an action
                # This value is thrust
                a, allQ = sess.run([deep_q.predict, deep_q.Q_output], feed_dict={deep_q.inputs: [s]})

                if np.random.rand(1) < e:
                    a[0] = random.randint(0, len(thrust_values)-1)
                # Get a new state and reward
                # The state is the x-y coordinates, r =0 if not reached

                action = thrust_values[int(a)]
                s1, r, done, _ = env.step(action, stepT)
                actions.append(action)

                experience.add((s, action, r, s1))

                # Grab the state from replay memory for training
                memory = experience.experience_replay()
                state_mem = np.asarray(memory[0:1]).flatten()
                action_mem = memory[1]
                reward_mem = memory[2]
                next_state_mem = np.asarray(memory[3:4]).flatten()

                # Obtain the Q value
                Q1 = sess.run(deep_q.Q_output, feed_dict={deep_q.inputs: [s1]})
                target_Q1 = sess.run(target_q.Q_output, feed_dict={target_q.inputs: [next_state_mem]})

                # Get maxQ and set target value for chosen action
                maxQ1 = np.argmax(Q1)
                targetQ = allQ

                # targetQ[0, a[0]] = r + y * maxQ1
                targetQ[0, a[0]] = reward_mem + y * target_Q1[0,maxQ1]


                # Train the NN using target and predicted Q values
                # _, W1 = sess.run([deep_q.update, deep_q.fc1], feed_dict={target_q.inputs: [s], target_q.next_Q: targetQ})
                # _, W2 = sess.run([deep_q.update, deep_q.fc2], feed_dict={deep_q.inputs: [s], deep_q.next_Q: targetQ})

                loss, _ = sess.run([target_q.loss, target_q.update], feed_dict={target_q.inputs: [state_mem], target_q.next_Q: targetQ})

                rall += r
                s = s1

                # print("==============")
                # print("loss: ", opt)
                loss_tot.append(loss)
                if done:
                    hit +=1
                    # Random action
                    # e = 1.0 / ((i / 50) + 10)
                    e = 0.01
                    r = 100
                    reward.append(r)
                    r_list.append(rall)
                    print("Episode {}, Fuel Mass: {}, date: {}".format(i, env.getTotalMass() - mass, env._currentOrbit.getDate()))
                    plt.title('completed episode')
                    plt.subplot(2, 1, 1)
                    plt.plot(np.asarray(env._px) / 1e3, np.asarray(env._py) / 1e3)
                    plt.xlabel('km')
                    plt.ylabel('km')
                    plt.subplot(2, 1, 2)
                    plt.plot(actions)
                    plt.xlabel('Mission Step ' + str(stepT) + 'sec per step')
                    plt.ylabel('Thrust (N)')
                    plt.tight_layout()
                    plt.show()
                    break
                reward.append(r)

            # stop = time.time()
            j_list.append(j)
            r_list.append(rall)

            # plt.subplot(2,1,1)
            # plt.plot(reward)
            # plt.subplot(2,1,2)
            # plt.plot(actions)
            # plt.show()

            # print('a final {}'.format(env._currentOrbit.getA()/1e3))
            track_a.append(env._currentOrbit.getA()/1e3)

            if i % 10 == 0:
                plt.title('iteration {}'.format(i))
                plt.subplot(2, 1, 1)
                plt.plot(np.asarray(env._px)/1e3, np.asarray(env._py)/1e3)
                plt.xlabel('km')
                plt.ylabel('km')
                plt.subplot(2, 1, 2)
                plt.plot(r_list)
                # plt.subplot(2,2,3)
                # plt.plot(actions)
                plt.tight_layout()
                plt.show()
            # print("episode {}, time {}".format(i, stop-start))
            print("episode {} of {}, orbit:{}".format(i, num_episodes, env._currentOrbit.getA()/1e3))
            if hit == 100:
                break
        if save:
            # Save the entire seesion just in case
            save_session = saver.save(sess, "log/complete_model/complete_model.ckpt")

            # Save the model for future use
            model_dir = "log/model.ckpt"
            save_path = tf.saved_model.simple_save(sess, model_dir,
                                   inputs={"input": deep_q.inputs},
                                   outputs={"output": deep_q.predict})
            print("model save in {}".format(str(save_path)))

        print("Target hit: {} of {} episodes".format(hit, num_episodes))
