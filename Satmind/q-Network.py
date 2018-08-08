import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import random
import gym
import gym.spaces

# from Satmind.env_orekit import OrekitEnv

# Create the enviornment
env = gym.make('FrozenLake-v0')
env.reset()

# Orekit env

# env = OrekitEnv()
# env.reset()
#
# year, month, day, hr, minute, sec = 2018, 8, 1, 9, 30, 00.00
# date = [year, month, day, hr, minute, sec]
# env.set_date(date)
#
# mass = 1000.0
# fuel_mass = 500.0
# duration = 24.0 * 60.0 ** 2
#
# a = 41_000.0e3
# e = 0.01
# i = 1.0
# omega = 0.1
# rann = 0.01
# lv = 0.01
#
# state = [a, e, i, omega, rann, lv]
#
# env.create_orbit(state)
# env.set_spacecraft(mass, fuel_mass)
# env.create_Propagator()
# env.setForceModel()
#
# final_date = env._initial_date.shiftedBy(duration)
# env._extrap_Date = env._initial_date
# stepT = 10.0
#
# thrust_mag = 30.0



# learning parameters
y = .99
e = 0.1
num_episodes = 5
# steps and rewards per episode (respectively)
j_list = []
r_list = []

# Network Model
num_inputs = 2
num_outputs = 4
layer_1_nodes = 50
layer_2_nodes = 20

# Establish feed-forward network
inputs = tf.placeholder(shape=[1,num_inputs], dtype=tf.float32)

# w1 = tf.Variable(tf.zeros[16,100])
# b1 = tf.variable(tf.zeros[100])

with tf.variable_scope('layer-1'):
    weights = tf.get_variable(name='weights-1', shape=(num_inputs, layer_1_nodes))
    layer_1_output = tf.matmul(inputs, weights)

# with tf.variable_scope('layer-2'):
#     weights = tf.get_variable(name='weights-2', shape=(layer_1_nodes, layer_2_nodes))
#     layer_2_output = tf.matmul(layer_1_output, weights)

# weights2 = tf.Variable(tf.random_uniform([16,4]
with tf.variable_scope('output'):
    weights = tf.get_variable(name='weight-out', shape=(layer_1_nodes, num_outputs))
    Q_output = tf.matmul(layer_1_output, weights)

predict = tf.argmax(Q_output, 1)

# Sum of squares loss between target and predicted Q

next_Q = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_Q-Q_output))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

with tf.variable_scope('train'):
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

        # Q-network
        while j < 99:
            j+=1

            #choose an action
            # This value is thrust
            a, allQ = sess.run([predict,Q_output],
                               feed_dict={inputs:np.identity(num_inputs)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            # Get a new state and reward
            # The state is the x-y coordinates, r =0 if not reached
            s1, r, d, _ = env.step(a[0])
            # Obtain the Q value
            Q1 = sess.run(Q_output, feed_dict={inputs:np.identity(num_inputs)[s1:s1+1]})
            # Get maxQ and set target value for chosen action
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1

            # Train the NN using target and predicted Q values
            _, W1 = sess.run([update, weights],
                             feed_dict={inputs:np.identity(num_inputs)[s:s+1],next_Q:targetQ})
            rall += r
            s = s1
            if d  == True:
                # Random action
                e = 1.0/((i/50) + 10)
                break
        j_list.append(j)
        r_list.append(rall)

        if i % 100 == 0:
            env.render_plots()

    print('KEY:\nSFFF(S: starting point, safe)\n',
          'FHFH(F: frozen surface, safe)\n',
          'FFFH(H: hole, fall to your doom)\n',
          'HFFG(G: goal, where the frisbee is located)')

    print("successful episodes " + str(sum(r_list)/num_episodes) + "%")
    print("Score: " + str(sum(r_list)))
    plt.plot(r_list)
    plt.show()





