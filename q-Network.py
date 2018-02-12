import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import gym

# Create the enviornment
env = gym.make('FrozenLake-v0')

# Network Implementation

# Establish feed-forward network
inputs = tf.placeholder(shape=[1,16], dtype=tf.float32)
weights = tf.Variable(tf.random_uniform([16,4], 0, 0.01))

Q_output = tf.matmul(inputs, weights)
predict = tf.argmax(Q_output,1)

# Sum of squares loss between target and predicted Q

next_Q = tf.placeholder(shape=[1,4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_Q-Q_output))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

update = trainer.minimize(loss)


# Network Training
init = tf.initialize_all_variables()

# learning parameters
y = .99
e = 0.1
num_episodes = 100
# rewards and steps per episode
j_list = []
r_list = []

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
            a, allQ = sess.run([predict,Q_output],
                               feed_dict={inputs:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            # Get a new state and reward
            s1, r, d, _ = env.step(a[0])
            # Obtain the Q value
            Q1 = sess.run(Q_output, feed_dict={inputs:np.identity(16)[s1:s1+1]})
            # Get maxQ and set target value for chosen action
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1

            # Train the NN using target and predicted Q values
            _, W1 = sess.run([update, weights],
                             feed_dict={inputs:np.identity(16)[s:s+1],nextQ:targetQ})
            rall += r
            s = s1
            if d  == True:
                # Random action
                e = 1.0/((i/50) + 10)
                break
        j_list.append(j)
        r_list.append(rall)
    print("successful episodes" + srt(sum(r_list)/num_episodes) + "%")





