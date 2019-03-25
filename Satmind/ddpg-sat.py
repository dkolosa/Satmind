import numpy as np
import tensorflow as tf
from collections import  deque
from env_orekit import OrekitEnv
import gym
import tflearn
import matplotlib.pyplot as plt
import random

class Actor:

    def __init__(self, features, n_actions, layer_1_nodes, layer_2_nodes, action_bound, tau, learning_rate, batch_size, name):

        self.tau = tau
        self.action_bound = action_bound
        self.features = features
        self.n_actions = n_actions

        # create the actor network and target network
        self.input, self.output, self.scaled_output = self.build_network(features, n_actions, layer_1_nodes, layer_2_nodes, name)
        self.network_parameters = tf.trainable_variables()
        self.target_input, self.target_output, self.target_scaled_output = self.build_network(features, n_actions, layer_1_nodes, layer_2_nodes, name='target_actor_')
        self.target_network_parameters = tf.trainable_variables()[len(self.network_parameters):]

        # This is retrieved from the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, n_actions])

        self.unnorm_actor_grad = tf.gradients(self.scaled_output, self.network_parameters, -self.action_gradient)
        self.actor_gradient = list(map(lambda x: tf.div(x, batch_size), self.unnorm_actor_grad))

        self.train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.actor_gradient, self.network_parameters))

        self.update_target_network_parameters = [self.target_network_parameters[i].assign(tf.multiply(self.network_parameters[i], self.tau) +
                                                 tf.multiply(self.target_network_parameters[i], 1. - self.tau))
         for i in range(len(self.target_network_parameters))]

        self.trainable_variables = len(self.network_parameters) + len(self.target_network_parameters)

    def build_network(self, features, n_actions, layer_1_nodes, layer_2_nodes,name):
        input = tf.placeholder(tf.float32, shape=[None, features])
        with tf.variable_scope(str(name) + '_layer_1'):
            layer_1 = tf.layers.dense(inputs=input,
                                      units=layer_1_nodes,
                                      activation=tf.nn.relu,
                                      )

        with tf.variable_scope(str(name) + '_layer_2'):
            layer_2 = tf.layers.dense(inputs=layer_1,
                                      units=layer_2_nodes,
                                      activation=tf.nn.relu,
                                      )

        with tf.variable_scope(str(name) + '_output'):
            output = tf.layers.dense(inputs=layer_2,
                                          units=n_actions,
                                          activation=tf.nn.tanh,
                                          kernel_initializer=tf.random_uniform_initializer(-0.003,0.003),
                                          use_bias=False,
                                     )

        scaled_output = tf.multiply(output, self.action_bound)

        return input, output, scaled_output

    def predict(self, state, sess):
        action = sess.run(self.scaled_output, {self.input: state})
        return action

    def predict_target(self, state, sess):
        action = sess.run(self.target_scaled_output, {self.target_input: state})
        return action

    def train(self, state, action_gradient, sess):
        sess.run(self.train_op, {self.input: state, self.action_gradient: action_gradient})

    def update_target_network(self, sess):
        """
        theta^u' <- tau*theta^u + (1-tau)theta^u'
        :return:
        """
        sess.run(self.update_target_network_parameters)

class Critic:

    def __init__(self, n_features, n_actions, layer_1_nodes, layer_2_nodes, learning_rate, tau, name, actor_trainable_variables):

        self.tau = tau
        self.n_features = n_features
        self.n_actions = n_actions

        self.input, self.action, self.output  = self.build_network(n_features, n_actions, layer_1_nodes, layer_2_nodes, name)
        self.network_parameters = tf.trainable_variables()[actor_trainable_variables:]

        self.input_target, self.action_target, self.output_target = self.build_network(n_features, n_actions, layer_1_nodes, layer_2_nodes, name='target_critic_')
        self.target_network_parameters = tf.trainable_variables()[(len(self.network_parameters) + actor_trainable_variables):]

        self.q_value = tf.placeholder(tf.float32, shape=[None, 1])

        self.update_target_network_parameters = \
            [self.target_network_parameters[i].assign(tf.multiply(self.network_parameters[i], self.tau) \
                                                  + tf.multiply(self.target_network_parameters[i], 1. - self.tau))
             for i in range(len(self.target_network_parameters))]

        self.loss = tf.losses.mean_squared_error(self.output, self.q_value)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # the action-value gradient to be used be the actor network
        self.action_grad = tf.gradients(self.output, self.action)

    def build_network(self, n_features, n_actions, layer_1_nodes, layer_2_nodes, name):
        l1_init = tf.random_normal_initializer(0.0, 0.1)
        b1_init = tf.constant_initializer(0.1)

        input = tf.placeholder(tf.float32, shape=[None, n_features])
        action = tf.placeholder(tf.float32, shape=[None, n_actions])

        layer_1 = tf.contrib.layers.fully_connected(input, layer_1_nodes)

        t1 = tflearn.fully_connected(layer_1, layer_2_nodes)
        t2 = tflearn.fully_connected(action, layer_2_nodes)

        layer_2 = tf.nn.relu(tf.matmul(layer_1, t1.W) + tf.matmul(action, t2.W) + t2.b)
        #
        # with tf.variable_scope(str(name) + '_layer_1'):
        #     input_weight = tf.get_variable('l1_state', [n_features, layer_1_nodes], initializer=l1_init)
        #
        #     action_weight = tf.get_variable('l1_action', [n_actions, layer_1_nodes], initializer=l1_init)
        #     bias = tf.get_variable('l1_bias', [1, layer_1_nodes], initializer=b1_init)
        #     layer_1 = tf.nn.relu(tf.matmul(input, input_weight) + tf.matmul(action, action_weight) + bias)

        with tf.variable_scope(str(name) + '_output'):
            output = tf.layers.dense(inputs=layer_2,
                                     units=1,
                                     activation=None,
                                     use_bias=False,
                                     kernel_initializer=tf.random_uniform_initializer(-0.003,0.003)
                                          )

        return input, action, output

    def predict(self, state, action, sess):
        return sess.run(self.output, {self.input: state,
                                      self.action: action})

    def predict_target(self, state, action, sess):
        return sess.run(self.output_target, {self.input_target: state,
                                             self.action_target: action})

    def train(self, state, action, q_value, sess):
        return sess.run([self.output, self.train_op], {self.input: state,
                                                       self.action: action,
                                                       self.q_value: q_value})

    def action_gradient(self, state, action, sess):
        return sess.run(self.action_grad, {self.input: state,
                                           self.action: action})

    def update_target_network(self, sess):
        """
        theta^Q' <- tau*theta^Q + (1-tau)theta^Q'
        where theta are all of the weights of the target network
        :return:
        """
        sess.run(self.update_target_network_parameters)


class OrnsteinUhlenbeck():
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class Experience:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.count = 0

    def add(self, experience):
        """
        Add an experience to the buff-er
        :param experience: (state, action, reward, next state)
        :return:
        """
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)


    def experience_replay(self, batch_size):
        """
        Get a random experience from the deque
        :return:  experience: (state, action, reward, next state, terminal(done))
        """
        if self.count < batch_size:
            return random.sample(self.buffer, self.count)
        else:
            return random.sample(self.buffer, batch_size)

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
            act = np.random.random_sample()*thrust_values
            state_1, reward, done_mem, _ = env.state(act, stepT)
            e = [state, act, reward, state_1]
            self.add(e)
            state = state_1

    @property
    def get_count(self):
        return self.count

    @property
    def print_buffer(self):
        '''
        Prints all of the experience data stored in the buffer

        :return: Printed list of the experience in the buffer
        '''
        for e in self.buffer: return e


def orekit_setup():
    # initialize enviornment
    year, month, day, hr, minute, sec = 2018, 8, 1, 9, 30, 00.00
    date = [year, month, day, hr, minute, sec]

    mass = 1000.0
    fuel_mass = 500.0
    duration = 2 * 24.0 * 60.0 ** 2

    # initial state
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

    stepT = 1000.0

    env = OrekitEnv(state, state_targ, date, duration, mass, fuel_mass, stepT)
    return env

def main():

    env = gym.make('Pendulum-v0')
    # env = gym.make('MountainCarContinuous-v0')
    # env = orekit_setup()

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 800
    iter_per_episode = 200
    batch_size = 200

    stepT = 100.0

    # Network inputs and outputs
    # features = 2
    # n_actions = 1
    # action_bound = 1

    features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bound = env.action_space.high

    layer_1_nodes, layer_2_nodes = 600, 500
    tau = 0.001
    actor_lr, critic_lr = 0.0001, 0.001
    GAMMA = 0.99

    # Initialize actor and critic network and targets
    actor = Actor(features, n_actions, layer_1_nodes, layer_2_nodes, action_bound, tau, actor_lr, batch_size, 'actor')
    actor_noise = OrnsteinUhlenbeck(np.zeros(n_actions))
    critic = Critic(features, n_actions, layer_1_nodes, layer_2_nodes, critic_lr, tau, 'critic', actor.trainable_variables)

    # Replay memory buffer
    replay = Experience(buffer_size=2000)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        actor.update_target_network(sess)
        critic.update_target_network(sess)

        for i in range(num_episodes):
            s = env.reset()
            sum_reward = 0
            sum_q = 0

            actions = []

            for j in range(iter_per_episode):

                env.render()

                # Select an action
                # a = abs(np.linalg.norm(actor.predict(s, sess) + actor_noise()))
                a = actor.predict(np.reshape(s, (1, features)), sess) + actor_noise()

                # Observe state and reward
                s1, r, done, _ = env.step(a[0])

                # actions.append(a)
                # Store in replay memory
                replay.add((np.reshape(s, (features,)), np.reshape(a, (n_actions,)), r, np.reshape(s1,(features,)), done))
                # sample from random memory
                if batch_size < replay.get_count:
                    mem = replay.experience_replay(batch_size)
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
                    predicted_q, _ = critic.train(s_rep, a_rep, np.reshape(y_i, (batch_size,1)), sess)
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
                if done:
                    print('Episode: {}, reward: {}, Q_max: {}'.format(i, int(sum_reward), sum_q/float(j)))
                    # print('a:' + str(env._currentOrbit.getA()) + 'ecc: ' + str(env._currentOrbit.getE()))
                    print('===========')
                    break
            # if i % 20 == 0:
            #     pos_x = env._targetOrbit.getPVCoordinates().getPosition().getX()
            #     pos_y = env._targetOrbit.getPVCoordinates().getPosition().getY()
            #     pos = q = np.column_stack((env._px, env._py)) / 1e3
            #     plt.subplot(2, 1, 1)
            #     plt.title('Episode: ' + str(i) + ' a_final:' + str(int(env._currentOrbit.getA() / 1e3)) + ' km')
            #     plt.plot(pos[:, 0], pos[:, 1], 'b-', pos_x / 1e3, pos_y / 1e3, 'ro')
            #     plt.xlabel('km')
            #     plt.ylabel('km')
            #     plt.subplot(2, 1, 2)
            #     plt.plot(actions)
            #     plt.xlabel('Mission Step ' + str(stepT) + 'sec per step')
            #     plt.ylabel('Thrust (N)')
            #     plt.tight_layout()
            # plt.show()



if __name__ == "__main__":
    main()