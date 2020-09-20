import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import gym
import gym.spaces
# import matplotlib.pyplot as plt
from Satmind.tf2model import Critic, Actor
from Satmind.utils import OrnsteinUhlenbeck, AdaptiveParamNoiseSpec
from Satmind.replay_memory import Per_Memory, Uniform_Memory
import os, datetime


def test_rl():
    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v3', 'LunarLanderContinuous-v2',
            'BipedalWalkerHardcore-v3')

    ENV = ENVS[0]

    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV), exist_ok=True)
    save_dir = os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV)

    env = gym.make(ENV)
    iter_per_episode = 200
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]
    action_bound = env.action_space.high

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 1001
    PER = False

    batch_size = 128
    #Pendulum
    layer_1_nodes, layer_2_nodes = 512, 256
    #lander

    tau = 0.01
    actor_lr, critic_lr = 0.0001, 0.001
    GAMMA = 0.99
    ep = 0.001

    actor_noise = OrnsteinUhlenbeck(np.zeros(n_action))

    agent = DDPG(n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, PER, GAMMA,
                 tau, batch_size, save_dir)

    agent.update_target_network(agent.actor, agent.actor_target, agent.tau)
    agent.update_target_network(agent.critic, agent.critic_target, agent.tau)

    load_models = True
    save = False
    # If loading model, a gradient update must be called once before loading weights
    if load_models:
        load_model(PER, actor_noise, agent, batch_size, env, ep, n_action, n_state)

    for i in range(num_episodes):
        s = env.reset()
        sum_reward = 0
        sum_q = 0
        j = 0

        while True:
            env.render()

            a = agent.actor(tf.convert_to_tensor([s], dtype=tf.float32))[0] + actor_noise()
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

            sum_reward += r
            s = s1
            j += 1
            if done:
                print('Episode: {}, reward: {}, q_max: {}'.format(i, int(sum_reward), sum_q))
                # rewards.append(sum_reward)
                print('===========')
                if save:
                    agent.save_model()
                break


def load_model(PER, actor_noise, agent, batch_size, env, ep, n_action, n_state):
    for i in range(batch_size + 1):
        s = env.reset()
        a = agent.actor(tf.convert_to_tensor([s], dtype=tf.float32))[0] + actor_noise()
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


class DDPG():
    def __init__(self, n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, PER, GAMMA,
                 tau, batch_size, save_dir):

        self.GAMMA = GAMMA
        self.batch_size = batch_size
        self.tau = tau
        self.PER = PER

        self.save_dir = save_dir

        self.actor = Actor(n_action, action_bound, layer_1_nodes, layer_2_nodes)
        self.critic = Critic(layer_1_nodes, layer_2_nodes)

        self.actor_target = Actor(n_action, action_bound, layer_1_nodes, layer_2_nodes, model_name='actor_target')
        self.critic_target = Critic(layer_1_nodes, layer_2_nodes, model_name='critic_target')

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))
        self.actor_target.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic_target.compile(optimizer=Adam(learning_rate=critic_lr))

        if self.PER:
            self.memory = Per_Memory(capacity=100000)
        else:
            self.memory = Uniform_Memory(buffer_size=100000)

    def train(self):
        # sample from memory
        if self.batch_size < self.memory.get_count:
            mem = self.memory.sample(self.batch_size)
            s_rep = tf.convert_to_tensor(np.array([_[0] for _ in mem]), dtype=tf.float32)
            a_rep = tf.convert_to_tensor(np.array([_[1] for _ in mem]), dtype=tf.float32)
            r_rep = tf.convert_to_tensor(np.array([_[2] for _ in mem]), dtype=tf.float32)
            s1_rep = tf.convert_to_tensor(np.array([_[3] for _ in mem]), dtype=tf.float32)
            d_rep = tf.convert_to_tensor(np.array([_[4] for _ in mem]), dtype=tf.float32)

            self.loss_critic(a_rep, d_rep, r_rep, s1_rep, s_rep)
            self.loss_actor(s_rep)

            # update target network
            self.update_target_network(self.actor, self.actor_target, self.tau)
            self.update_target_network(self.critic, self.critic_target, self.tau)

    @tf.function
    def loss_actor(self, s_rep):
        with tf.GradientTape() as tape:
            actions = self.actor(s_rep)
            actor_loss = -tf.reduce_mean(self.critic(s_rep, actions))
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    @tf.function
    def loss_critic(self,a_rep, d_rep, r_rep, s1_rep, s_rep):
        targ_actions = self.actor_target(s1_rep)
        target_q = tf.squeeze(self.critic_target(s1_rep, targ_actions), 1)
        y_i = r_rep + self.GAMMA * target_q * (1 - d_rep)
        with tf.GradientTape() as tape:
            q = tf.squeeze(self.critic(s_rep, a_rep), 1)
            td_error = q - y_i
            critic_loss = tf.math.reduce_mean(tf.math.square(td_error))
        critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))

    def update_target_network(self, network_params, target_network_params, tau=.001):
        weights = network_params.get_weights()
        target_weights = target_network_params.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        target_network_params.set_weights(target_weights)

    def save_model(self):
        self.actor.save_weights(os.path.join(self.save_dir, self.actor.model_name))
        self.critic.save_weights(os.path.join(self.save_dir, self.critic.model_name))
        self.actor_target.save_weights(os.path.join(self.save_dir, self.actor_target.model_name))
        self.critic_target.save_weights(os.path.join(self.save_dir, self.critic_target.model_name))

    def load_model(self):
        self.actor.load_weights(os.path.join(self.save_dir, self.actor.model_name))
        self.critic.load_weights(os.path.join(self.save_dir, self.critic.model_name))
        self.actor_target.load_weights(os.path.join(self.save_dir, self.actor_target.model_name))
        self.critic_target.load_weights(os.path.join(self.save_dir, self.critic_target.model_name))


if __name__ == '__main__':
    test_rl()