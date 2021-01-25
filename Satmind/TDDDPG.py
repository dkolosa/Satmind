import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from Satmind.tf2model import Critic, Actor
from Satmind.replay_memory import Per_Memory, Uniform_Memory

class TDDDPG():
    def __init__(self, n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, PER, GAMMA,
                 tau, batch_size, save_dir):

        self.GAMMA = GAMMA
        self.batch_size = batch_size
        self.tau = tau
        self.PER = PER
        self.policy_delay = 2

        self.save_dir = save_dir

        self.actor = Actor(n_action, action_bound, layer_1_nodes, layer_2_nodes)
        self.critic = Critic(layer_1_nodes, layer_2_nodes)
        self.critic2 = Critic(layer_1_nodes, layer_2_nodes)

        self.actor_target = Actor(n_action, action_bound, layer_1_nodes, layer_2_nodes, model_name='actor_target')
        self.critic_target = Critic(layer_1_nodes, layer_2_nodes, model_name='critic_target')
        self.critic_target2 = Critic(layer_1_nodes, layer_2_nodes,
        model_name='critic_target2')

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))
        self.critic2.compile(optimizer=Adam(learning_rate=critic_lr))
        self.actor_target.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic_target.compile(optimizer=Adam(learning_rate=critic_lr))
        self.critic_target2.compile(optimizer=Adam(learning_rate=critic_lr))


        if self.PER:
            self.memory = Per_Memory(capacity=100000)
        else:
            self.memory = Uniform_Memory(buffer_size=100000)

        self.sum_q = 0
        self.actor_loss = 0
        self.critic_loss = 0

    def train(self, j):
        # sample from memory
        if self.batch_size < self.memory.get_count:
            if self.PER:
                mem, idxs, self.isweight = self.memory.sample(self.batch_size)
            else:
                mem = self.memory.sample(self.batch_size)
            s_rep = tf.convert_to_tensor(np.array([_[0] for _ in mem]), dtype=tf.float32)
            a_rep = tf.convert_to_tensor(np.array([_[1] for _ in mem]), dtype=tf.float32)
            r_rep = tf.convert_to_tensor(np.array([_[2] for _ in mem]), dtype=tf.float32)
            s1_rep = tf.convert_to_tensor(np.array([_[3] for _ in mem]), dtype=tf.float32)
            d_rep = tf.convert_to_tensor(np.array([_[4] for _ in mem]), dtype=tf.float32)

            td_error, critic_loss = self.loss_critic(a_rep, d_rep, r_rep, s1_rep, s_rep)
           
            if j % self.policy_delay == 0:
                actor_loss = self.loss_actor(s_rep)
                self.actor_loss += np.amax(actor_loss)
            
            if self.PER:
                for i in range(self.batch_size):
                    update_error = np.array(tf.reduce_mean(td_error))
                    self.memory.update(idxs[i], update_error)

            self.sum_q += np.amax(tf.squeeze(self.critic(s_rep, a_rep), 1))
            self.critic_loss += np.amax(critic_loss)

            # update target network
            if j % self.policy_delay == 0: 
                self.update_target_network(self.actor, self.actor_target, self.tau)
                self.update_target_network(self.critic, self.critic_target, self.tau)
                self.update_target_network(self.critic, self.critic_target2,
            self.tau)

    @tf.function
    def loss_actor(self, s_rep):
        with tf.GradientTape() as tape:
            actions = self.actor(s_rep)
            actor_loss = -tf.reduce_mean(self.critic(s_rep, actions))
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        return actor_loss

    @tf.function
    def loss_critic(self,a_rep, d_rep, r_rep, s1_rep, s_rep):
        targ_actions = self.actor_target(s1_rep)
        target_q = tf.squeeze(self.critic_target(s1_rep, targ_actions), 1)
        target_q2 = tf.squeeze(self.critic_target2(s1_rep, targ_actions), 1)

        y_i = r_rep + self.GAMMA * (1 - d_rep) * tf.math.minimum(target_q, target_q2)

        with tf.GradientTape() as tape:
            q = tf.squeeze(self.critic(s_rep, a_rep), 1)
            td_error = y_i - q
            if not self.PER:
                critic_loss = tf.math.reduce_mean(tf.math.square(td_error))
            else:
                critic_loss = tf.math.reduce_mean(tf.math.square(td_error) * self.isweight)

        with tf.GradientTape() as tape2:
            q2 = tf.squeeze(self.critic2(s_rep, a_rep), 1)
            td_error2 = y_i - q2
            if not self.PER:
                critic_loss2 = tf.math.reduce_mean(tf.math.square(td_error2))
            else:
                critic_loss2 = tf.math.reduce_mean(tf.math.square(td_error2) * self.isweight)

        critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_gradient2 = tape2.gradient(critic_loss2, self.critic2.trainable_variables)

        self.critic.optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))
        self.critic2.optimizer.apply_gradients(zip(critic_gradient2,
        self.critic2.trainable_variables))

        td_error = tf.math.abs(tf.math.minimum(td_error, td_error2))
        critic_loss = tf.math.minimum(critic_loss, critic_loss2)
        return td_error, critic_loss

    def update_target_network(self, network_params, target_network_params, tau=.001):
        weights = network_params.get_weights()
        target_weights = target_network_params.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        target_network_params.set_weights(target_weights)

    def save_model(self):
        self.actor.save_weights(os.path.join(self.save_dir, self.actor.model_name))
        self.critic.save_weights(os.path.join(self.save_dir, self.critic.model_name))

    def load_model(self):
        self.actor.load_weights(os.path.join(self.save_dir, self.actor.model_name))
        self.critic.load_weights(os.path.join(self.save_dir, self.critic.model_name))
        self.actor_target.load_weights(os.path.join(self.save_dir, self.actor.model_name))
        self.critic_target.load_weights(os.path.join(self.save_dir, self.critic.model_name))
