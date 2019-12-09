import tensorflow as tf
import numpy as np
import gym
import gym.spaces
import matplotlib.pyplot as plt
from Satmind.actor_critic import Actor, Critic
from Satmind.utils import OrnsteinUhlenbeck, AdaptiveParamNoiseSpec
from Satmind.replay_memory import Per_Memory, Uniform_Memory


def test_training():
    """Test if training has taken place (update of weights)"""

    ENV = 'Pendulum-v0'
    env = gym.make(ENV)
    features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bound = env.action_space.high

    actor = Actor(features, n_actions, 128, 128, action_bound, 0.0001, .001,1, 'actor')
    critic = Critic(features, n_actions, 128, 128, 0.001, 0.001,'critic', actor.trainable_variables)

    s = env.reset()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        before = actor.trainable_variables

        a = actor.predict(np.reshape(s, (1, features)), sess)
        s1, r, done, _ = env.step(a[0])

        grad = critic.action_gradient(s, a, sess)
        actor.train(s, grad[0], sess)

        after = actor.trainable_variables
        for b, a, n in zip(before, after):
        # Make sure something changed
            assert (b != a).any()


def pre_train(critic, actor, env, features, n_actions, sess):
    GAMMA = 0.99
    # Heurisic: suboptimal, have no notion of balance.
    for _ in range(10):
        s0 = env.reset()
        steps = 0
        total_reward = 0
        a = np.array([0.0, 0.0, 0.0, 0.0])
        STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
        SPEED = 0.29  # Will fall forward on higher speed
        state = STAY_ON_ONE_LEG
        moving_leg = 0
        supporting_leg = 1 - moving_leg
        SUPPORT_KNEE_ANGLE = +0.1
        supporting_knee_angle = SUPPORT_KNEE_ANGLE
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            steps += 1

            moving_s_base = 4 + 5*moving_leg
            supporting_s_base = 4 + 5*supporting_leg

            hip_targ  = [None,None]   # -0.8 .. +1.1
            knee_targ = [None,None]   # -0.6 .. +0.9
            hip_todo  = [0.0, 0.0]
            knee_todo = [0.0, 0.0]

            if state==STAY_ON_ONE_LEG:
                hip_targ[moving_leg]  = 1.1
                knee_targ[moving_leg] = -0.6
                supporting_knee_angle += 0.03
                if s[2] > SPEED: supporting_knee_angle += 0.03
                supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
                knee_targ[supporting_leg] = supporting_knee_angle
                if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                    state = PUT_OTHER_DOWN
            if state==PUT_OTHER_DOWN:
                hip_targ[moving_leg]  = +0.1
                knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
                knee_targ[supporting_leg] = supporting_knee_angle
                if s[moving_s_base+4]:
                    state = PUSH_OFF
                    supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
            if state==PUSH_OFF:
                knee_targ[moving_leg] = supporting_knee_angle
                knee_targ[supporting_leg] = +1.0
                if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                    state = STAY_ON_ONE_LEG
                    moving_leg = 1 - moving_leg
                    supporting_leg = 1 - moving_leg

            if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
            if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
            if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
            if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

            hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
            hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
            knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
            knee_todo[1] -= 15.0*s[3]

            a[0] = hip_todo[0]
            a[1] = knee_todo[0]
            a[2] = hip_todo[1]
            a[3] = knee_todo[1]
            a = np.clip(0.5*a, -1.0, 1.0)

            # env.render()

            s0 = np.reshape(s, (1, features))

            target_q = critic.predict_target(np.reshape(s, (1,features)), np.reshape(a, (1, n_actions)), sess)

            y_i = []
            if done:
                y_i.append(r)
            else:
                y_i.append(r + GAMMA * target_q)

            # update the critic network
            error, predicted_q, _ = critic.train(s0, np.reshape(a, (1, n_actions)), np.reshape(y_i, (1,1)), np.reshape(1, (1,1)), sess)


            a_output = actor.predict(s0, sess)
            grad = critic.action_gradient(s0, a_output, sess)
            actor.train(s0, grad[0], sess)

            # update target network
            critic.update_target_network(sess)
            actor.update_target_network(sess)
            if done: break


def test_rl():
    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v2', 'LunarLanderContinuous-v2',
            'BipedalWalkerHardcore-v2')

    ENV = ENVS[0]
    env = gym.make(ENV)
    iter_per_episode = 200
    features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bound = env.action_space.high

    env.seed(1234)
    np.random.seed(1234)

    # num_episodes = 250
    num_episodes = 1001

    batch_size = 100
    #Pendulum
    # layer_1_nodes, layer_2_nodes = 250, 150
    #lander
    # layer_1_nodes, layer_2_nodes = 450, 300
    #Walker
    # layer_1_nodes, layer_2_nodes = 500, 400
    layer_1_nodes, layer_2_nodes = 500, 400

    tau = 0.001
    actor_lr, critic_lr = 0.0001, 0.001
    GAMMA = 0.99

    actor = Actor(features, n_actions, layer_1_nodes, layer_2_nodes, action_bound, tau, actor_lr, batch_size,'actor')
    actor_noise = OrnsteinUhlenbeck(np.zeros(n_actions))
    noise_decay = 1.0

    critic = Critic(features, n_actions, layer_1_nodes, layer_2_nodes, critic_lr, tau, 'critic', actor.trainable_variables)
    PER = True

    # Replay memory buffer
    if PER:
        memory = Per_Memory(capacity=100000)
    else:
        memory = Uniform_Memory(buffer_size=1000)

    saver = tf.compat.v1.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        actor.update_target_network(sess)
        critic.update_target_network(sess)

        # Run one training loop (biped-walker only)
        # if ENV == 'BipedalWalker-v2': pre_train(critic, actor, env, features, n_actions, sess)
        rewards = []
        noise_decay = 1
        for i in range(num_episodes):
            s = env.reset()
            sum_reward = 0
            sum_q = 0
            j = 0

            noise_decay = np.clip(noise_decay-0.001,0.01,1)

            while True:

                env.render()

                a = actor.predict(np.reshape(s, (1, features)), sess) + actor_noise()*noise_decay
                s1, r, done, _ = env.step(a[0])

                # Store in replay memory
                if PER:
                    error = abs(r)  # D_i = max D
                    memory.add(error, (np.reshape(s, (features,)), np.reshape(a[0], (n_actions,)), r, np.reshape(s1, (features,)), done))
                else:
                    memory.add((np.reshape(s, (features,)), np.reshape(a[0], (n_actions,)), r, np.reshape(s1,(features,)), done))

                # sample from memory
                # if batch_size < memory.get_count:
                    # mem = memory.sample(batch_size)
                    # s_rep = np.array([_[0] for _ in mem])
                    # a_rep = np.array([_[1] for _ in mem])
                    # r_rep = np.array([_[2] for _ in mem])
                    # s1_rep = np.array([_[3] for _ in mem])
                    # d_rep = np.array([_[4] for _ in mem])

                if batch_size < memory.count:
                    mem, idxs, isweight = memory.sample(batch_size)
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
                    error, predicted_q, _ = critic.train(s_rep, a_rep, np.reshape(y_i, (batch_size,1)), np.reshape(isweight, (batch_size,1)), sess)

                    for n in range(batch_size):
                        idx = idxs[n]
                        memory.update(idx, abs(error[n][0]))

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
                j += 1
                if done:
                    print('Episode: {}, reward: {}, Q_max: {}'.format(i, int(sum_reward), sum_q/float(j)))
                    # rewards.append(sum_reward)
                    print('===========')
                    saver.save(sess, 'test_walkder/model.ckpt')
                    break


if __name__ == '__main__':
    test_rl()