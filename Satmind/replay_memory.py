from collections import  deque
import random
import numpy as np

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

    def populate_memory(self, env, features, n_actions, thrust_values):
        """
        Populate with experiences (initial guess)
        :param env: Agent enviornment object
        # :param thrust_values: Given list of possible thrust levels
        :param stepT: Thrust step values
        :return:
        """
        state = env.reset()
        thrust_values = np.array([0.00, 0.0, -0.9])
        while env._extrap_Date.compareTo(env.final_date) <= 0:
            state_1, r, done = env.step(thrust_values)
            self.add((np.reshape(state, (features,)), np.reshape(thrust_values, (n_actions,)), r,
                      np.reshape(state_1, (features,)), done))
            state = state_1
        print("Population complete")

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