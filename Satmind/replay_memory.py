from collections import  deque
import random
import numpy as np
from Satmind.utils import SumTree


class Uniform_Memory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.count = 0

    def add(self, experience):
        """
        Add an experience to the buff-er
        :param experience: (state, action, reward, next state, done)
        :return:
        """
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def smaple(self, batch_size):
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
        thrust_values = np.array([0.00001, 0.0001, -0.7])
        while env._extrap_Date.compareTo(env.final_date) <= 0:
            state_1, r, done = env.step(thrust_values)
            self.add((np.reshape(state, (features,)), np.reshape(thrust_values, (n_actions,)), r,
                      np.reshape(state_1, (features,)), done))
            state = state_1
            # kep = env.convert_to_keplerian(env._currentOrbit)
            # ta = kep.getMeanAnomaly()
            # if ta >=0:
            #     thrust_values = np.array([0.0, 0.0, 1.0])
            # else:
            #     thrust_values = np.array([0.0, 0.0, -1.0])
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


class Per_Memory:  # stored as ( s, a, r, next_state, done ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.count = 0

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):

        if self.count < self.capacity:
            p = self._get_priority(error)
            self.tree.add(p, sample)
            self.count += 1
        else:
            # self.buffer.popleft()
            p = self._get_priority(error)
            self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def pre_populate(self, env, features, n_actions, thrust_values):
        state = env.reset()
        thrust_values = np.array([0.001, 0.6, 0.0001])
        i = 0
        while env._extrap_Date.compareTo(env.final_date) <= 0:
            state_1, r, done = env.step(thrust_values)
            error = abs(r)

            self.add(error, (np.reshape(state, (features,)), np.reshape(thrust_values, (n_actions,)), r,
                      np.reshape(state_1, (features,)), done))
            # if i % 2 == 0:
            #     thrust_values = np.array([0.0, 0.0, 0.6])
            # else:
            #     thrust_values = np.array([0.0, 0.0, -0.6])
            # state = state_1
            # i += 1
        print("Population complete")
