from collections import  deque
import random

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