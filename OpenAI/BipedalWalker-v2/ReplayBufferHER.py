# https://github.com/viraat/hindsight-experience-replay/blob/master/dqn-her.ipynb
import random
from collections import namedtuple

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'goal'))

class ReplayMemory:

    def __init__(self, capacity = 1e5):
        self.capacity = capacity
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

import numpy as np

class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []

        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))

        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
