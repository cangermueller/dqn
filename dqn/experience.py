import numpy as np


class Experience(object):

    def __init__(self, size, state_shape=None):
        self.size = size
        shape = [size]
        if state_shape is not None:
            shape += list(state_shape)
        self.prestates = np.empty(shape, dtype=np.float32)
        self.poststates = self.prestates.copy()
        self.actions = np.empty([size], dtype=np.int32)
        self.rewards = np.empty([size], dtype=np.float32)
        self.terminals = np.empty([size], dtype=np.int8)
        self.idx = 0
        self._is_full = False
        self.shuffle = np.arange(self.size)

    def __len__(self):
        return self.size

    def is_full(self):
        return self._is_full

    def add(self, prestate, action, reward, poststate, terminal):
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.poststates[self.idx] = poststate
        self.terminals[self.idx] = terminal
        self.idx += 1
        if self.idx >= self.size:
            self.idx = 0
            self._is_full = True

    def sample(self, count):
        if not self.is_full():
            raise ValueError('Buffer not yet full!')
        if count > self.size:
            raise ValueError('Insufficient samples in Buffer!')
        np.random.shuffle(self.shuffle)
        idx = self.shuffle[:count]
        return (self.prestates[idx], self.actions[idx], self.rewards[idx],
                self.poststates[idx], self.terminals[idx])
