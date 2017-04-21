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
        self._free_idxs = np.arange(self.size)
        np.random.shuffle(self._free_idxs)

    def is_full(self):
        return len(self._free_idxs) == 0

    def add(self, prestate, action, reward, poststate, terminal):
        if not self.is_full():
            idx = self._free_idxs[-1]
            self._free_idxs = self._free_idxs[:-1]
        else:
            idx = np.random.randint(0, self.size)
        self.prestates[idx] = prestate
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.poststates[idx] = poststate
        self.terminals[idx] = terminal

    def sample(self, count):
        if not self.is_full():
            raise ValueError('Buffer not yet full!')
        if count > self.size:
            raise ValueError('Insufficient samples in Buffer!')
        idx = np.random.randint(0, self.size - count + 1)
        idx = slice(idx, idx + count)
        return (self.prestates[idx], self.actions[idx], self.rewards[idx],
                self.poststates[idx], self.terminals[idx])
