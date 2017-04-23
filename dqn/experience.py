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
        # TODO: Remove
        assert np.all((self.rewards == 0) | (self.rewards == 1))
        assert np.all((self.terminals == 0) | (self.terminals == 1))
        idx = np.random.randint(0, self.size - count + 1)
        idx = slice(idx, idx + count)
        return (self.prestates[idx], self.actions[idx], self.rewards[idx],
                self.poststates[idx], self.terminals[idx])


class Experience2(object):

    def __init__(self, size, state_shape=None):
        self.size = size
        self.state_shape = state_shape
        self.buffer = []

    def add(self, *data):
        if len(self.buffer) == self.size:
            idx = np.random.randint(0, self.size)
            self.buffer[idx] = data
        else:
            idx = np.random.randint(0, len(self.buffer) + 1)
            self.buffer.insert(idx, data)

    def sample(self, batch_size=1, stack=True):
        if len(self.buffer) < batch_size:
            raise ValueError('Insufficient samples in buffer!')
        idx = np.random.randint(0, len(self.buffer) - batch_size + 1)
        batch = self.buffer[idx:idx + batch_size]
        self.buffer = self.buffer[:idx] + self.buffer[idx + batch_size:]
        if stack:
            tmp = []
            for i in range(len(batch[0])):
                tmp.append(np.vstack([v[i] for v in batch]).squeeze())

            batch = tmp
        return tmp

    def __len__(self):
        return len(self.buffer)
