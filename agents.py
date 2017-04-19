import numpy as np


class Experience(object):

    def __init__(self, size, state_shape):
        self.size = size
        self.prestates = np.empty([size] + list(state_shape), dtype=np.float32)
        self.poststates = self.prestates.copy()
        self.actions = np.empty([size], dtype=np.int32)
        self.rewards = np.empty([size], dtype=np.float32)
        self.terminals = np.empty([size], dtype=np.int8)
        self._free_idxs = np.arange(self.size)
        np.random.shuffle(self._free_idxs)

    def is_full(self):
        return self.count == self.size

    def add(self, prestate, action, reward, poststate, terminal):
        if not self.is_full():
            idx = self.free_idxs[self.count]
            self.count += 1
        else:
            idx = np.random.uniform(0, self.count)
        self.prestates[idx] = prestate
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.poststates[idx] = poststate
        self.terminals[idx] = terminal

    def sample(self, count):
        if not self.is_full()
            raise ValueError('Buffer not yet full!')
        if count > self.size:
            raise ValueError('Insufficient samples in Buffer!')
        idx = np.random.uniform(0, self.size - count + 1)
        idx = slice(idx, idx + count)
        return (self.prestates[idx], self.actions[idx], self.rewards[idx],
                self.poststates[idx], self.terminals[idx])


class Agent(object):

    def __init__(self, pred_net, target_net,
                 experience_size=10000,
                 epsilon=0.1,
                 epsilon_min=0.0001,
                 epsilon_steps=10000,
                 learning_rate=0.001,
                 target_rate=1.0,
                 batch_size=32,
                 discount=0.99,
                 update_freq=4,
                 double_dqn=False):
        self.pred_net = pred_net
        self.target_net = target_net
        self.experience = Experience(experience_size)
        self.epison = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon - epsilon_min) / epsilon_steps
        self.learning_rate = learning_rate
        self.target_rate = target_rate
        self.batch_size = batch_size
        self.discount = discount
        self.double_dqn = double_dqn


    def explore(self, env, nb_episode,
                max_step=10000,
                ):

        nb_step_tot = 0

        for episode in range(1, nb_episode + 1):
            state = env.reset()
            step = 0
            terminal = False
            while not done and step < max_step:
                step += 1
                nb_step_tot += 1
                if nb_step_tot < self.nb_pretrain_step or \
                        np.random.rand() < self.eps:
                    action = np.random.randint(0, self.pred_net.nb_action)
                else:
                    tmp = state.reshape(1, -1)
                    action = sess.run(self.pred_net.action,
                                      feed_dict={self.pred_net.state: tmp})
                poststate, reward, terminal, info = env.step(action)
                self.experience.add(state, action, reward, poststate, terminal)
                if nb_step_tot % update_freq == 0:
                    self.update()
                state = poststate

    def update(self):
        prestates, actions, rewards, poststates, terminals = \
            self.experience.sample(self.batch_size)

        targets = self.sess.run(self.target_net.q_value,
                                feed_dict={self.target_net.state: poststates})


        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)







    def play(self, env):
        pass
