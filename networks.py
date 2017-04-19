import tensorflow as tf
from tensorflow.contrib import slim
import gym


class Network(object):

    def __init__(self, env, dual=False):
        self.env = env
        self.nb_action = self.env.action_space.n
        self.dual = dual

    def _build_stem(self):
        pass

    def _build(self):
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.state = tf.placeholder(tf.float32, [None], name='state')
            self._state = tf.one_hot(self.state, self.env.observation_space.n)
        else:
            self.state = tf.placeholder(
                tf.float32, [None] + list(self.env.observation_space.shape),
                name='state')
            self._state = self.state
        self._build()
        self._add_output_layers()

    def _add_output_layers(self):
        if self.dual:
            self.value = slim.fully_connected(self.stem, 1)
            self.advantage = slim.fully_connected(self.stem, self.nb_action)
            self.advantage -= tf.reduce_mean(self.advantage, axis=1,
                                             keep_dims=True)
            self.q_value = self.advantage + self.value
        else:
            self.q_value = slim.fully_connected(self.stem, self.nb_action)
        self.action = tf.argmax(self.q_value, axis=1)


class Mlp(Network):

    def __init__(self, nb_hidden=[10], *args, **kwargs):
        self.nb_hidden = nb_hidden
        super(Mlp, self).__init__(*args, **kwargs)

    def _build_stem(self):
        layer = self._state
        for idx, nb_hidden in range(len(self.nb_hidden)):
            layer = slim.fully_connected(layer, nb_hidden,
                                         name='fc%d' % (nb_hidden + 1))
        self.stem = layer
