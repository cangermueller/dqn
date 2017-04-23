import tensorflow as tf
from tensorflow.contrib import slim


class Network(object):

    def __init__(self, state, nb_action, prepro_state=None, dual=False,
                 *args, **kwargs):
        self.state = state
        if prepro_state is None:
            self.prepro_state = self.state
        else:
            self.prepro_state = prepro_state
        self.nb_action = nb_action
        self.dual = dual
        self._build()

    def _build_stem(self):
        pass

    def _build(self):
        self._build_stem()
        self._add_output_layers()
        self.trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            tf.get_variable_scope().name)

    def _add_output_layers(self):
        if self.dual:
            self.value = slim.fully_connected(self.stem, 1,
                                              activation_fn=None)
            self.advantage = slim.fully_connected(self.stem, self.nb_action,
                                                  activation_fn=None)
            self.advantage -= tf.reduce_mean(self.advantage, axis=1,
                                             keep_dims=True)
            self.q_value = self.advantage + self.value
        else:
            self.q_value = slim.fully_connected(self.stem, self.nb_action,
                                                activation_fn=None)
        self.action = tf.argmax(self.q_value, axis=1)


class Mlp(Network):

    def __init__(self, nb_hidden=[10], *args, **kwargs):
        self.nb_hidden = nb_hidden
        super(Mlp, self).__init__(*args, **kwargs)

    def _build_stem(self):
        layer = self.prepro_state
        for nb_hidden in self.nb_hidden:
            layer = slim.fully_connected(layer, nb_hidden,
                                         activation_fn=tf.nn.relu)
        self.stem = layer
        return self.stem


class Cnn(Network):

    def __init__(self, nb_kernels=[64, 128], kernel_sizes=[3, 3]):
        self.nb_kernels = nb_kernels
        self.kernel_sizes = kernel_sizes

    def _build_stem(self):
        layer = self.prepro_state
        for idx in range(len(self.nb_kernels)):
            layer = slim.conv2d(layer, self.nb_kernels[idx],
                                self.kernel_sizes[idx])
            layer = slim.max_pool2d(layer, 2)
        self.stem = slim.flatten(layer)
        return self.stem
