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

    def __init__(self,
                 nb_kernel=[64, 128],
                 kernel_sizes=[3, 3],
                 pool_sizes=[2, 2],
                 nb_hidden=[1024],
                 dropout=0.1,
                 *args, **kwargs):
        self.nb_kernel = nb_kernel
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.nb_hidden = nb_hidden
        self.dropout = dropout
        super(Cnn, self).__init__(*args, **kwargs)

    def _build_stem(self):
        layer = self.prepro_state
        for idx in range(len(self.nb_kernel)):
            layer = slim.conv2d(layer, self.nb_kernel[idx],
                                self.kernel_sizes[idx])
            layer = slim.max_pool2d(layer, self.pool_sizes[idx])
        layer = slim.flatten(layer)
        for nb_hidden in self.nb_hidden:
            layer = slim.fully_connected(layer, nb_hidden)
        if self.dropout:
            layer = slim.dropout(layer, 1 - self.dropout)
        self.stem = layer
        return self.stem
