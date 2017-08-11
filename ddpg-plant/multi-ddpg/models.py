# https://github.com/openai/baselines/baselines/ddpg/models.py

import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib import rnn

class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        vars_without_optimizer = [var for var in vars if 'optimizer' not in var.name]       
        return vars_without_optimizer

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

# Initialization can't be determined temporally

class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True, time_step=5):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.time_step = time_step

    def __call__(self, obs, n_hidden=64, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # tensorflow should treat palceholder like a scalar.
            # if set batch_size to one, compute the Pi or Q values.

            # TODO USE dynamic data of different lengths ( but need to change the experience replay )
            # self.sequence_length = tf.placeholder(tf.int32, [None])

            # split into batch_size
            # TODO Dynamic data need to be splited. Currently, just use reshape can work!
            # self.split = tf.placeholder(tf.int32, [None])

            # embedding
            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            # original (batch_size*time_step, hidden) -> (batch_size, time_step, hidden)
            x = tf.nn.relu(x) # no need to extend one dimension

            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, self.time_step, shape[-1]])

            # build bidirection lstm
            lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

            # TODO use output_states for process like policy iteration ---- WOW excite ideas.
            x, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                                     dtype=tf.float32)
                                                                     # sequence_length=self.sequence_length)
            # outputs [batch_size, time_step, mum_output(num_hidden)]
            # TODO v1 split to compute different actions.
            # v2( Or one convolution Network with three channels )

            # TODO v2 turn on the batch_norm after lstm
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # x = tf.nn.relu(x)

            # number of convolution kernel. --> num_actions. default (-1, 1)
            # convert to [batch_size, time_step*n_hidden], channels_last
            x = tf.reshape(x, [-1, self.time_step*n_hidden, 1])
            x = tf.layers.conv1d(x, self.nb_actions, kernel_size = n_hidden, stride = n_hidden,
                                 kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

            # [ batch_size, time_step, nb_actions]
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True, time_step=5):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.time_step = time_step

    def __call__(self, obs, action, n_hidden=64, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # x [ batch_size*time_step, DATA_NUM]
            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            # format action to be [ batch_size*time_step, nb_actions]
            x = tf.concat([x, action], axis=-1)

            # another dense layer
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, self.time_step, shape[-1]])

            lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

            x, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)

            # TODO v2 turn on the batch_norm after lstm
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # x = tf.nn.relu(x)

            x = tf.reshape(x, [-1, self.time_step * n_hidden, 1])
            # Q value of each
            x = tf.layers.conv1d(x, 1, kernel_size=n_hidden, stride=n_hidden,
                                 kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.squeeze(x)

            p = tf.layers.conv1d(x, 1, kernel_size=n_hidden, stride=n_hidden,
                                 kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            p = tf.squeeze(p)
            if self.layer_norm:
                p = tc.layers.layer_norm(p, center=True, scale=True)
            p = tf.nn.relu(p)

            p = tf.layers.dense(p, self.time_step, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            p = tf.softmax(p)
            # TODO add the temperature
            # p = tf.softmax(p/t)
            Q = tf.reduce_sum(tf.multiply(p, x), axis=1)
        return Q

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
