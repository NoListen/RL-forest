# https://github.com/openai/baselines/baselines/ddpg/models.py

import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib import rnn
import numpy as np


# cover 2d and 3d
def get_w_bound(filter_shape):
    return np.sqrt(6./(np.prod(filter_shape[:-2]))*np.sum(filter_shape[-2:]))


# modified from https://github.com/openai/universe-starter-agent/model.py
def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1),
           pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
        w_bound = get_w_bound(filter_shape)
        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


# modified from https://github.com/openai/universe-starter-agent/model.py
def conv3d(x, num_filters, name, filter_size=(1, 3, 3), stride=(1, 1, 1),
           pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope( name):
        stride_shape = [1, stride[0], stride[1], stride[2], 1]
        filter_shape = [filter_size[0], filter_size[1], filter_size[2], int(x.get_shape()[4]), num_filters]
        w_bound = get_w_bound(filter_shape)
        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv3d(x, w, stride_shape, pad) + b


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

# simple
class Dynamic_Actor(Model):
    def __init__(self, nb_unit_actions, name='actor', layer_norm=True, time_step=5):
        super(Dynamic_Actor, self).__init__(name=name)
        self.nb_unit_actions = nb_unit_actions
        self.layer_norm = layer_norm
        self.time_step = time_step

    # au alive units.
    def __call__(self, ud, mask, au, n_hidden=64, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = ud
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)

            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)  # no need to extend one dimension

            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, self.time_step, shape[-1]])
            # build bidirection lstm
            lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            x, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32,
                                                   sequence_length=au)
            x = tf.concat(x, 2)

            # TODO v2 turn on the batch_norm after lstm
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.reshape(x, [-1, self.time_step * n_hidden * 2, 1])
            x = tf.layers.conv1d(x, self.nb_unit_actions, kernel_size=n_hidden * 2, strides=n_hidden * 2,
                                 kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x

class Dynamic_Critic(Model):
    def __init__(self, name='critic', layer_norm=True, time_step=5):
        super(Dynamic_Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.time_step = time_step

    def __call__(self, ud, action, mask, au,  n_hidden=64, reuse=False, unit_data = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # x [ batch_size*time_step, DATA_NUM]
            x = ud
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
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
                                                   dtype=tf.float32,
                                                   sequence_length=au)
            x = tf.concat(x, 2)
            # TODO v2 turn on the batch_norm after lstm
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.reshape(x, [-1, self.time_step * n_hidden * 2, 1])
            # Q value of each
            q = tf.layers.conv1d(x, 1, kernel_size=n_hidden*2, strides=n_hidden*2,
                                 kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            q = tf.squeeze(q, [-1])

            # p = tf.layers.conv1d(x, 1, kernel_size=n_hidden*2, strides=n_hidden*2,
            #                      kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            # p = tf.squeeze(p, [-1])
            # if self.layer_norm:
            #     p = tc.layers.layer_norm(p, center=True, scale=True)
            # p = tf.nn.relu(p)
            #
            # p = tf.layers.dense(p, self.time_step, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            # p = tf.nn.softmax(p)
            Q_mask = tf.multiply(q, mask)
            #print(mask.get_shape().as_list(), pQ_mask.get_shape().as_list(), "mask")
            Q = tf.reduce_sum(Q_mask, axis=1, keep_dims=True)
            #print(Q.get_shape().as_list, "Q")
        if unit_data:
            return Q, Q_mask
        return Q

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars

class Conv_Actor(Model):
    def __init__(self, nb_unit_actions, name='actor', layer_norm=True, time_step=5):
        super(Conv_Actor, self).__init__(name=name)
        self.nb_unit_actions = nb_unit_actions
        self.layer_norm = layer_norm
        self.time_step = time_step

    # obs is a set of vector.
    # Concatenate the unit location and the feature map after the first convolutional layer.
    #TODO no graident from the critic would result in no gradient in the actor too ???
    # Are there something correlated.

    def __call__(self, obs, unit_locations, n_hidden=256, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # TODO USE dynamic data of different lengths ( but need to change the experience replay )
            # self.sequence_length = tf.placeholder(tf.int32, [None])

            # split into batch_size
            # TODO Dynamic data need to be splited. Currently, just use reshape can work!
            # self.split = tf.placeholder(tf.int32, [None])

            # embedding
            x = obs
            # [batch_size, myself_num, ms, ms, 1]
            u = unit_locations
            u_shape = u.get_shape().as_list()
            # tf.shape maybe also ok
            assert (u_shape[1] == self.time_step)
            # TODO try different initializer.
            # the hyper-parameters need to be adjusted

            # 40 -> 20
            x = conv2d(x, 24, "conv1", (3, 3), (2, 2)) # 4 map
            u = conv3d(u, 8, "u_conv1", (1, 3, 3), (1, 2, 2)) # 1 map

            # u [batch_size*myself_num, ms/2, ms/2, 1] -> [batch_size, myself, ms/2, ms/2, 1]
            # x [batch_size, ms/2, ms/2, c]
            u_list = tf.split(u, self.time_step, axis=1)
            u_list = [tf.squeeze(unit, [1]) for unit in u_list]
            ux = tf.stack([tf.concat([x, unit], -1) for unit in u_list], axis=1)

            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)
            # 20 -> 10
            ux = conv3d(ux, 32, "conv2", (1, 3, 3), (1, 2, 2))

            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            # 10 -> 5.
            ux = conv3d(ux, 32, "conv3", (1, 3, 3), (1, 2, 2))

            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            #TODO add one (1,1,1) layer to reorganize the features.
            ux_shape = ux.get_shape().as_list()
            ux = tf.reshape(ux, (-1, self.time_step, int(np.prod(ux_shape[2:]))))

            # TODO is it necessary to add one layer here???
            # ux = tf.layers.dense(ux, n_hidden)
            #
            # if self.layer_norm:
            #     ux = tc.layers.layer_norm(ux, center=True, scale=True)
            # ux = tf.nn.relu(ux) # no need to extend one dimension

            # build bidirection lstm
            lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

            # TODO use output_states for process like policy iteration ---- WOW exciting ideas.
            ux, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, ux,
                                                                     dtype=tf.float32)
                                                                     # sequence_length=self.sequence_length)
            ux = tf.concat(ux, 2)

            # TODO v2 turn on the batch_norm after lstm
            if self.layer_norm:
                 ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            ux = tf.layers.dense(ux, 256, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)
            # number of convolution kernel. --> num_actions. default (-1, 1)
            # convert to [batch_size, time_step*n_hidden], channels_last
            ux = tf.reshape(ux, [-1, self.time_step*256, 1])
            ux = tf.layers.conv1d(ux, self.nb_unit_actions, kernel_size = 256, strides = 256,
                                 kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            # [ batch_size, time_step, nb_actions]
            ux = tf.nn.tanh(ux)
        return ux


class Conv_Critic(Model):
    def __init__(self, name='critic', layer_norm=True, time_step=5):
        super(Conv_Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.time_step = time_step

    def __call__(self, obs, unit_locations, action, mask, t, n_hidden=64, reuse=False, unit_data=False, mask_loss=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # x [ batch_size*time_step, DATA_NUM]
            x = obs
            u = unit_locations

            u_shape = u.get_shape().as_list()
            assert (u_shape[1] == self.time_step)

            x = conv2d(x, 24, "conv1", (3, 3), (2, 2))  # 4 map
            u = conv3d(u, 8, "u_conv1", (1, 3, 3), (1, 2, 2))  # 1 map

            u_list = tf.split(u, self.time_step, axis=1)
            u_list = [tf.squeeze(unit, [1]) for unit in u_list]
            # stack in the time_step axis
            # TODO extract the weight for unit location
            ux = tf.stack([tf.concat([x, unit], -1) for unit in u_list], axis=1)

            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            # [batch_size, myself_num, ms/2, ms/2, ???]
            ux = conv3d(ux, 32, "conv2", (1, 3, 3), (1, 2, 2))

            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            # 10 -> 5.
            ux = conv3d(ux, 32, "conv3", (1, 3, 3), (1, 2, 2))

            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            #TODO add one (1,1,1) layer to reorganize the features.
            ux_shape = ux.get_shape().as_list()
            ux = tf.reshape(ux, (-1, self.time_step, int(np.prod(ux_shape[2:]))))

            ux = tf.layers.dense(ux, n_hidden)
            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux) # no need to extend one dimension
            ux = tf.concat([ux, action], axis=-1)

            lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

            ux, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, ux,
                                                   dtype=tf.float32)
            ux = tf.concat(ux, 2)
            # TODO v2 turn on the batch_norm after lstm
            if self.layer_norm:
                 ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            ux = tf.layers.dense(ux, 256, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            #ux = tf.nn.relu(ux)
            ux = tf.nn.dropout(ux, keep_prob=0.5)
            ux = tf.reshape(ux, [-1, self.time_step * 256, 1])
            # Q value of each
            q = tf.layers.conv1d(ux, 1, kernel_size=256, strides=256,
                                 kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            q = tf.squeeze(q, [-1])

            #p = tf.layers.conv1d(ux, 1, kernel_size=256, strides=256,
            #                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            #p = tf.squeeze(p, [-1])
            #if self.layer_norm:
            #    p = tc.layers.layer_norm(p, center=True, scale=True)
            #p = tf.nn.relu(p)

            #p = tf.layers.dense(p, self.time_step, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            #p = tf.nn.softmax(tf.divide(p, t))
            # TODO add the temperature
            # TODO  v1 punish those probabilty to be zero
            # TODO v2 punish the Q values to be zero
            """ kill the gradient using the mask """
            #print(p.get_shape().as_list(), q.get_shape().as_list(), "pq")
            #pQ = tf.multiply(p,q)
            Q_mask = tf.multiply(q, mask)
            #print(mask.get_shape().as_list(), pQ_mask.get_shape().as_list(), "mask")
            Q = tf.reduce_sum(Q_mask, axis=1, keep_dims=True)
        if unit_data:
            return Q,q
        if mask_loss:
            #TODO check this putput
            qm = tf.multiply(1-mask, q)
            return Q,qm
            #print(Q.get_shape().as_list, "Q")
        return Q

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars

class Dynamic_Conv_Actor(Model):
    def __init__(self, nb_unit_actions, name='actor', layer_norm=True, time_step=5):
        super(Dynamic_Conv_Actor, self).__init__(name=name)
        self.nb_unit_actions = nb_unit_actions
        self.layer_norm = layer_norm
        self.time_step = time_step

    # obs is a set of vector.
    # Concatenate the unit location and the feature map after the first convolutional layer.
    #TODO no graident from the critic would result in no gradient in the actor too ???
    # Are there something correlated.

    # the input is input
    # mask can be not stored.
    # then I search through the alive units.

    # the information is also stored in the observation_shape, dtype.

    # the differences between static ones and dynamic ones.
    # static [1, 0, 0, 0, 1] dynamic [1, 1, 0, 0, 0]
    # different arrangements of alive units.



    # not use mask temproally
    def __call__(self, s, ul, mask, au, n_hidden=256, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # TODO USE dynamic data of different lengths ( but need to change the experience replay )

            # split into batch_size
            # TODO Dynamic data need to be splited. Currently, just use reshape can work!
            # self.split = tf.placeholder(tf.int32, [None])

            # embedding
            x = s
            # [batch_size, myself_num, ms, ms, 1]
            u = ul
            u_shape = u.get_shape().as_list()
            # tf.shape maybe also ok
            assert (u_shape[1] == self.time_step)
            # TODO try different initializer.
            # the hyper-parameters need to be adjusted

            # 40 -> 20
            x = conv2d(x, 24, "conv1", (3, 3), (2, 2)) # 4 map
            u = conv3d(u, 8, "u_conv1", (1, 3, 3), (1, 2, 2)) # 1 map

            # u [batch_size*myself_num, ms/2, ms/2, 1] -> [batch_size, myself, ms/2, ms/2, 1]
            # x [batch_size, ms/2, ms/2, c]
            u_list = tf.split(u, self.time_step, axis=1)
            u_list = [tf.squeeze(unit, [1]) for unit in u_list]
            ux = tf.stack([tf.concat([x, unit], -1) for unit in u_list], axis=1)

            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)
            # 20 -> 10
            ux = conv3d(ux, 32, "conv2", (1, 3, 3), (1, 2, 2))

            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            # 10 -> 5.
            ux = conv3d(ux, 32, "conv3", (1, 3, 3), (1, 2, 2))

            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            #TODO add one (1,1,1) layer to reorganize the features.
            ux_shape = ux.get_shape().as_list()
            ux = tf.reshape(ux, (-1, self.time_step, int(np.prod(ux_shape[2:]))))

            # TODO is it necessary to add one layer here???
            # ux = tf.layers.dense(ux, n_hidden)
            #
            # if self.layer_norm:
            #     ux = tc.layers.layer_norm(ux, center=True, scale=True)
            # ux = tf.nn.relu(ux) # no need to extend one dimension

            # build bidirection lstm
            lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

            # TODO use output_states for process like policy iteration ---- WOW exciting ideas.
            ux, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, ux,
                                                                     dtype=tf.float32,
                                                                     sequence_length=au)
            ux = tf.concat(ux, 2)

            # TODO v2 turn on the batch_norm after lstm
            if self.layer_norm:
                 ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            ux = tf.layers.dense(ux, 256, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)
            # number of convolution kernel. --> num_actions. default (-1, 1)
            # convert to [batch_size, time_step*n_hidden], channels_last
            ux = tf.reshape(ux, [-1, self.time_step*256, 1])
            ux = tf.layers.conv1d(ux, self.nb_unit_actions, kernel_size = 256, strides = 256,
                                 kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            # [ batch_size, time_step, nb_actions]
            ux = tf.nn.tanh(ux)
        return ux

class Dynamic_Conv_Critic(Model):
    def __init__(self, name='critic', layer_norm=True, time_step=5):
        super(Dynamic_Conv_Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.time_step = time_step

    # the parameter's location has been changed
    def __call__(self, s, ul, mask, au, action, n_hidden=64, reuse=False, unit_data=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # x [ batch_size*time_step, DATA_NUM]
            x = s
            u = ul

            u_shape = u.get_shape().as_list()
            assert (u_shape[1] == self.time_step)

            x = conv2d(x, 24, "conv1", (3, 3), (2, 2))  # 4 map
            u = conv3d(u, 8, "u_conv1", (1, 3, 3), (1, 2, 2))  # 1 map

            u_list = tf.split(u, self.time_step, axis=1)
            u_list = [tf.squeeze(unit, [1]) for unit in u_list]
            # stack in the time_step axis
            # TODO extract the weight for unit location
            ux = tf.stack([tf.concat([x, unit], -1) for unit in u_list], axis=1)

            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            # [batch_size, myself_num, ms/2, ms/2, ???]
            ux = conv3d(ux, 32, "conv2", (1, 3, 3), (1, 2, 2))

            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            # 10 -> 5.
            ux = conv3d(ux, 32, "conv3", (1, 3, 3), (1, 2, 2))

            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            #TODO add one (1,1,1) layer to reorganize the features.
            ux_shape = ux.get_shape().as_list()
            ux = tf.reshape(ux, (-1, self.time_step, int(np.prod(ux_shape[2:]))))

            ux = tf.layers.dense(ux, n_hidden)
            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)
            # Maybe I need an embedding.
            ux = tf.concat([ux, action], axis=-1)

            lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

            ux, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, ux,
                                                   dtype=tf.float32,
                                                   sequence_length=au)
            ux = tf.concat(ux, 2)
            # TODO v2 turn on the batch_norm after lstm
            if self.layer_norm:
                 ux = tc.layers.layer_norm(ux, center=True, scale=True)
            ux = tf.nn.relu(ux)

            ux = tf.layers.dense(ux, 256, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            if self.layer_norm:
                ux = tc.layers.layer_norm(ux, center=True, scale=True)
            #ux = tf.nn.relu(ux)
            ux = tf.nn.dropout(ux, keep_prob=0.5)
            ux = tf.reshape(ux, [-1, self.time_step * 256, 1])
            # Q value of each
            # some computations are wasted
            q = tf.layers.conv1d(ux, 1, kernel_size=256, strides=256,
                                 kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            q = tf.squeeze(q, [-1])

            """ kill the gradient using the mask """
            #print(p.get_shape().as_list(), q.get_shape().as_list(), "pq")
            #pQ = tf.multiply(p,q)
            Q_mask = tf.multiply(q, mask)
            #print(mask.get_shape().as_list(), pQ_mask.get_shape().as_list(), "mask")
            Q = tf.reduce_sum(Q_mask, axis=1, keep_dims=True)
        if unit_data:
            return Q,Q_mask
        # if mask_loss:
        #     #TODO check this putput
        #     qm = tf.multiply(1-mask, q)
        #     return Q,qm
        #     #print(Q.get_shape().as_list, "Q")
        return Q

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
