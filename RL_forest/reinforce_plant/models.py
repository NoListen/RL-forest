# TODO implement RNN algorithm
import tensorflow as tf
import numpy as np
# network does no relationship with choosing action policy

"""
input:
* for all
- obs:          observation placeholder
* for train
- action:       action chosen
- r:            MonteCarlo Return

output:
- p:            policy to take
- policy_loss:  policy gradient loss
          
"""

def get_w_bound(filter_shape):
    # return np.sqrt(6./(np.prod(filter_shape[:-2]))*np.sum(filter_shape[-2:]))
    return np.sqrt(6./((np.prod(filter_shape[:-2]))*np.sum(filter_shape[-2:])))


# modified from https://github.com/openai/universe-starter-agent/model.py
# in NHWC manner
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

class MlpNetwork(object):
    def __init__(self, name, *args, **kargs):
        with tf.variable_scope(name):
            self._init(*args, **kargs)
            self.scope = tf.get_variable_scope().name
            self.cls = 'mlp'

    def _init(self, input_size, num_output, hid_size, num_hid_layers):
        assert len(input_size) == 1
        self.ob = tf.placeholder(tf.float32, shape=(None,) + input_size, name='obs')

        last_out = self.ob
        for i in range(num_hid_layers):
            last_out = tf.nn.relu(tf.layers.dense(last_out, hid_size))

        # probabilities of all actions
        self.p = tf.nn.softmax(tf.layers.dense(last_out, num_output))

        # the value function
        self.v = tf.layers.dense(last_out, 1)

        self.action = tf.placeholder(tf.int32, shape=(None,), name="action")
        # use Monte-Carlo Return at first.
        self.r = tf.placeholder(tf.float32, shape=(None,), name="return")

        onehot_action = tf.one_hot(self.action, num_output, 1.0, 0.0, name="action_one_hot") # output_size = num_actions
        # log is the key idea in policy graident
        logp = tf.log(tf.clip_by_value(self.p, 1e-20, 1.0)) # avoid extreme situation
        entropy = -tf.reduce_sum(self.p * logp, axis=1)

        logpa = tf.reduce_sum(logp*onehot_action, axis=1)
        self.policy_loss = -tf.reduce_sum(self.r * logpa + entropy*0.01)

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class ConvNetwork(object):
    def __init__(self, name, *args, **kargs):
        with tf.variable_scope(name):
            self._init(*args, **kargs)
            self.scope = tf.get_variable_scope().name
            self.cls = 'conv'

    def _init(self, input_size, num_output):
        self.ob = tf.placeholder(tf.float32, shape=(None,) + input_size, name='obs')
        last_out = self.ob

        last_out = tf.nn.relu(conv2d(last_out, 16, "conv1", (8, 8), (4, 4), pad="VALID"))
        print(last_out.shape)
        last_out = tf.nn.relu(conv2d(last_out, 32, "conv2", (4, 4), (2, 2), pad="VALID"))
        print(last_out.shape)
        num_feat = np.prod(last_out.shape[1:])
        last_out = tf.reshape(last_out, [-1, num_feat])

        last_out = tf.nn.relu(tf.layers.dense(last_out, 256))
                                              # kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)))

        self.p = tf.nn.softmax(tf.layers.dense(last_out, num_output))

        self.action = tf.placeholder(tf.int32, shape=(None,), name="action")
        # Monte-Carlo Return
        self.r = tf.placeholder(tf.float32, shape=(None,), name="return")

        onehot_action = tf.one_hot(self.action, num_output, 1.0, 0.0, name="action_one_hot") # output_size = num_actions

        logp = tf.log(tf.clip_by_value(self.p, 1e-20, 1.0)) # avoid extreme situation
        logpa = tf.reduce_sum(tf.multiply(logp, onehot_action), axis=1)
        entropy = - tf.reduce_sum(logp * self.p, axis=1)

        # maximize r*logpa
        self.policy_loss = -tf.reduce_sum(self.r * logpa + 0.01*entropy)

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
