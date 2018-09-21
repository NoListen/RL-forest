# TODO implement RNN algorithm
import tensorflow as tf
import numpy as np


def get_w_bound(filter_shape):
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


def max_pool(x, filter_size=(2,2), stride=(2, 2)):
    stride_shape = [1, stride[0], stride[1], 1]
    filter_shape = [1, filter_size[0], filter_size[1], 1]
    return tf.nn.max_pool(x, ksize = filter_shape, strides = stride_shape, padding = "SAME")

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

# Assume the iput size is 60 by 60

# conv kernel are fixed
# conv1 8 by 8 stride 4  = 15
# pool1 2 by 2          = 8
# conv2 4 by 4 stride 2 = 4
# conv3 3 by 3 stride 1 = 2

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
        num_feat = np.prod(last_out.shape[1:])
        last_out = tf.reshape(last_out, [-1, num_feat])

        last_out = tf.nn.relu(tf.layers.dense(last_out, 256))
                                              # kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)))

        self.p = tf.nn.softmax(tf.layers.dense(last_out, num_output))
        # Critic architecture
        self.v = tf.layers.dense(last_out, 1)

        self.action = tf.placeholder(tf.int32, shape=(None,), name="action")
        # Monte-Carlo Return
        self.r = tf.placeholder(tf.float32, shape=(None,), name="return")
        # TD-1 or GAE
        self.td = tf.placeholder(tf.float32, shape=(None,), name="td")

        onehot_action = tf.one_hot(self.action, num_output, 1.0, 0.0, name="action_one_hot") # output_size = num_actions

        logp = tf.log(tf.clip_by_value(self.p, 1e-20, 1.0)) # avoid extreme situation
        logpa = tf.reduce_sum(tf.multiply(logp, onehot_action), axis=1)
        entropy = - tf.reduce_sum(logp * self.p, axis=1)

        # maximize r*logpa
        self.policy_loss = -tf.reduce_sum(self.td * logpa + 0.01*entropy)
        self.value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)
        self.total_loss = self.policy_loss + self.value_loss

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
