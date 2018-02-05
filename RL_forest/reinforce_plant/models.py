# TODO implement RNN algorithm
import tensorflow as tf

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
class MlpNetwork(object):
    def __init__(self, name, *args, **kargs):
        with tf.variable_scope(name):
            self._init(*args, **kargs)
            self.scope = tf.get_variable_scope().name
            self.cls = 'mlp'

    def _init(self, input_size, output_size, hid_size, num_hid_layers):
        self.ob = tf.placeholder(tf.float32, shape=(None, input_size), name='obs')

        last_out = self.ob
        for i in range(num_hid_layers):
            last_out = tf.nn.relu(tf.layers.dense(last_out, hid_size))

        last_out = tf.layers.dense(last_out, output_size)
        # probabilities of all actions
        self.p = tf.nn.softmax(last_out)

        #  preparing loss
        self.action = tf.placeholder(tf.int32, shape=(None,), name="action")
        # use Monte-Carlo Return at first.
        self.r = tf.placeholder(tf.float32, shape=(None,), name="return")

        onehot_action = tf.one_hot(self.action, output_size, 1.0, 0.0, name="action_one_hot") # output_size = num_actions
        pa = tf.reduce_sum(tf.multiply(self.p * onehot_action), axis=1)
        # log is the key idea in policy graident
        logpa = tf.log(tf.clip(pa, 1e-20, 1.0)) # avoid extreme situation

        # maximize r*logpa
        self.policy_loss = -tf.reduce_sum(self.r * logpa)

        # TODO add entropy
        # igore entropy temporally.
        # Note: entropy loss penalizes the exploration.
        # As long as the entropy of action is small, it can be not reverted.
        # the action with less prob may be not easily exploited to find higher values.

        # TODO add Critic Structure

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
