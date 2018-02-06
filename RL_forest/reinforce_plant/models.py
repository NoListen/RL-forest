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

    def _init(self, input_size, num_output, hid_size, num_hid_layers):
        self.ob = tf.placeholder(tf.float32, shape=(None,) + input_size, name='obs')

        last_out = self.ob
        for i in range(num_hid_layers):
            last_out = tf.nn.relu(tf.layers.dense(last_out, hid_size))

        # probabilities of all actions
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
        self.value_loss = tf.nn.l2_loss(self.r - self.v)
        self.total_loss = self.policy_loss + self.value_loss

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
