from models import MlpNetwork
from utils import *
import numpy as np
import tensorflow as tf


# in charge of
# 1. training
# 2. action chosen (according to policy)

# kargs include
# input_size, output_size, hid_size, num_hid_layers


# TODO batch_size may be necessary
class Agent(object):
    def __init__(self, sess, lr, action_dict=None, policy_name="pi", **kargs):
        self.sess = sess
        self.net = MlpNetwork(name=policy_name, **kargs)
        self.action_dict = action_dict
        self.opt = tf.train.AdamOptimizer(learning_rate=lr,
                                beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(self.net.policy_loss)

    def action(self, ob, stochastic=True):
        # obs is only one state by default
        feed_dict = {self.net.ob: ob[None]}
        prob = self.sess.run(self.net.p, feed_dict=feed_dict)
        # the batch_size is one
        prob = prob[0]

        if not stochastic:
            action = np.argmax(prob, -1)
        else:
            action = np.random.choice(np.arange(len(prob)), p=prob)

        if self.action_dict:
            action = self.action_dict[action]
        return action

    # ep should have 1. ob 2. ret(discounted return) 3. action
    def train(self, ep):
        feed_dict = {self.net.ob: ep["ob"],
                     self.net.r: ep["ret"],
                     self.net.action:ep["action"]}

        self.sess.run(self.opt, feed_dict=feed_dict)

