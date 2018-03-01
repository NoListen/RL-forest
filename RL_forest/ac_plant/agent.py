from models import ConvNetwork
from utils import *
import numpy as np
import tensorflow as tf


# in charge of
# 1. training
# 2. action chosen (according to policy)

# kargs include
# input_size, output_size, hid_size, num_hid_layers

# then you'll know why ppo is good at these tasks

class Agent(object):
    def __init__(self, sess, lr, action_dict=None, policy_name="pi", **kargs):
        self.sess = sess
        self.net = ConvNetwork(name=policy_name, **kargs)
        self.action_dict = action_dict
        self.opt = tf.train.AdamOptimizer(learning_rate=lr,
                                beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(self.net.policy_loss)

    def action(self, ob, stochastic=True):
        # obs is only one state by default
        feed_dict = {self.net.ob: ob[None]}
        prob, value = self.sess.run([self.net.p, self.net.v], feed_dict=feed_dict)
        # the batch_size is one
        prob = prob[0]
        value = value[0]

        # print("the action prob & value " , prob, value)

        if not stochastic:
            action = np.argmax(prob, -1)
        else:
            action = np.random.choice(np.arange(len(prob)), p=prob)

        if self.action_dict:
            action = self.action_dict[action]
        return action, value

    def evaluate(self, ob):
        feed_dict = {self.net.ob: ob[None]}
        value = self.sess.run(self.net.v, feed_dict=feed_dict)
        value = value[0]
        return value

    # ep should have 1. ob 2. ret(discounted return) 3. action
    def train(self, seg):
        feed_dict = {self.net.ob: seg["ob"],
                     self.net.r: seg["tdlamret"],
                     self.net.action:seg["ac"],
                     self.net.td: seg["adv"]}

        self.sess.run(self.opt, feed_dict=feed_dict)

