# https://github.com/openai/baselines/baselines/ddpg/ddpg.py

from copy import copy
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tc
from util import reduce_std

def get_target_updates(vars, target_vars, tau):
    print('setting up target updates ...')
    soft_updates = []
    init_updates = []
    for i in vars:
        print(i.name)
    for i in target_vars:
        print(i.name)
    print(len(vars), len(target_vars))
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)

class DDPG(object):
    def __init__(self, actor, critic, memory, observation_shape, action_shape, action_noise=None,
                 gamma=0.99, tau=0.001, batch_size=128, action_range=(-1., 1.), critic_l2_reg=0,
                 actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.):
        # train input
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')

        # critic_target is used to update the Q value of critic
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')

        # Parameters.
        self.gamma = gamma # discount factor.
        self.tau = tau  # soft update
        self.memory = memory
        self.action_noise = action_noise  # bool
        # I want A zip( the function ) vector !
        # I'll see it later.
        self.action_range = action_range  # action range. ( data type. vector range ? )
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr # apply to optimizer.
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm  # update the parameter
        self.reward_scale = reward_scale  # different from returns normalization.
        self.batch_size = batch_size  # How to balance the data imbalance.
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg  # regularization. I think the probable form is weight decay.

        target_actor = copy(actor)
        target_actor.name = 'target_actor' # build the network only when _call is invoked
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic  # a network or an object.

        # Create networks and core TF parts that are shared across setup parts.
        # When the network is too wide or deep, it tends to be overfitting.
        self.actor_tf = actor(self.obs0)
        self.critic_tf = critic(self.obs0, self.actions)
        self.critic_with_actor_tf = critic(self.obs0, self.actor_tf, reuse=True)

        Q_obs1 = target_critic(self.obs1, target_actor(self.obs1))
        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        # Set up parts.
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        #  used to debug and follow the process.
        self.setup_stats()
        self.setup_target_network_updates()

    def setup_target_network_updates(self):
        # init & soft
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars,
                                                                      self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_actor_optimizer(self):
        print("setting up actor optimizer")
        # because of reuse, no extra computation.
        # gradient ascent. use Q's direction to update. Thus. the gradient need to be small.
        # Maybe I should use advantage function.
        # How to update advantage function. ( update Q and V simultaneously )
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        print('  actor shapes: {}'.format(actor_shapes))
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr,
                                                      beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(self.actor_loss)
        
	# different to flat gradient ?
        #actor_grads = self.actor_optimizer.compute_gradients(self.actor_loss, var_list=self.actor.trainable_vars)
        #if self.clip_norm:
        #    self.actor_grads = tf.clip_by_global_norm(actor_grads, self.clip_norm)
        #else:
        #    self.actor_grads = actor_grads

    # the update of value is almost independent of updating policy.
    def setup_critic_optimizer(self):
        print('setting up critic optimizer')
        # Most cases ( train ), off policy
        self.critic_loss = tf.reduce_mean(tf.square(self.critic_tf - self.critic_target))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if
                               'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                print('  regularizing: {}'.format(var.name))
            print('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        print('  critic shapes: {}'.format(critic_shapes))
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr,
                                                       beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.critic_loss)
        #critic_grads = self.critic_optimizer.compute_gradients(self.critic_loss, self.critic.trainable_vars)
        #if self.clip_norm:
        #    self.critic_grads = tf.clip_by_global_norm(critic_grads, self.clip_norm)
        #else:
        #    self.critic_grads = critic_grads


    def setup_stats(self):
        ops = []
        names = []

        # if self.normalize_returns:
        #     ops += [self.ret_rms.mean, self.ret_rms.std]
        #     names += ['ret_rms_mean', 'ret_rms_std']

        # if self.normalize_observations:
        #     ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
        #     names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def pi(self, obs, apply_noise=True, compute_Q=True):

        actor_tf = self.actor_tf
        feed_dict = {self.obs0: [obs]}
        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None
        action = action.flatten()
        # I don't think the noise is adjusting to the action.
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
            # noise is a value in each dimension
        #  Maybe I need a additional specification in this situation.
        # Continuous Space
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale
        self.memory.append(obs0, action, reward, obs1, terminal1)
        # moving average.
        # if self.normalize_observations:
        #     self.obs_rms.update(np.array([obs0]))

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        target_Q = self.sess.run(self.target_Q, feed_dict={
            self.obs1: batch['obs1'],
            self.rewards: batch['rewards'],
            self.terminals1: batch['terminals1'].astype('float32'),
        })

        # Get all gradients and perform a synced update.
        ops = [self.actor_optimizer, self.actor_loss, self.critic_optimizer, self.critic_loss]
        _, actor_loss, _, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target: target_Q,
        })
        # self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        # self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        return critic_loss, actor_loss

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0: self.stats_sample['obs0'],
            self.actions: self.stats_sample['actions'],
        })

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        # if self.param_noise is not None:
        #     stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()


