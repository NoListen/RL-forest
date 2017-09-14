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

# the observation is consistent among the "1. memory 2. model 3. env"
def build_obs_placeholder(observation_shape, observation_dtype, name):
    d = {}
    print("building all the observation", list([observation_shape.keys()]))
    for k in observation_shape.keys():
        obs_shape = (None,) + observation_shape[k]
        obs_name = name + "_" + k
        obs_dtype = observation_dtype[k]
        # no computation supports uint8
        if obs_dtype == "uint8":
            obs_dtype = "float32"
        print(obs_name+" has shape ", obs_shape)
        d[k] = tf.placeholder(obs_dtype, obs_shape, name=obs_name)
    return d

# PAY ATTENTION to observation_shape
class Dynamic_DDPG(object):
    def __init__(self, actor, critic, memory, observation_shape, observation_dtype, action_shape, action_noise=None,
                 gamma=0.99, tau=0.001, batch_size=128, action_range=(-1., 1.), critic_l2_reg=0,
                 actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1., n_hidden=64, reward_shape=(1,)):
        """n_hidden is for lstm unit"""
        assert memory.cls=="compound", "compound"

        # self.t_value = 10
        # self.anneal_delta = 1

        # train input
        self.obs0 = build_obs_placeholder(observation_shape, observation_dtype, "obs0")
        self.obs1 = build_obs_placeholder(observation_shape, observation_dtype, 'obs1')
        self.t = tf.placeholder(tf.float32, None, name="temperature")

        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')

        self.rewards = tf.placeholder(tf.float32, shape=(None, )+reward_shape, name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')

        # critic_target is used to update the Q value of critic
        self.critic_target = tf.placeholder(tf.float32, shape=(None,)+reward_shape, name='critic_target')

        # Parameters.
        self.gamma = gamma # discount factor.
        # tau is also an important hyper-parameter
        self.tau = tau  # soft update
        self.memory = memory
        self.action_noise = action_noise  # bool
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
        self.actor_tf = actor(n_hidden=n_hidden, **self.obs0)
        print(self.actor_tf.name,"EMMMMMMMMMMMMMMMMMMMM SHAPE HERE")
        # change the __call__ parameters to be "map" rather than the "observation"
        # map the dictionary to those variables
        self.critic_tf, self.uq = critic(action=self.actions, n_hidden=n_hidden, unit_data=True, **self.obs0)
        self.critic_with_actor_tf, self.uq_with_actor = critic(action=self.actor_tf, n_hidden=n_hidden, reuse=True, unit_data=True,
                                             **self.obs0)

        # well, it's combined from several units.
        Q_obs1, uq_obs1 = target_critic(action=target_actor(n_hidden=n_hidden, **self.obs1), n_hidden=n_hidden, unit_data=True,
                               **self.obs1)
        self.target_uq = self.rewards + (1. - self.terminals1) * gamma * uq_obs1

        # Set up parts.
        self.setup_critic_optimizer()
        self.setup_actor_optimizer()

        #  used to debug and follow the process.
        self.setup_stats()
        self.setup_target_network_updates()

    # No probability used temporally
    # def anneal_t(self):
    #     self.t_value = max(1, self.t_value - self.anneal_delta)

    def setup_target_network_updates(self):
        # init & soft
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)

        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars,
                                                                      self.tau)

        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_actor_optimizer(self):
        print("setting up actor optimizer")
        # Maybe I should use advantage function.
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        print('  actor shapes: {}'.format(actor_shapes))
        with tf.variable_scope('actor_optimizer'):
            self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr,
                                                     beta1=0.9, beta2=0.999,epsilon=1e-8)
            # self.actor_optimizer = tf.train.RMSPropOptimizer(self.actor_lr, momentum=0.95, epsilon=0.01)
            # if apply gradients to specific variables, it will also update the critic network.
            actor_grads = tf.gradients(self.actor_loss, self.actor.trainable_vars)
            if self.clip_norm:
                self.actor_grads,_ = tf.clip_by_global_norm(actor_grads, self.clip_norm)
            else:
                self.actor_grads = actor_grads
            grads_and_vars = list(zip(self.actor_grads, self.actor.trainable_vars))
            self.actor_train_op = self.actor_optimizer.apply_gradients(grads_and_vars)
            

    # the update of value is almost independent of updating policy.
    def setup_critic_optimizer(self):
        print('setting up critic optimizer')
        # Most cases ( train ), off policy
        #self.critic_loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(self.mask, axis=1, keep_dims=True),tf.square(self.critic_tf - self.critic_target))+\
        #    tf.reduce_sum(tf.square(self.critic_qm),axis=1, keep_dims=True))
        loss1 = tf.reduce_sum(tf.square(self.uq - self.critic_target), axis=1, keep_dims = True)
        # loss2 = tf.reduce_mean(tf.square(self.critic_qm),axis=1, keep_dims=True)
        self.critic_loss = tf.reduce_mean(loss1)

        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if
                               ('kernel' in var.name or 'W' in var.name) and 'output' not in var.name]
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
        with tf.variable_scope('critic_optimizer'):
            # Adam may be better. RMSProp can result in Q value explosion.
            # self.critic_optimizer = tf.train.RMSPropOptimizer(self.critic_lr, momentum=0.95, epsilon=0.01)
            self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr,
                                                      beta1=0.9, beta2=0.999, epsilon=1e-8)
            critic_grads = tf.gradients(self.critic_loss, self.critic.trainable_vars)
            if self.clip_norm:
                self.critic_grads,_ = tf.clip_by_global_norm(critic_grads, self.clip_norm)
            else:
                self.critic_grads = critic_grads
            grads_and_vars = list(zip(self.critic_grads, self.critic.trainable_vars))
            self.critic_train_op = self.critic_optimizer.apply_gradients(grads_and_vars)

    def setup_stats(self):
        ops = []
        names = []

        # if self.normalize_returns:
        #     ops += [self.ret_rms.mean, self.ret_rms.std]
        #     names += ['ret_rms_mean', 'ret_rms_std']

        # if self.normalize_observations:
        #     ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
        #     names += ['obs_rms_mean', 'obs_rms_std']

        # TODO add all values of all Q values.
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
        """ Obs is composed of [[myself_obs, mask], [enemy_obs, mask], map] """
        # NOW i want to know the network's memorization ability. MODEL the enemy.
        # dead one will be ignored by my environment.
        assert obs.keys() == self.obs0.keys(), "the observation is not valid. Can't be used to compute the policy"

        actor_tf = self.actor_tf
        # TODO utilize the map and enenmy's information in theb network computation.
        # add [] to expand the dim
        feed_dict = {self.obs0[k]: [obs[k]] for k in obs.keys()}
        if compute_Q:
            action, q, uq = self.sess.run([actor_tf, self.critic_with_actor_tf, self.uq_with_actor], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None
            uq = None
        action = np.squeeze(action, [0])
        # I don't think the noise is adjusting to the action.
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
            # noise is a value in each dimension
        #  Maybe I need a additional specification in this situation.
        # Continuous Space
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q, uq

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale
        # TODO store the enemy's information and the map
        self.memory.append(obs0, action, reward, obs1, terminal1)
        # moving average.
        # if self.normalize_observations:
        #     self.obs_rms.update(np.array([obs0]))

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        target_feed_dict = {self.obs1[k]: batch["obs1"][k] for k in self.obs1.keys()}
        target_feed_dict.update({
            self.rewards: batch["rewards"],
            self.terminals1: batch["terminals1"].astype('float32')
        })
        target_uq = self.sess.run(self.target_uq, feed_dict=target_feed_dict)

        # Get all gradients and perform a synced update.
        ops = [self.actor_train_op, self.actor_loss, self.critic_train_op, self.critic_loss]

        feed_dict = {self.obs0[k]: batch["obs0"][k] for k in self.obs0.keys()}
        feed_dict.update({
            self.actions: batch["actions"],
            self.critic_target: target_uq
        })

        _, actor_loss, _, critic_loss = self.sess.run(ops, feed_dict=feed_dict)
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
        feed_dict = {self.obs0[k]: self.stats_sample["obs0"][k] for k in self.obs0.keys()}
        feed_dict.update({
            self.actions:self.stats_sample["actions"]
        })
        values = self.sess.run(self.stats_ops, feed_dict=feed_dict)

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


