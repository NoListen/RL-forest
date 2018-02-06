# reinforce is one of the most simple PG algorithm
from agent import Agent
from utils import traj_segment_generator, add_vtarg_and_adv
from collections import deque
import numpy as np
import tensorflow as tf

def reinforce(env, sess, obs_processor, lr, gamma, lam, action_dict, horizon, **kargs):
    pi = Agent(sess, lr, action_dict, **kargs)
    sess.run(tf.global_variables_initializer())
    seg_gen = traj_segment_generator(pi, env, obs_processor, horizon, stochastic=True)

    ep_rets = deque(maxlen=50)
    ep_steps = deque(maxlen=50)
    episodes = 0

    # TODO set terminal state.
    while True:
        # generate one episode
        seg = seg_gen.__next__()
        ep_rets.extend(seg["ep_ret"])
        ep_steps.extend(seg["ep_steps"])
        add_vtarg_and_adv(seg, gamma, lam)
        episodes += 1

        pi.train(seg)
