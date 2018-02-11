# reinforce is one of the most simple PG algorithm
from agent import Agent
from utils import traj_segment_generator, add_vtarg_and_adv
from collections import deque
import numpy as np
import tensorflow as tf

def ac(env, sess, obs_processor, lr, gamma, lam, action_dict, horizon, **kargs):
    pi = Agent(sess, lr, action_dict, **kargs)
    sess.run(tf.global_variables_initializer())
    # sample according to horizon not each episode.
    seg_gen = traj_segment_generator(pi, env, obs_processor, horizon, stochastic=True)

    ep_rets = deque(maxlen=100)
    ep_steps = deque(maxlen=100)
    episodes = 0

    # TODO set terminal state.
    while True:
        # generate one episode
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        # output the recent one.
        if len(seg["ep_ret"]) > 0:
            ep_rets.extend(seg["ep_ret"])
            ep_steps.extend(seg["ep_steps"])
            for i in range(len(seg["ep_ret"])):
                print("ep%i ret:%.2f steps:%i average_ret:%.1f average_steps:%.1f" %
                    (episodes+i+1, ep["ep_ret"][i], ep["ep_steps"][i], np.mean(ep_rets), np.mean(ep_steps)))
            episodes += len(seg["ep_ret"])

        pi.train(seg)


