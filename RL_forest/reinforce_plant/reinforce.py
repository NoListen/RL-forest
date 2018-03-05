# reinforce is one of the most simple PG algorithm
from agent import Agent
from utils import traj_segment_generator, discount, saveToFlat, loadFromFlat
from collections import deque
import numpy as np
import tensorflow as tf
import os

def reinforce(env, sess, obs_processor, lr, gamma, action_dict, model_dir, model_path, phase, **kargs):



    pi = Agent(sess, lr, action_dict, **kargs)
    sess.run(tf.global_variables_initializer())
    ep_gen = traj_segment_generator(pi, env, obs_processor, stochastic=True)

    ep_rets = deque(maxlen=100)
    ep_steps = deque(maxlen=100)
    episodes = 0

    var_list = pi.net.get_variables()
    # train or not
    train = (phase == "train")

    if model_path is not None:
        print("loading model" + model_path)
        loadFromFlat(var_list, model_path)

    # TODO set terminal state.
    while True:
        # generate one episode
        ep = ep_gen.__next__()
        ep_rets.append(ep["ep_ret"])
        ep_steps.append(ep["ep_steps"])
        episodes += 1

        if train:
            ep["ret"] = discount(ep["rew"], gamma)
            ep["ret"] -= np.mean(ep["ret"])
            ep["ret"] /= np.std(ep["ret"])
            pi.train(ep)

        print("ep%i ret:%.2f steps:%i average_ret:%.1f average_steps:%.1f" %
              (episodes, ep["ep_ret"], ep["ep_steps"], np.mean(ep_rets), np.mean(ep_steps)))\

        if episodes % 100 == 0:
            saveToFlat(var_list, model_dir + "/ep%i.pickle"%episodes)