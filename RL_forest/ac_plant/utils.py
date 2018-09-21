from skimage.transform import resize
import numpy as np
import random


# resize_shape should include the channels.
# if not, it'll be a common 2D image
class ObsProcessor(object):
    def __init__(self, obs_shape, crop_area = None, resize_shape = None, flatten=True):
        self.crop_area = crop_area #(x1, y1, x2, y2)
        self.resize_shape = resize_shape
        self.flatten = flatten

        # resize rescales 0-255 to 0-1
        # if you don't want to rescale, use cv2.resize(img, shape, inter_nearest)
        if resize_shape:
            shape = resize_shape
        elif crop_area:
            shape = (crop_area[3]-crop_area[1], crop_area[2]-crop_area[0], obs_shape[-1])
        else:
            shape = obs_shape

        if flatten:
            self.out_shape = (np.prod(shape), )
        else:
            self.out_shape = shape

    # (y, x)
    def process(self, obs):
        if self.crop_area:
            obs = obs[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2]]
        if self.resize_shape:
            obs = resize(obs, self.resize_shape, order=0) # no interpolation. Can change this.
        if self.flatten:
            obs = obs.astype(np.float).ravel()
        return obs



def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    # I think is has been simplified
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


# Maybe train batch by batch may work
# obs_processor: ObsProcessor
# TODO multi-threading data genration
def traj_segment_generator(pi, env, obs_processor, horizon=64, stochastic=True):
    t = 0
    new = True
    ob = env.reset()
    ob = obs_processor.process(ob)

    obs = np.array([ob for _ in range(horizon)])
    # discrete simple action
    acs = np.zeros(horizon, "int32")
    news = np.zeros(horizon, "int32")
    vpreds = np.zeros(horizon, "float32")
    rews = np.zeros(horizon, "float32")

    ep_rets = []
    ep_lens = []
    cur_ep_ret = 0
    cur_ep_len = 0

    while True:
        ac, vpred = pi.action(ob, stochastic)

        if t % horizon == 0 and t > 0:
            yield{"ob": obs, "ac": acs, "rew": rews, "vpred": vpreds, "new": news,
                  "nextvpred": vpred * (1-new), "ep_rets": ep_rets, "ep_lens": ep_lens}
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        # pong's action need to be a tuple
        ob, rew, new,  _ = env.step([ac, None])
        ob = obs_processor.process(ob)
        rews[i] = rew[0] #( the left side reward)

        cur_ep_ret += rew[0]
        cur_ep_len += 1

        t += 1

        if new:
            ep_lens.append(cur_ep_len)
            ep_rets.append(cur_ep_ret)
            cur_ep_ret = 0
            cur_ep_len = 0

            ob = env.reset()
            ob = obs_processor.process(ob)


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)
