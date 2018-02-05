from skimage.transform import resize
import numpy as np
import scipy.signal

# only support 2D's operation.
class ObsProcessor(object):
    def __init__(self, crop_area = None, resize_shape = None, flatten=True):
        self.crop_area = crop_area #(x1, y1, x2, y2)
        self.resize_shape = resize_shape
        self.flatten = flatten

    # (y, x)
    def process(self, obs):
        if self.crop_area:
            obs = obs[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2]]
        if self.flatten:
            obs = resize(obs, self.resize_shape)
        if self.flatten:
            obs = obs.astype(np.float).ravel()
        return obs

def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.

    inputs
    ------
    x: ndarray
    gamma: float

    outputs
    -------
    y: ndarray with same shape as x, satisfying

        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]


# Maybe train batch by batch may work
# obs_processor: ObsProcessor
# TODO multi-threading data genration
def traj_segment_generator(pi, env, obs_processor, stochastic=True):
    ob = env.reset()

    obs = []
    actions = []
    rews = []

    ep_ret = 0
    ep_steps = 0
    while True:
        ob = obs_processor(ob)
        action = pi.action(ob, stochastic)
        obs.append(ob)
        actions.append(action)

        ob, rew, done = env.step(action)

        rews.append(rew)

        ep_ret += rew
        ep_steps += 1

        if done:
            yield {"ob": np.array(obs), "action": np.array(actions),
            "rew": np.array(rews), 'ep_ret': ep_ret, 'ep_steps': ep_steps}
            obs = []
            actions = []
            rews = []
            ep_ret = 0
            ep_steps = 0
            ob = env.reset()
