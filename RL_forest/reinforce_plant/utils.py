from skimage.transform import resize
import numpy as np
import scipy.signal
import random
from scipy.misc import imsave
from collections import deque
import tensorflow as tf
import pickle


# var_list is returned by the policy.
# Thus, they should be the same. I assume.
def saveToFlat(var_list, param_pkl_path):
    # get all the values
    var_values = np.concatenate([v.flatten() for v in tf.get_default_session().run(var_list)])
    pickle.dump(var_values, open(param_pkl_path, "wb"))

def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params.astype(np.float32)

def loadFromFlat(var_list, param_pkl_path):
    flat_params = load_from_file(param_pkl_path)
    print("the type of the parameters stored is ", flat_params.dtype)
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        print(v.name)
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})

# TODO support IMg with RGB channels
# only support 2D's operation.
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
            obs = resize(obs, self.resize_shape) # no interpolation. Can change this.
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
        ob = obs_processor.process(ob)
        action = pi.action(ob, stochastic)
        obs.append(ob)
        actions.append(action)

        ob, rew, done, _ = env.step([action, None])

        # pong env support two players.
        rews.append(rew[0])

        ep_ret += rew[0]
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


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)
