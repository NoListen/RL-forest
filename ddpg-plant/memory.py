# https://github.com/openai/baselines/baselines/ddpg/memory.py

# I want to use the memory used in mxdqn.

import numpy as np

# TODO use the same storage space for both the obs0 and obs1. USE HALF Memory.
# TODO use a dict to save all observation. (Key to key storation)
""" I DON'T NEED TO CHANGE THE API AGAIN
DICT HAS key:shape --> create the space
DICT ALSO HAS key:data --> append the data

Engineering Problem
"""


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype=np.float32):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape, dtype=dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)

# TODO integrate attacking map into unit_location map
# TODO reuse the memory.(obs0 obs1 too many redundance)

# 1. alive units - num
# 2. The storage still uses fixed size.
# 3. mask useless.
# 4. pass a dictionary to the Memory.

class ObservationBuffer(object):
    def __init__(self, limit, observation_shape, observation_dtype):
        assert (observation_dtype.keys() == observation_dtype.keys())
        self.d = {}
        self.k_set = observation_dtype.keys()
        for k in observation_dtype.keys():
            self.d[k] = RingBuffer(limit, shape=observation_shape[k], dtype=observation_dtype[k])


    def get_batch(self, batch_idx):
        b = {}
        for k in self.k_set:
            b[k] = array_min2d(self.d[k].get_batch(batch_idx))
        return b

    def append(self, v):
        assert(v.keys() == self.k_set)
        for k in self.k_set:
            self.d[k].append(v[k])


    def __len__(self):
        # pass the length to the upper level
        return len(self.d[self.k_set[0]])

class CompoundMemory(object):
    def __init__(self, limit, action_shape, observation_shape, observation_dtype):
        self.limit = limit

        # TODO change unit_location and mask as boolean type
        assert(type(observation_shape) == type({}))
        assert(type(observation_dtype) == type({}))
        self.observations0 = ObservationBuffer(limit, observation_shape, observation_dtype)
        self.observations1 = ObservationBuffer(limit, observation_shape, observation_dtype)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.length = 0



    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)
        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': obs0_batch,
            'obs1': obs1_batch,
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch)
        }

        return result

    # when it is changed. I think training will be changed.
    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return
        self.length = min(self.limit, self.length+1)
        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.observations0)

