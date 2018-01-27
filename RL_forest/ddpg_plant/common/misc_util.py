import gym
import numpy as np
import random
import time


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)

class SimpleMonitor(gym.Wrapper):
    def __init__(self, env):
        """Adds two qunatities to info returned by every step:

            num_steps: int
                Number of steps takes so far
            rewards: [float]
                All the cumulative rewards for the episodes completed so far.
        """
        super().__init__(env)
        # current episode state
        self._current_reward = None
        self._num_steps = None
        # temporary monitor state that we do not save
        self._time_offset = None
        self._total_steps = None
        # monitor state
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_end_times = []

    def _reset(self):
        obs = self.env.reset()
        # recompute temporary state if needed
        if self._time_offset is None:
            self._time_offset = time.time()
            if len(self._episode_end_times) > 0:
                self._time_offset -= self._episode_end_times[-1]
        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)
        # update monitor state
        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._episode_end_times.append(time.time() - self._time_offset)
        # reset episode state
        self._current_reward = 0
        self._num_steps = 0

        return obs

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        info['steps'] = self._total_steps
        info['rewards'] = self._episode_rewards
        return (obs, rew, done, info)

    def get_state(self):
        return {
            'env_id': self.env.unwrapped.spec.id,
            'episode_data': {
                'episode_rewards': self._episode_rewards,
                'episode_lengths': self._episode_lengths,
                'episode_end_times': self._episode_end_times,
                'initial_reset_time': 0,
            }
        }

    def set_state(self, state):
        assert state['env_id'] == self.env.unwrapped.spec.id
        ed = state['episode_data']
        self._episode_rewards = ed['episode_rewards']
        self._episode_lengths = ed['episode_lengths']
        self._episode_end_times = ed['episode_end_times']