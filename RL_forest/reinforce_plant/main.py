import gym
from reinforce import reinforce
from utils import set_global_seeds, ObsProcessor
import tensorflow as tf
import config

# lr, gamma, action_dict,
def train(env_id, seed, action_mapping, **kargs):
    set_global_seeds(seed)
    env = gym.make(env_id)

    sess = tf.Session()

    # only discrete actions
    num_output = env.action_space.n
    action_dict = None
    if action_mapping:
        action_dict = config.action_dict
        num_output = len(action_dict)

    def make_processor():
        return ObsProcessor(env.observation_space.shape[:2], crop_area=(0, 35, 160, 195), resize_shape=(80, 80), flatten=True)

    obs_processor = make_processor()
    reinforce(env, sess, obs_processor, action_dict=action_dict, input_size=obs_processor.out_shape,
              num_output=num_output, **kargs)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env-id', help='environment ID', default='Pong-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--num-hid-layers', type=int, default=1)
    parser.add_argument('--hid-size', type=int, default=200)
    parser.add_argument("--action-mapping", action="store_true", default=False)
    args = parser.parse_args()
    reinforce(**args)
