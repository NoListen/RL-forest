import gym
from reinforce import reinforce
from utils import set_global_seeds, ObsProcessor
import tensorflow as tf
import config
from pong import PongGame
import json
import os

# lr, gamma, action_dict,
# def train(env_id, seed, action_mapping, **kargs):
def train(seed,  model_dir, **kargs):
    set_global_seeds(seed)
    if model_dir[-1] == '/':
        model_dir = model_dir[:-1]

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    env = PongGame(frame_skip=1)

    sess = tf.InteractiveSession()

    # only discrete actions
    num_output = config.num_actions
    # action_dict = None
    # if action_mapping:
    #     action_dict = config.action_dict
    #     num_output = len(action_dict)

    def make_processor():
        return ObsProcessor(config.ob_shape, crop_area=None, resize_shape=(60, 60), flatten=True)

    obs_processor = make_processor()
    reinforce(env, sess, obs_processor, action_dict=None, input_size=obs_processor.out_shape,
              num_output=num_output, model_dir = model_dir, **kargs)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--env-id', help='environment ID', default='Pong-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num-hid-layers', type=int, default=1)
    parser.add_argument('--hid-size', type=int, default=200)
    parser.add_argument('--model-dir', default="checkpoints") # where to save the model
    parser.add_argument('--model-path', default=None) # which model to load
    parser.add_argument('--phase', default="train")
    # parser.add_argument("--action-mapping", action="store_true", default=False)
    args = vars(parser.parse_args())
    with open('args.json', 'w') as f:
        json.dump(args, f)
    train(**args)

if __name__ == '__main__':
    main()