# https://github.com/openai/baselines/baselines/ddpg/main.py
# the TensorBoardOutputFormat of logger of baselines is perfect.

import argparse
import time
import os
from tempfile import mkdtemp
import sys
import json
import gym_starcraft.envs.war_map_battle_env as sc

from misc_util import (
    set_global_seeds,
    boolean_flag )

import training
from models import Conv_Actor, Conv_Critic
from memory import Memory
from noise import *

import gym
import tensorflow as tf
import os

def run(env_id, seed, noise_type, layer_norm, logdir, evaluation, nb_units, ip, port, frame_skip, **kwargs):
    kwargs['logdir'] = logdir
    print("Well I am going to print the ip", ip)
    # remove evaluation environment.
    if env_id == "StarCraft":
        env = sc.WarMapBattleEnv(ip, port, frame_skip = frame_skip)
    else:
        env = gym.make(env_id)

    # Parse noise_type
    action_noise = None
    nb_actions = env.action_space.shape
    nb_unit_actions = env.nb_unit_actions
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                        sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape, 
        unit_location_shape=env.unit_location_shape, mask_shape=env.mask_shape)
    critic = Conv_Critic(layer_norm=layer_norm, time_step=nb_units)
    actor = Conv_Actor(nb_unit_actions, layer_norm=layer_norm, time_step=nb_units)

    # Seed everything to make things reproducible.

    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    start_time = time.time()
    training.train(env=env, action_noise=action_noise, actor=actor, critic=critic, memory=memory,
                   evaluation=evaluation ,**kwargs)

    env.close()
    print('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-id', type=str, default='StarCraft')
    parser.add_argument('--ip', help="server ip")
    parser.add_argument('--port', help="server port", type=int, default=11111)
    parser.add_argument('--save-epoch-interval', type=int, default=5)

    parser.add_argument('--nb-units', type=int, default=5)
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    parser.add_argument('--seed', type=int, default=123457)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=2e-5)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=32)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-cycles', type=int, default=20)  # per epoch cycle and MPI worker
    parser.add_argument('--frame-skip', type=int, default=2)
    # parser.add_argument('--nb-rollout-steps', type=int, default=300)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str,
                        default='ou_0.1')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--logdir', type=str, default='checkpoints')
    boolean_flag(parser, 'evaluation', default=True)


    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    # Figure out what logdir to use.
    if args['logdir'] is None:
        args['logdir'] = os.getenv('OPENAI_LOGDIR')

    # Print and save arguments.
    print('Arguments:')
    for key in sorted(args.keys()):
        print('{}: {}'.format(key, args[key]))
    print('')
    if not os.path.exists(args['logdir']):
        os.mkdir(args['logdir'])

    if args['logdir']:
        with open(os.path.join(args['logdir'], 'args.json'), 'w') as f:
            json.dump(args, f)
    # Run actual script.
    run(**args)
