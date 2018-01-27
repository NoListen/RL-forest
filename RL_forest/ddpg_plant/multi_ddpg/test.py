import tensorflow as tf
import argparse
import numpy as np
from models import Conv_Actor, Conv_Critic, Dynamic_Conv_Actor, Dynamic_Conv_Critic, Dynamic_Actor, Dynamic_Critic
from dynamic_multi_ddpg import Dynamic_DDPG
import gym_starcraft.envs.compound_battle_env as dsc
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="checkpoints/model")
parser.add_argument('--ip', help="server ip")
parser.add_argument('--port', help="server port", type=int, default=11111)
parser.add_argument('--dynamic', default=True)
parser.add_argument('--nb-units', type=int, default=15)
parser.add_argument('--critic-l2-reg', type=float, default=1e-3)
parser.add_argument('--frame-skip', type=int, default=2)
parser.add_argument('--n-hidden', type=int, default=64)

vars_dict = vars(parser.parse_args())
graph_path = vars_dict["model"]+".meta"
print("the path is", graph_path)
saver = tf.train.import_meta_graph(graph_path)

def get_obs_tensor(observation_keys, name, graph):
    d = {}
    for k in observation_keys:
        d[k] = graph.get_tensor_by_name(name+"_"+k+":0")
        print(k)
    return d

env = dsc.CompoundBattleEnv(vars_dict["ip"], vars_dict["port"], frame_skip=vars_dict["frame_skip"], map_types_table=("unit_data",))
observation_shape=env.observation_shape
observation_dtype=env.observation_dtype
nb_unit_actions = env.nb_unit_actions
print("env set up")

num_win = 0
with tf.Session() as sess:
    saver.restore(sess, vars_dict["model"])
    graph = tf.get_default_graph()
    obs = get_obs_tensor(observation_dtype.keys(),"obs0", graph)
    env_obs = env.reset()
    done = False
    output = graph.get_tensor_by_name("actor/Tanh:0")
    assert(set(env_obs.keys()) == set(obs.keys()))
    for cycle in tqdm(range(10), ncols=50):
        while not done:
            #print([ k for k in env_obs.keys()])
            feed_dict = {obs[k]: [env_obs[k]] for k in env_obs.keys()}
            action = sess.run(output, feed_dict=feed_dict)
            action = np.squeeze(action, [0])
            # TODO temporally ignore max_action
            env_obs, r, done, info = env.step(action)
            if done:
                if env._check_win():
                    num_win += 1
                env_obs = env.reset()
        done = False



    
