import tensorflow as tf
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="checkpoints/model")
vars_dict = vars(parser.parse_args())
sess = tf.Session()
graph_path = vars_dict["model"]+".meta"
print("the path is", graph_path)
saver = tf.train.import_meta_graph(graph_path)
saver.restore(sess, vars_dict["model"])
with sess:
    #with tf.variable_scope('critic', reuse=True):
        #w1 = tf.get_variable('conv2/W')
    np_w1 = sess.run("critic/conv1d/kernel:0")
    print(np_w1)
    #m1 = np_w1[...,:24]
    #m1_max = np.max(m1)
    #m1_min = np.min(m1)
   # m1_mean = np.mean(m1)

    #m2 = np_w1[...,24:]
    #m2_min = np.min(m2)
    #m2_max = np.max(m2)
    #m2_mean = np.mean(m2)
    #print("m1's situation",m1_max, m1_min, m1_mean)
    #print("m2's situation",m2_max, m2_min, m2_mean)
