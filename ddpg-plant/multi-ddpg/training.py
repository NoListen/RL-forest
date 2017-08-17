import os
import time
from collections import deque
import pickle
from tqdm import tqdm
from multi_ddpg import DDPG
import numpy as np
import tensorflow as tf

# change the training from step to be episode like training.
# train the data while running in the environment !=
# DON'T FEAR, WE ARE THE GPU !
# TODO Add saving checkpoints fucntion.
# save this session.

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, actor, critic,
          critic_l2_reg, actor_lr, critic_lr, action_noise, logdir,
          gamma, clip_norm, nb_train_steps, nb_eval_cycles, batch_size, memory, evaluation,
          tau=0.01, eval_env=None, save_epoch_interval=None):

    # indeed [-1. 1]
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    print('scaling actions by {} before executing in env'.format(max_action))
    # the observation shape doesn't means the shape of obs returned by the environment.
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.mask_shape, env.action_space.shape,
                 gamma=gamma, tau=tau, batch_size=batch_size, action_noise=action_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, clip_norm=clip_norm,
                 reward_scale=reward_scale)
    print('Using agent with the following configuration:')
    print(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if save_epoch_interval:
        saver = tf.train.Saver()
    else:
        saver = None

    """ In marine's setting, each episode only lasts no more than 200 steps """
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    # Config proto
    with tf.Session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()
        epoch_actor_losses = []
        epoch_critic_losses = []

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        log_file = open('log','a')
        for epoch in range(nb_epochs):
            epoch_start_time = time.time()
            for cycle in tqdm(range(nb_epoch_cycles), ncols=50): # well, each episode is considered as one rollout
                # Perform rollouts.
                while not done:
                    # Predict next action.
                    # TODO the call function has been changed in the model file, so we need to change the ddpg file.

                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    if render:
                        env.render()
                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step( max_action * action)
                    # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    t += 1
                    if render:
                        env.render()

                    episode_reward += r
                    episode_step += 1
                    r = max(min(r, 1), -1)
                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    #print(t, done)
                    if agent.memory.length > 50*batch_size:
                        cl, al = agent.train()
                        epoch_critic_losses.append(cl)
                        epoch_actor_losses.append(al)
                        agent.update_target_net()

                    # TODO revise the agent to include the mask.
                    # TODO scale the data in the map.

                    # TODO v1 outline of circle.
                    # TODO v2 filled circle.
                    # TODO v3 accumulated circle.

                    # TODO Consider the enemy's data information, i.e. obs[1]
                    # TODO Data is normalized and includes the scale.
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs
                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()
                done = False
                
                
                # Train.
             #   for t_train in range(nb_train_steps):
             #       cl, al = agent.train()
             #       epoch_critic_losses.append(cl)
             #       epoch_actor_losses.append(al)
             #       agent.update_target_net()
                # Evaluate. OBS reset has not been used yet.
            eval_wins = 0
            eval_episode_rewards = []
            eval_qs = []
            if evaluation:
               eval_episode_reward = 0.
               # TODO change the evaluation method.
               for t_rollout in tqdm(range(nb_eval_cycles), ncols=50):
                   while not done:
                       eval_action, eval_q = agent.pi(obs, apply_noise=False, compute_Q=True)
                       obs, eval_r, done, eval_info = env.step(
                           max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                       if render_eval:
                           eval_env.render()
                       eval_episode_reward += eval_r
                       eval_qs.append(eval_q)
                       if done:
                           if eval_episode_reward > 0:
                               eval_wins += 1
                           obs = env.reset()
                           eval_episode_rewards.append(eval_episode_reward)
                           eval_episode_rewards_history.append(eval_episode_reward)
                           eval_episode_reward = 0.
                   done = False
            if save_epoch_interval and epoch%save_epoch_interval == 0:
               saver.save(sess, logdir+'/epoch_%i_winrate_%.2f' % (epoch, float(eval_wins)/nb_eval_cycles), global_step = t)
           # Log stats.
            epoch_train_duration = time.time() - epoch_start_time
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = {}
            for key in sorted(stats.keys()):
                combined_stats[key] = np.mean(stats[key])

            # Rollout statistics.


            combined_stats['rollout/avergae_return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/average_return over 100 episodes'] = np.mean(np.mean(episode_rewards_history))
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['rollout/duration'] = epoch_train_duration

            epoch_episode_rewards = []
            epoch_episode_steps = [] # How many steps each episode
            epoch_actions = []
            epoch_qs = []

            # Train statistics.
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            epoch_actor_losses = []
            epoch_critic_losses = []

            # Evaluation statistics.
            if evaluation:
                combined_stats['eval/return'] = np.mean(eval_episode_rewards)
                combined_stats['eval/return_history'] = np.mean(np.mean(eval_episode_rewards_history))
                combined_stats['eval/Q'] = np.mean(eval_qs)
                combined_stats['eval/episodes'] = len(eval_episode_rewards)
                combined_stats['eval/win_rate'] = float(eval_wins)/nb_eval_cycles

            # Total statistics.
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = np.mean(episodes)
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t
            log_file.write("epoch %i\n" % epoch)
            log_file.write("#############################\n")
            for key in sorted(combined_stats.keys()):
                log_file.write("%s\t%s\n" % (key, str(combined_stats[key])))
            log_file.write("#############################\n")
            log_file.flush()
            print("epoch %i finished" % epoch)
        log_file.close()
