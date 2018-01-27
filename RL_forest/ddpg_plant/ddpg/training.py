import os
import time
from collections import deque
import pickle

from ddpg import DDPG
import numpy as np
import tensorflow as tf


def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, actor, critic,
          critic_l2_reg, actor_lr, critic_lr, action_noise, logdir,
          gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
          tau=0.01, eval_env=None, save_iter=None, baseline=False):

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    print('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
                 gamma=gamma, tau=tau, batch_size=batch_size, action_noise=action_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, clip_norm=clip_norm,
                 reward_scale=reward_scale, baseline=baseline)
    print('Using agent with the following configuration:')
    print(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if save_iter:
        saver = tf.train.Saver()
    else:
        saver = None

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

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    if render:
                    # Execute next action.
                    # normally, I won't choose to render
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

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(
                            max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.

            # Log stats.
            epoch_train_duration = time.time() - epoch_start_time
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = {}
            for key in sorted(stats.keys()):
                combined_stats[key] = np.mean(stats[key])

            # Rollout statistics.


            combined_stats['rollout/avergae_return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/max_return'] = np.max(epoch_episode_rewards)
            combined_stats['rollout/average_return_100'] = np.mean(np.mean(episode_rewards_history))
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)

            epoch_episode_rewards = []
            epoch_episode_steps = [] # How many steps each episode
            epoch_actions = []
            epoch_qs = []

            # Train statistics.
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)

            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = np.mean(eval_episode_rewards)
                combined_stats['eval/return_history'] = np.mean(np.mean(eval_episode_rewards_history))
                combined_stats['eval/Q'] = np.mean(eval_qs)
                combined_stats['eval/episodes'] = len(eval_episode_rewards)

            # Total statistics.
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = np.mean(episodes)
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            print("#############################")
            for key in sorted(combined_stats.keys()):
                print(key, '\t', combined_stats[key])
            print("#############################")

