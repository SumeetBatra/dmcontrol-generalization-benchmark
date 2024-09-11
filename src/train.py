import torch
import os
import numpy as np
import gym
import time
import wandb
from src.arguments import parse_args
from src.env.wrappers import make_env
from src.algorithms.factory import make_agent
from src.logger import Logger
from src.video import VideoRecorder
from src import utils
from typing import Dict, Any

use_wandb = True

def _setup_wandb(args) -> None:
    """Sets up wandb experiment tracking if enabled"""
    wandb.init(
        project='ai_sim2real',
        entity='qdrl',
        group=args.wandb_group,
        name=args.wandb_name,
    )

def evaluate(env, agent, video, num_episodes, L, step, suffix=''):
	episode_rewards = []
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs)
			obs, reward, done, _ = env.step(action)
			video.record(env)
			episode_reward += reward

		if L is not None:
			video.save(f'{step}{suffix}.mp4')
			L.log(f'eval/episode_reward{suffix}', episode_reward, step)
			if use_wandb:
				wandb.log({
					'env_step': step,
					f'eval/episode_reward{suffix}': episode_reward
				})
		episode_rewards.append(episode_reward)
	
	return np.mean(episode_rewards)


def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		frame_stack=args.frame_stack,
		image_size=args.image_size,
		mode='train'
	)
	color_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		frame_stack=args.frame_stack,
		image_size=args.image_size,
		mode='color_hard',
		intensity=args.distracting_cs_intensity
	)

	distracting_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		frame_stack=args.frame_stack,
		image_size=args.image_size,
		mode='distracting_cs',
		intensity=args.distracting_cs_intensity
	)

	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, str(args.seed))
	print('Working directory:', work_dir)
	assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
	utils.write_info(args, os.path.join(work_dir, 'info.log'))

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	replay_buffer = utils.ReplayBuffer(
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		capacity=args.train_steps,
		batch_size=args.batch_size
	)
	cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
	print('Observations:', env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)

	start_step, episode, episode_reward, done = 0, 0, 0, True
	L = Logger(work_dir)
	start_time = time.time()
	for step in range(start_step, args.train_steps+1):
		if done:
			if step > start_step:
				L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				L.dump(step)

			# Evaluate agent periodically
			if step % args.eval_freq == 0:
				print('Evaluating:', work_dir)
				L.log('eval/episode', episode, step)
				evaluate(env, agent, video, args.eval_episodes, L, step)
				evaluate(color_env, agent, video, args.eval_episodes, L, step, suffix='_color_hard')
				evaluate(distracting_env, agent, video, args.eval_episodes, L, step, suffix='_distracting_cs')
				L.dump(step)

			# Save agent periodically
			if step > start_step and step % args.save_freq == 0:
				torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

			L.log('train/episode_reward', episode_reward, step)
			if use_wandb:
				wandb.log({
					'env_step': step,
					'train/episode_reward': episode_reward,
				})

			obs = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1

			L.log('train/episode', episode, step)

		# Sample action for data collection
		if step < args.init_steps:
			action = env.action_space.sample()
		else:
			with utils.eval_mode(agent):
				action = agent.sample_action(obs)

		# Run training update
		if step >= args.init_steps:
			num_updates = args.init_steps if step == args.init_steps else 1
			for _ in range(num_updates):
				agent.update(replay_buffer, L, step)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		replay_buffer.add(obs, action, reward, next_obs, done_bool)
		episode_reward += reward
		obs = next_obs

		episode_step += 1

	print('Completed training for', work_dir)


if __name__ == '__main__':
	args = parse_args()
	if use_wandb:
		_setup_wandb(args)
	main(args)
