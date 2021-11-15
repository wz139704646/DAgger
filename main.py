from numpy.lib.npyio import load, save
from arguments import get_args
from Dagger import MyDaggerAgent, QueryAgent
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# from gym.utils import play
from actors import CNNActor
import pickle
import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from logger import TensorboardLogger
import random
import traceback
import sys


def plot(record, save_file='performance.png'):
	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(record['steps'], record['mean'],
	        color='blue', label='reward')
	ax.fill_between(record['steps'], record['min'], record['max'],
	                color='blue', alpha=0.2)
	ax.set_xlabel('number of steps')
	ax.set_ylabel('Average score per episode')
	ax1 = ax.twinx()
	ax1.plot(record['steps'], record['query'],
	         color='red', label='query')
	ax1.set_ylabel('queries')
	reward_patch = mpatches.Patch(lw=1, linestyle='-', color='blue', label='score')
	query_patch = mpatches.Patch(lw=1, linestyle='-', color='red', label='query')
	patch_set = [reward_patch, query_patch]
	ax.legend(handles=patch_set)
	fig.savefig(save_file)

	plt.close(fig)


# the wrap is mainly for speed up the game
# the agent will act every num_stacks frames instead of one frame
class Env(object):
	def __init__(self, env_name, num_stacks):
		self.env = gym.make(env_name)
		# num_stacks: the agent acts every num_stacks frames
		# it could be any positive integer
		self.num_stacks = num_stacks
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space

	def step(self, action):
		reward_sum = 0
		for stack in range(self.num_stacks):
			obs_next, reward, done, info = self.env.step(action)
			reward_sum += reward
			if done:
				self.env.reset()
				return obs_next, reward_sum, done, info
		return obs_next, reward_sum, done, info

	def reset(self):
		return self.env.reset()

	def close(self):
		return self.env.close()


def preprocess(obs):
	"""preprocess obs data"""
	return np.transpose(obs, (2, 0, 1))


def get_save_files(base_dir):
	return {
		'agent': {
			'imitator': os.path.join(base_dir, 'imitator.pkl'),
			'train': os.path.join(base_dir, 'train_cfg.pkl')
		},
		'data': os.path.join(base_dir, 'data.pkl'),
		'record': os.path.join(base_dir, 'record.pkl')
	}


def ensure_dir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)


def main():
	# load hyper parameters
	args = get_args()
	num_updates = int(args.num_frames // args.num_steps)
	start = time.time()
	record = {'steps': [0],
	          'max': [0],
	          'mean': [0],
	          'min': [0],
	          'query': [0]}
	# query_cnt counts queries to the expert
	query_cnt = 0

	# seed
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	random.seed(args.seed)

	# environment initial
	envs = Env(args.env_name, args.num_stacks)
	# action_shape is the size of the discrete action set, here is 18
	# Most of the 18 actions are useless, find important actions
	# in the tips of the homework introduction document
	action_shape = envs.action_space.n
	# observation_shape is the shape of the observation
	# here is (210,160,3)=(height, weight, channels)
	observation_shape = envs.observation_space.shape
	h, w, c = observation_shape
	print(action_shape, observation_shape)

	ensure_dir(args.log_base)
	log_path = os.path.join(
		args.log_base, args.env_name, 'CNNActor',
		"seed_{}_{}".format(args.seed, datetime.datetime.now().strftime("%m%d-%H_%M_%S")))
	writer = SummaryWriter(log_path)
	logger = TensorboardLogger(writer)
	# log args
	logger.logs(str(args), 0, tag='args')

	# agent initial
	# you should finish your agent with DaggerAgent
	# e.g. agent = MyDaggerAgent()
	imitator = CNNActor(
		c, h, w, action_shape,
		lr=args.lr,
		device=args.device,
		hidden_sizes=args.hidden_sizes)
	my_agent = MyDaggerAgent(
		imitator, args.batch_size,
		args.update_epochs, logger)
	expert = QueryAgent()

	# initial saving configuration
	base_dir = args.checkpoints_base
	ensure_dir(base_dir)
	save_files = get_save_files(base_dir)

	# You can play this game yourself for fun
	if args.play_game:
		# Mode 1: play game frame by frame with image file
		# ensure_dir('./imgs')
		# obs = envs.reset()
		# while True:
		# 	im = Image.fromarray(obs)
		# 	im.save('imgs/' + str('screen') + '.jpeg')
		# 	action = int(input('input action'))
		# 	while action < 0 or action >= action_shape:
		# 		action = int(input('re-input action'))
		# 	obs_next, reward, done, _ = envs.step(action)
		# 	obs = obs_next
		# 	if done:
		# 		obs = envs.reset()

		# Mode 2: play game frame by frame use keyboard input
		agent = QueryAgent()
		obs = envs.reset()
		while True:
			im = Image.fromarray(obs)
			im.save('imgs/screen.jpeg')
			action = agent.select_action([obs])
			obs_next, reward, done, _ = envs.step(action)
			obs = obs_next
			if done:
				obs = envs.reset()

		# Mode 3: play game via pygame
		# play.play(envs.env)
		# return

	data_set = {'data': [], 'label': []}

	if args.reload:
		# continue training
		load_base = args.load_base
		ensure_dir(load_base)
		load_files = get_save_files(load_base)
		my_agent.load(load_files['agent'])

		# load data and records only when training
		if not args.test:
			with open(load_files['data'], 'rb') as f:
				data_set = pickle.load(f)
			with open(load_files['record'], 'rb') as f:
				record = pickle.load(f)

	if args.test:
		# turn the eval mode on
		my_agent.train(False)

		agent = my_agent
		# evaluate model by testing in the environment
		for i in range(args.test_times):
			obs = envs.reset()
			obs_ = preprocess(obs)
			reward_episode_set = []
			reward_episode = 0

			for _ in range(args.test_steps):
				action = agent.select_action([obs_])
				# render to get visual results
				if args.test_render:
					envs.env.render()

				obs_next, reward, done, _ = envs.step(action[0])
				reward_episode += reward
				obs = obs_next
				obs_ = preprocess(obs)
				if done:
					reward_episode_set.append(reward_episode)
					reward_episode = 0
					envs.reset()
			if len(reward_episode_set) == 0:
				reward_episode_set.append(reward_episode)

			print(
				"Test {}, num timesteps {}, avrage/min/max reward {:.1f}/{:.1f}/{:.1f}".format(
					i, args.test_steps,
					np.mean(reward_episode_set),
					np.min(reward_episode_set),
					np.max(reward_episode_set)
				)
			)

			record['steps'].append(i)
			record['mean'].append(np.mean(reward_episode_set))
			record['max'].append(np.max(reward_episode_set))
			record['min'].append(np.min(reward_episode_set))
			record['query'].append(0)

		envs.close()

		plot(record, 'performance_test.png')
		return

	# start train your agent
	try:
		if args.save_img:
			ensure_dir('./imgs')

		start_update = my_agent.cur_update
		for i in range(start_update, num_updates):
			# an example of interacting with the environment
			# we init the environment and receive the initial observation
			obs = envs.reset()
			# preprocessed observation
			obs_ = preprocess(obs)
			# we get a trajectory with the length of args.num_steps

			# turn the train mode on
			my_agent.train(True)

			if i == 0:
				# \beta_0 = 1
				agent = expert
			else:
				agent = my_agent

			for step in range(args.num_steps):
				# an example of saving observations
				if args.save_img:
					im = Image.fromarray(obs)
					im.save('imgs/' + str(step) + '.jpeg')
				data_set['data'].append(obs_)

				# Sample actions
				epsilon = 0.05
				if np.random.rand() < epsilon and agent == my_agent:
					# we choose a random action
					action = envs.action_space.sample()
				else:
					# we choose a special action according to our model
					if agent == expert:
						query_cnt += 1

					action = int(agent.select_action([obs_])[0])

				# interact with the environment
				# we input the action to the environments and it returns some information
				# obs_next: the next observation after we do the action
				# reward: (float) the reward achieved by the action
				# down: (boolean)  whether it’s time to reset the environment again.
				#           done being True indicates the episode has terminated.
				obs_next, reward, done, _ = envs.step(action)
				# we view the new observation as current observation
				obs = obs_next
				# if the episode has terminated, we need to reset the environment.
				if done:
					obs = envs.reset()

				obs_ = preprocess(obs)

			# You need to label the images in 'imgs/' by recording the right actions in label.txt

			# After you have labeled all the images, you can load the labels
			# for training a model
			# with open('/imgs/label.txt', 'r') as f:
			# 	for label_tmp in f.readlines():
			# 		data_set['label'].append(label_tmp)
			labels_ = expert.select_action(data_set['data'][i*args.num_steps:])
			data_set['label'].extend(labels_)
			query_cnt += len(labels_)

			# design how to train your model with labeled data
			my_agent.update(data_set['data'], data_set['label'])

			# save checkpoints
			if (i + 1) % args.save_interval == 0:
				my_agent.save(save_files['agent'])
				with open(save_files['data'], 'wb') as f:
					pickle.dump(data_set, f)
				with open(save_files['record'], 'wb') as f:
					pickle.dump(record, f)

			if (i + 1) % args.log_interval == 0:
				agent = my_agent
				total_num_steps = (i + 1) * args.num_steps
				obs = envs.reset()
				obs_ = preprocess(obs)
				reward_episode_set = []
				reward_episode = 0

				# turn the eval mode on
				my_agent.train(False)

				# evaluate your model by testing in the environment
				for step in range(args.test_steps):
					action = agent.select_action([obs_])
					# you can render to get visual results
					# envs.render()
					obs_next, reward, done, _ = envs.step(action)
					reward_episode += reward
					obs = obs_next
					obs_ = preprocess(obs)
					if done:
						reward_episode_set.append(reward_episode)
						reward_episode = 0
						envs.reset()
				if len(reward_episode_set) == 0:
					reward_episode_set.append(reward_episode)

				end = time.time()
				print(
					"TIME {} Updates {}, num timesteps {}, FPS {} \n query {}, avrage/min/max reward {:.1f}/{:.1f}/{:.1f}"
						.format(
						time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
						i, total_num_steps,
						int(total_num_steps / (end - start)),
						query_cnt,
						np.mean(reward_episode_set),
						np.min(reward_episode_set),
						np.max(reward_episode_set)
					))
				record['steps'].append(total_num_steps)
				record['mean'].append(np.mean(reward_episode_set))
				record['max'].append(np.max(reward_episode_set))
				record['min'].append(np.min(reward_episode_set))
				record['query'].append(query_cnt)
				plot(record)

	except Exception as e:
		print("Unexpected Exception: {}".format(e))
		traceback.print_exc()

	finally:
		envs.close()


if __name__ == "__main__":
	main()
	sys.exit()
