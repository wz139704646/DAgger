import numpy as np
from abc import abstractmethod

import matplotlib.pyplot as plt
from numpy.core.numeric import indices
from pynput.keyboard import Key, Listener
import threading
import time
import numpy as np
import random
import pickle


class DaggerAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass


# here is an example of creating your own Agent
class ExampleAgent(DaggerAgent):
	def __init__(self, necessary_parameters=None):
		super(DaggerAgent, self).__init__()
		# init your model
		self.model = None

	# train your model with labeled data
	def update(self, data_batch, label_batch):
		self.model.train(data_batch, label_batch)

	# select actions by your model
	def select_action(self, data_batch):
		label_predict = self.model.predict(data_batch)
		return label_predict


class QueryAgent(DaggerAgent):
	UP = Key.up
	DOWN = Key.down
	LEFT = Key.left
	RIGHT = Key.right
	FIRE = Key.space
	ENTER = Key.enter
	ESC = Key.esc
	DEL = Key.backspace
	hot_keys = [UP, DOWN, LEFT, RIGHT, FIRE, ENTER, ESC, DEL]
	NAMES = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R', FIRE: 'J'}
	ACTIONS = {
		0: 'NOOP',
		1: 'FIRE',
		2: 'UP',
		3: 'RIGHT',
		4: 'LEFT',
		5: 'DOWN',
		6: 'UP RIGHT',
		7: 'UP LEFT',
		8: 'DOWN RIGHT',
		9: 'DOWN LEFT',
		10: 'UP FIRE',
		11: 'RIGHT FIRE',
		12: 'LEFT FIRE',
		13: 'DOWN FIRE',
		14: 'UP RIGHT FIRE',
		15: 'UP LEFT FIRE',
		16: 'DOWN RIGHT FIRE',
		17: 'DOWN LEFT FIRE'
	}

	def __init__(self, **kwargs):
		super(QueryAgent, self).__init__()

		self.keys_list = [] # record the keys pressed
		self.keys_lock = threading.Lock()
		self.wait_input = False
		self.exit = False

		self.input_interval = 0.1 # interval during two inputs
		self.fresh_interval = 0.05 # interval for freshing frames
		self.font = {'color': 'blue', 'size': 15, 'family': 'Times New Roman'}

	# select action by querying expert
	def select_action(self, data_batch):
		# start monitoring the keyboard
		listener = self.init_keyboard_listener()
		listener.start()

		plt.ion()
		plt.cla()
		actions = []
		img_ax = None
		text_ax = None
		for i in range(len(data_batch)):
			d = data_batch[i]
			# show every frame
			if img_ax == None:
				img_ax = plt.imshow(d.transpose((1, 2, 0)))
			else:
				img_ax.set_data(d.transpose(1, 2, 0))

			if text_ax == None:
				text_ax = plt.text(-50, -10, "", fontdict=self.font)

			# wait for expert(keyboard) input
			self.wait_input = True
			while self.wait_input:
				# show current input
				cur_action = self.keys_to_action()
				cur_input = self.keys_to_namestr()
				text_ax.set_text(
					"{}-Inputs: {} ===> Action: {}, {}".format(
						i, cur_input, QueryAgent.ACTIONS[cur_action], cur_action
					))

				plt.pause(self.fresh_interval)

			if self.exit:
				# stop immediately
				break

			actions.append(self.keys_to_action())
			self.keys_list.clear()

			time.sleep(self.input_interval)

		plt.cla()
		plt.close()
		listener.stop()

		if self.exit:
			raise Exception("Query Agent Exits")

		return actions

	def update(self, data_batch, label_batch):
		pass

	def load(self, filepaths):
		pass

	def save(self, filepaths):
		pass

	def keys_to_namestr(self):
		return ','.join([QueryAgent.NAMES[k] for k in self.keys_list])

	def keys_to_action(self):
		directions = [
			QueryAgent.UP in self.keys_list,
			QueryAgent.RIGHT in self.keys_list,
			QueryAgent.LEFT in self.keys_list,
			QueryAgent.DOWN in self.keys_list ]
		jump = QueryAgent.FIRE in self.keys_list

		if sum(directions) > 2:
			# too many keys pressed, return noop
			return 0
		elif sum(directions) == 0:
			action = 1 if jump else 0 # -3 means no events
		else:
			if directions == [1, 0, 0, 0]:
				action = 2
			elif directions == [0, 1, 0, 0]:
				action = 3
			elif directions == [0, 0, 1, 0]:
				action = 4
			elif directions == [0, 0, 0, 1]:
				action = 5
			elif directions == [1, 1, 0, 0]:
				action = 6
			elif directions == [1, 0, 1, 0]:
				action = 7
			elif directions == [0, 1, 0, 1]:
				action = 8
			elif directions == [0, 0, 1, 1]:
				action = 9
			else:
				# UP & DOWN or LEFT & RIGHT as NOOP
				action = 0

			if jump and action != 0:
				action += 8

		return action

	def init_keyboard_listener(self):
		def key_pressed(key):
			# key pressed will modify the keys_list
			if self.wait_input and key not in self.keys_list \
				and key in QueryAgent.hot_keys \
				and self.keys_lock.acquire(blocking=False):
				if key == QueryAgent.ENTER:
					self.wait_input = False
				elif key == QueryAgent.DEL:
					if len(self.keys_list) > 0:
						self.keys_list.pop()
				elif key == QueryAgent.ESC:
					self.exit = True
					self.wait_input = False
				else:
					self.keys_list.append(key)

				self.keys_lock.release()

		def key_released(key):
			# do nothing
			pass

		return Listener(on_press=key_pressed, on_release=key_released)


class MyDaggerAgent(DaggerAgent):
	def __init__(
		self, imitator, batch_size, update_schedule,
		logger, cur_update=0, **kwargs):
		super(DaggerAgent, self).__init__()

		self.imitator = imitator
		self.batch_size = batch_size
		self.update_schedule = update_schedule # start, interval and end
		self.cur_update = cur_update
		self.logger = logger
		self.global_step = 0 # for logger

	# train your model with labeled data
	def update(self, data_batch, label_batch):
		_data = np.array(data_batch)
		_label = np.array(label_batch)
		tot_num = len(data_batch)
		batch_size = min(self.batch_size, tot_num)
		indices = list(range(tot_num))

		sel = max if self.update_schedule[1] < 0 else min
		epochs = int(sel(
			self.update_schedule[0] + self.update_schedule[1]*self.cur_update,
			self.update_schedule[2]))
		for i in range(epochs):
			random.shuffle(indices)
			for j in range(tot_num // batch_size):
				ind = indices[j*batch_size:(j+1)*batch_size]
				res = self.imitator.update(_data[ind], _label[ind])

			# log the results of each update epoch
			self.logger.logkv(res, self.global_step + i, tag='train')
		self.global_step += epochs

		# log the accuracy of the imitator
		ind = random.sample(indices, batch_size)
		outputs = self.select_action(_data[ind])
		acc = np.sum(outputs == _label[ind]) / len(outputs)
		self.logger.logkv({"imitator_acc": acc}, self.cur_update, tag='train')

		self.cur_update += 1

	# select actions by your model
	def select_action(self, data_batch):
		logits = self.imitator(data_batch).detach()
		return np.argmax(logits, axis=-1).tolist()

	def save(self, filepaths):
		self.imitator.save(filepaths['imitator'])

		with open(filepaths['train'], 'wb') as f:
			pickle.dump({
				'cur_update': self.cur_update,
				'global_step': self.global_step
			}, f)

	def load(self, filepaths):
		self.imitator.load(filepaths['imitator'])

		with open(filepaths['train'], 'rb') as f:
			train_cfg = pickle.load(f)
		self.cur_update = train_cfg['cur_update']
		self.global_step = train_cfg['global_step']

	def train(self, mode=True):
		self.imitator.train(mode)