import random
import pickle
import numpy as np

class LinearQModel:

	def __init__(self, learning_rate, gamma, actions, n_features):
		self.lr = learning_rate
		self.gamma = gamma
		self.actions = actions

		self.weights = {
			action: np.zeros(n_features, dtype=float)
			for action in actions
		}

	def choose_action(self, state, epsilon):
		"""epsilon-greedy strategy"""

		# Exploration:
		if random.random() < epsilon:
			return random.choice(self.actions)

		# Exploitation:
		q_values = {
			action: self.get_q_value(state, action)
			for action in self.actions
		}
		max_q = max(q_values.values())

		best_actions = [
			act for act, q in q_values.items() if q == max_q
		]
		return random.choice(best_actions)

	def get_q_value(self, state, action):
		"""Returns Q-value for a pair (state, action)"""
		features = np.asarray(state, dtype=float)
		return np.dot(
			self.weights[action],
			features
		)

	def update(self, transition):
		"""Update using Bellman's equation."""

		state = np.asarray(transition.state, dtype=float)
		next_state = transition.next_state
		action = transition.action
		reward = transition.reward

		current_q = self.get_q_value(state, action)

		if next_state is None:
			target = reward
		else:
			next_q_values = [
				self.get_q_value(next_state, next_action)
				for next_action in self.actions
			]

			target = (reward + self.gamma * max(next_q_values))

		# TD error
		td_error = target - current_q

		# Linear Q-learning update
		self.weights[action] += (
			self.lr
			* td_error
			* state
		)


	def save(self, filename):
		with open(filename, "wb") as file:
			pickle.dump(self.weights, file)


	def load(self, file):
		self.weights = pickle.load(file)
