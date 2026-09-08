import random
import pickle

class TabularQModel:

	def __init__(self, learning_rate, gamma, actions):
		self.q_table = {}
		self.lr = learning_rate
		self.gamma = gamma
		self.actions = actions

	def choose_action(self, state, epsilon):
		"""epsilon-greedy strategy"""
		if state not in self.q_table:
			self.q_table[state] = {act: 0.0 for act in self.actions}

		# Exploration:
		if random.random() < epsilon:
			return random.choice(self.actions)

		# Exploitation:
		q_values = self.q_table[state]
		max_q = max(q_values.values())

		best_actions = [
			act for act, q in q_values.items() if q == max_q
		]
		return random.choice(best_actions)

	def get_q_value(self, state, action):
		"""Returns Q-value for a pair (state, action)"""
		if state not in self.q_table:
			return 0.0
		return self.q_table[state].get(action, 0.0)

	def update(self, transition):
		"""Update the q-table using Bellman's equation."""

		state = transition.state
		next_state = transition.next_state
		action = transition.action
		reward = transition.reward

		if state not in self.q_table:
			self.q_table[state] = {act: 0.0 for act in self.actions}

		if next_state is None:
			max_next_q = 0.0
		else:
			if next_state not in self.q_table:
				self.q_table[next_state] = {act: 0.0 for act in self.actions}
			max_next_q = max(self.q_table[next_state].values())

		# Q value for current (state, action)
		current_q = self.get_q_value(state, action)		

		# Q-Learning
		new_q = current_q + self.lr * (
			reward + self.gamma * max_next_q - current_q
		)

		self.q_table[state][action] = new_q

	def save(self, filename):
		try:
			with open(filename, "wb") as file:
				pickle.dump(self.q_table, file)
		except Exception as e:
			# self.logger.error()
			pass

	def load(self, file):
		try:
		# with open(filename, "rb") as f:
			self.q_table = pickle.load(file)
		except:
			print("ERROR")
		# self.logger.info(f"Modelo cargado con éxito PATATAAAA desde {filename}")
		# except FileNotFoundError:
		# 	self.logger.warning(
		# 		f"No se encontró el archivo {filename}. Se iniciará con una tabla Q vacía."
		# 	)
		# except Exception as e:
		# 	self.logger.error(f"Error al cargar el modelo: {e}")
