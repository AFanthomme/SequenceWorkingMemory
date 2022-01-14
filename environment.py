import numpy as np
import torch as tch

device = tch.device('cuda:0')

class StaticEncoder(tch.nn.Module):
	"""docstring for StaticEncoder"""
	def __init__(self, in_size=2, encoding_size=128, device=device):
		super(StaticEncoder, self).__init__()
		self.device = device
		self.encoding_size = encoding_size
		self.in_size = in_size
		self.layer = tch.nn.Linear(self.in_size, self.encoding_size)
		self.layer2 = tch.nn.Linear(self.encoding_size, self.encoding_size)
		self.layer3 = tch.nn.Linear(self.encoding_size, self.encoding_size)
		self.activation = tch.nn.ReLU()
		self.layer.to(self.device)
		self.layer2.to(self.device)
		self.layer3.to(self.device)

	def forward(self, x):
		x = self.activation(self.layer(x))
		x = self.activation(self.layer2(x))
		x = self.activation(self.layer3(x))
		return x



class CircularDots(object):
	"""CircularDots environment class
		* n_dots : number of dots on the circle to consider
		* T : length of sequences (can be overridden)

	"""
	def __init__(self, n_dots=6, T=3, device=device, encoding_size=128):
		super(CircularDots, self).__init__()
		self.n_dots = n_dots
		self.dot_positions = np.zeros((n_dots, 2))
		self.T = T
		self.device = device
		self.encoding_size = encoding_size

		for i in range(self.n_dots):
			theta = 2 * np.pi * i / self.n_dots
			self.dot_positions[i] = np.array([np.cos(theta), np.sin(theta)])

		self.dot_positions = tch.from_numpy(self.dot_positions).float().to(self.device)
		self.static_encoder = StaticEncoder(device=device, encoding_size=encoding_size)

	def get_sequences(self, bs=64, T=None):
		if T is None:
			T = self.T

		indices = tch.from_numpy(np.random.randint(self.n_dots, size=(bs, T))).to(self.device).long()
		positions = self.dot_positions[indices]
		encodings = self.static_encoder(positions)

		return encodings, positions, indices



if __name__ == '__main__':
	env = CircularDots(n_dots=6, T=3)
	encodings, positions, indices = env.get_sequences(bs=10)
	# print(indices[0], positions[0], encodings[0].shape)
	print(indices.shape, positions.shape, encodings.shape)
