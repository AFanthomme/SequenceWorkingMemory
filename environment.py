import numpy as np
import torch as tch


class Encoder(tch.nn.Module):
	"""docstring for Encoder"""
	def __init__(self, in_size=2, encoding_size=128):
		super(Encoder, self).__init__()
		self.encoding_size = encoding_size
		self.in_size = in_size
		self.layer = tch.nn.Linear(self.in_size, self.encoding_size)
		self.activation = tch.nn.ReLU

	def forward(self, input_batch):
		return self.activation(self.layer(input_batch))
		


class CircularDots(object):
	"""CircularDots environment class 
		* n_dots : number of dots on the circle to consider
		* T : length of sequences (can be overridden)

	"""
	def __init__(self, n_dots=6, T=3):
		super(CircularDots, self).__init__()
		self.n_dots = n_dots
		self.T = T
		self.dot_positions = np.zeros((n_dots, 2))

		for i in range(self.n_dots):
			theta = 2 * np.pi * i / self.n_dots
			self.dot_positions[i] = np.array([np.cos(theta), np.sin(theta)])
			
		self.encoder = Encoder()

	def get_sequences(self, bs=64, T=None):
		if T is None:
			T = self.T

		indices = np.random.randint(max=self.n_dots, size=(bs, T))
		positions = self.dot_positions[indices]
		encodings = self.encoder(sequences_positions)

		return encodings, positions, indices

if __name__ == '__main__':
	env = CircularDots(n_dots=6, T=3)
	encodings, positions, indices = env.get_sequences(bs=10)
	print(indices[0], positions[0], encodings[0].shape)





