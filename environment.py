import numpy as np
import torch as tch

device = tch.device('cuda:0')

letter_to_int = {'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E':14, 'F':15, 'G':16, 'H': 17, 'I': 18, 'J':19, 'K':20, 'L': 21, 'M':22,
					'N': 23, 'O': 24}


class ObservationNet(tch.nn.Module):
	"""docstring for StaticEncoder"""
	def __init__(self, in_size=2, observation_size=128, device=device):
		super(ObservationNet, self).__init__()
		self.device = device
		self.observation_size = observation_size
		self.in_size = in_size
		self.layer = tch.nn.Linear(self.in_size, self.observation_size)
		self.layer2 = tch.nn.Linear(self.observation_size, self.observation_size)
		self.layer3 = tch.nn.Linear(self.observation_size, self.observation_size)
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
	def __init__(self, n_dots=6, T=3, device=device, observation_size=128, load_from=None):
		super(CircularDots, self).__init__()
		self.n_dots = n_dots
		self.dot_angles = np.zeros((n_dots))
		self.dot_positions = np.zeros((n_dots, 2))
		self.T = T
		self.device = device
		self.observation_size = observation_size
		self.n_seqs = (self.n_dots**T)

		for i in range(self.n_dots):
			theta = 2 * np.pi * i / self.n_dots
			self.dot_angles[i] = theta
			self.dot_positions[i] = np.array([np.cos(theta), np.sin(theta)])

		self.dot_positions = tch.from_numpy(self.dot_positions).float().to(self.device)
		self.observation_net = ObservationNet(device=device, observation_size=observation_size)

		if load_from is not None:
			self.load(load_from)

	def make_seq(self, seq_hash):
		tmp = np.zeros(self.T, np.int32)
		rep = np.base_repr(seq_hash, base=self.n_dots)[::-1]
		# print('rep', rep)
		for idx, val in enumerate(rep):
			try:
				tmp[idx] = int(val)
			except ValueError:
				tmp[idx] = letter_to_int[val]
		return tmp

	def get_sequences(self, bs=64, T=None):
		if T is None:
			T = self.T

		indices = tch.from_numpy(np.random.randint(self.n_dots, size=(bs, T))).to(self.device).long()
		positions = self.dot_positions[indices]
		observations = self.observation_net(positions)

		return observations, positions, indices

	def generate_all_sequences(self, bs=64):
		n_seqs = self.n_dots ** self.T
		assert n_seqs < 10**8, 'Number of possible sequences too large for exhaustive enumeration'
		count = 0
		while count*bs < n_seqs:
			seq_batch = np.zeros((bs, self.T), np.int32)
			plop = np.arange(count*bs, min((count+1)*bs, n_seqs))
			for idx, item in enumerate(plop):
				seq_batch[idx] = self.make_seq(item)
			count += 1
			observations_batch = self.observation_net(self.dot_positions[seq_batch])
			yield observations_batch[:len(plop)], seq_batch[:len(plop)]


	def save(self, filename):
		tch.save(self.observation_net.state_dict(), filename)

	def load(self, filename):
		self.observation_net.load_state_dict(tch.load(filename))



if __name__ == '__main__':
	env = CircularDots(n_dots=6, T=3)
	encodings, positions, indices = env.get_sequences(bs=10)
	# print(indices[0], positions[0], encodings[0].shape)
	print(indices.shape, positions.shape, encodings.shape)
