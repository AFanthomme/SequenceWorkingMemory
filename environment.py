import numpy as np

class CircularDots(object):
	"""CircularDots environment class 
		* n_dots : number of dots on the circle to consider
		* T : length of sequences (can be overridden)

	"""
	def __init__(self, n_dots, T=3):
		super(CircularDots, self).__init__()
		self.n_dots = n_dots
		self.T = T

		self.dot_positions = np.zeros((n_dots, 2))
