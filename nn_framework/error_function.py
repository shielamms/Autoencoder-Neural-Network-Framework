import numpy as np

class SqrErr():
	@staticmethod
	def calc(x, y):
		# Assume that x and y are arrays
		return (y - x)**2

	@staticmethod
	def calc_derivative(x, y):
		return 2 * (y - x)


class AbsErr():
	@staticmethod
	def calc(x, y):
		return np.abs(y - x)

	@staticmethod
	def calc_derivative(x, y):
		return np.sign(y - x)