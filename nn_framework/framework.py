import matplotlib.pyplot as plt
import numpy as np
import os

class ANN():
	def __init__(self,
				 model,
				 input_pixel_range=(-1,1),
				 error_func=None,
				 visualizer=None):
		self.model = model
		self.n_iter_train = int(1e8)
		self.n_iter_evaluate = int(1e4)
		self.input_pixel_range = input_pixel_range

		# reporting
		self.errors = []
		self.report_interval = int(1e5) # generate report every 1e5 iterations
		self.report_bin_size = int(1e3)
		self.report_min = -3
		self.report_max = 0
		self.report_path = 'reports'
		self.report_file = 'nn_performance_report.png'

		self.error_func = error_func
		self.visualizer = visualizer

		try:
			os.mkdir(self.report_path)
		except Exception:
			pass


	def train(self, training_set):
		for i in range(self.n_iter_train):
			x = next(training_set()).ravel() # flatten the 2D input to 1D
			x = self.normalize(x)
			y = self.forward_propagate(x)
			error = self.error_func.calc(x, y)
			error = (np.mean(error**2))**0.5	# root mean square
			error_derivative = self.error_func.calc_derivative(x, y)
			self.errors.append(error)

			if (i+1) % self.report_interval == 0:
				self.report()
				self.visualizer.render(self, x, 'train_iter'+str(i))

			self.back_propagate(error_derivative)

	def evaluate(self, evaluation_set):
		for i in range(self.n_iter_evaluate):
			x = next(evaluation_set()).ravel() # flatten the 2D input to 1D
			x = self.normalize(x)
			y = self.forward_propagate(x)
			error = self.error_func(x, y)
			self.errors.append(error)

			if i % self.report_interval == 0:
				self.report()
				self.visualizer.render(self, x, 'eval_iter'+str(i))

	def normalize(self, values):
		# Scale the values between -0.5 and 0.5
		min_val = self.input_pixel_range[0]
		max_val = self.input_pixel_range[1]
		scale_factor = max_val - min_val
		return (values - min_val) / scale_factor - 0.5

	def denormalize(self, transformed_values):
		# Use the inverse of the normalize function to convert the normalized
		# values back to the original values
		min_val = self.input_pixel_range[0]
		max_val = self.input_pixel_range[1]
		scale_factor = max_val - min_val
		return (transformed_values + 0.5) * (scale_factor + min_val)

	def forward_propagate(self, x):
		y = x.ravel()[np.newaxis, :] # ravel(): flatten to a single row
		for layer in self.model:
			y = layer.forward_propagate(y)
		return y.ravel()

	def forward_propagate_to_layer(self, x, i_layer):
		y = x.ravel()[np.newaxis, :] # ravel(): flatten to a single row
		for layer in self.model[:i_layer]:
			y = layer.forward_propagate(y)
		return y.ravel()

	def forward_propagate_from_layer(self, x, i_layer):
		y = x.ravel()[np.newaxis, :] # ravel(): flatten to a single row
		for layer in self.model[i_layer:]:
			y = layer.forward_propagate(y)
		return y.ravel()

	def back_propagate(self, de_dy):
		# Iterate through the layers backwards and propagate from end to start
		for i, layer in enumerate(self.model[::-1]):
			de_dx = layer.back_propagate(de_dy)
			de_dy = de_dx

	def report(self):
		n_bins = int(len(self.errors) // self.report_bin_size)
		average_errors = []

		for b in range(n_bins):
			average_errors.append(
				np.mean(self.errors[
							b * self.report_bin_size :
							(b+1) * self.report_bin_size]
				)
			)

		log_errors = np.log10(np.array(average_errors) + 1e-10)

		# plot vars
		y_min = np.minimum(self.report_min, np.min(log_errors))
		y_max = np.maximum(self.report_max, np.max(log_errors))
		fig = plt.figure()
		ax = plt.gca()

		ax.plot(log_errors)
		ax.set_ylim(y_min, y_max)
		ax.set_ylabel('Errors (logarithm)')
		ax.set_xlabel('Iterations')
		ax.grid()
		fig.savefig(os.path.join(self.report_path, self.report_file))
		plt.close()














