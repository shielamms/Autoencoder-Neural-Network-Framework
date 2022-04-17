import data_loader as dat
import data_loader_nordic_runes as runes_data
from autoencoder_viz import Printer as NN_Visualizer
from nn_framework import (
    framework,
    layer,
    activation,
    error_function
)
from nn_framework.regularization import L1, L2, Limit

# Load the data
training_set, evaluation_set = runes_data.get_data_sets()
sample = next(training_set())
print('Shape of input data:', sample.shape)

# Build the structure of the nueral network, where the number of input nodes is
# the size of an input image, and the number of output nodes is the same.
# Set an arbitary number of hidden layers each with an arbitrary number of nodes
n_pixels = sample.shape[0] * sample.shape[1]
n_inputs = n_outputs = n_pixels
n_hidden_nodes = [41]  # for nordic runes data
# n_hidden_nodes = [24]
n_nodes = [n_inputs] + n_hidden_nodes + [n_outputs]
model = []

input_pixel_range = (0, 1)  # pixel values are from 0 to 1
learning_rate = 0.001

# Build the model without the output layer yet because it is not a Dense layer.
# The number of outputs of one layer is the number of nodes of the next layer.
for i_layer in range(len(n_nodes) - 1):
    new_layer = layer.Dense(n_nodes[i_layer],
                            n_nodes[i_layer+1],
                            activation.Tanh,
                            learning_rate=learning_rate)
    new_layer.add_regularizer(L1())
    new_layer.add_regularizer(L2())
    new_layer.add_regularizer(Limit(1.0))
    model.append(new_layer)

autoencoder = framework.ANN(
    model=model,
    input_pixel_range=input_pixel_range,
    error_func=error_function.AbsErr,
    visualizer=NN_Visualizer(input_shape=sample.shape)
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
