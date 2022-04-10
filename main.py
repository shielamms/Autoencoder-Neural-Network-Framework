import data_loader as dat
import data_loader_nordic_runes as runes_data
from autoencoder_viz import Printer as NN_Visualizer
from nn_framework import (
    framework,
    layer,
    activation,
    error_function
)

# Load the data
training_set, evaluation_set = runes_data.get_data_sets()
sample = next(training_set())
print('Shape of input data:', sample.shape)

# Build the structure of the nueral network, where the number of input nodes is
# the size of an input image, and the number of output nodes is the same.
# Set an arbitary number of hidden layers each with an arbitrary number of nodes
n_pixels = sample.shape[0] * sample.shape[1]
n_inputs = n_outputs = n_pixels
# n_hidden_nodes = [5, 7, 4]
# n_hidden_nodes = [9]
n_hidden_nodes = [41]  # for nordic runes data
n_nodes = [n_inputs] + n_hidden_nodes + [n_outputs]
model = []

input_pixel_range = (0, 1)  # pixel values are from 0 to 1
learning_rate = 0.001

# Build the model without the output layer yet because it is not a Dense layer.
# The number of outputs of one layer is the number of nodes of the next layer.
for i_layer in range(len(n_nodes) - 1):
    model.append(
        layer.Dense(n_nodes[i_layer],
                    n_nodes[i_layer+1],
                    activation.Tanh,
                    # activation.Logistic    # alternative activation function
                    learning_rate=learning_rate,
        )
    )

autoencoder = framework.ANN(
    model=model,
    input_pixel_range=input_pixel_range,
    error_func=error_function.AbsErr,
    visualizer=NN_Visualizer(input_shape=sample.shape),
    # error_func=error_function.SqrErr    # alternative error function
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
