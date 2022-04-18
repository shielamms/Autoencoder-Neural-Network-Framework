import data_loader as dat
import data_loader_nordic_runes as runes_data
from autoencoder_viz import Printer as NN_Visualizer
from nn_framework import (
    framework,
    activation,
    error_function
)
from nn_framework.layer import Dense, Difference, RangeNormalization
from nn_framework.optimizers import SGD, Momentum, Adam
from nn_framework.regularization import L1, L2, Limit

# Load the data
training_set, evaluation_set = runes_data.get_data_sets()
sample = next(training_set())
print('Shape of input data:', sample.shape)

# Build the structure of the nueral network, where the number of input nodes is
# the size of an input image, and the number of output nodes is the same.
# Set an arbitary number of hidden layers each with an arbitrary number of nodes
n_pixels = sample.shape[0] * sample.shape[1]
n_outputs = n_pixels
n_hidden_nodes = [41]  # for nordic runes data

# specify only the number of hidden and output nodes, infer the input nodes
n_nodes = n_hidden_nodes + [n_outputs]
model = []
dropout_rates = [0.2, 0.5] # 0.2 on input layer, 0.5 on hidden layer

# The first layer is a Normalization layer
model.append(RangeNormalization(training_set)) # infer the input range

print('Number of layers: ', len(n_nodes))

# The middle layers are fully connected layers
for i_layer in range(len(n_nodes)):
    # infer the number of input nodes based on the given number of output nodes
    new_layer = Dense(n_nodes[i_layer],
                      activation.Tanh,
                      previous_layer=model[-1],
                      optimizer=Momentum(),
                      # dropout_rate=dropout_rates[i_layer]
    )
    new_layer.add_regularizer(L1())
    # new_layer.add_regularizer(L2())
    new_layer.add_regularizer(Limit(4.0))
    model.append(new_layer)

# The last layer is a Difference layer (difference between last layer and
# first layer)
model.append(Difference(model[-1], model[0]))

autoencoder = framework.ANN(
    model=model,
    error_func=error_function.SqrErr,
    visualizer=NN_Visualizer(input_shape=sample.shape)
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
