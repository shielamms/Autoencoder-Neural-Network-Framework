from autoencoder_viz import Printer as NN_Visualizer
import data_loader as dat
import data_loader_nordic_runes as runes_data
from nn_framework.activation import Tanh
from nn_framework.error_function import SqrErr
from nn_framework.framework import ANN
from nn_framework.layer import Dense, Difference, RangeNormalization
from nn_framework.optimizers import Momentum
from nn_framework.regularization import L1, Limit

training_set, evaluation_set = runes_data.get_data_sets()
sample = next(training_set())
n_pixels = sample.shape[0] * sample.shape[1]
n_hidden_nodes = [41]
n_nodes = n_hidden_nodes + [n_pixels]
model = []

model.append(RangeNormalization(training_set)) # infer the input range

for i_layer in range(len(n_nodes)):
    new_layer = Dense(n_nodes[i_layer],
                      activation=Tanh,
                      previous_layer=model[-1],
                      optimizer=Momentum(),
    )
    new_layer.add_regularizer(L1())
    new_layer.add_regularizer(Limit(4.0))
    model.append(new_layer)

model.append(Difference(model[-1], model[0]))

autoencoder = ANN(
    model=model,
    error_func=SqrErr,
    visualizer=NN_Visualizer(input_shape=sample.shape)
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
