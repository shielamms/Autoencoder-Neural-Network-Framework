import numpy as np
from .initializers import Glorot
from .optimizers import SGD


class GenericLayer():
    def __init__(self, previous_layer):
        self.previous_layer = previous_layer
        # The number of nodes in this layer is the same as the number of
        # outputs of the previous layer
        self.size = self.previous_layer.y.size
        self.reset()

    def reset(self):
        # Reset all inputs, outputs, and gradients of this layer
        # for the next iteration
        self.x = self.y = self.de_dx = self.de_dy = np.zeros((1, self.size))

    def forward_propagate(self, **kwargs):
        self.x += previous_layer.y
        self.y = self.x

    def back_propagate(self):
        self.de_dx = self.de_dy
        self.previous_layer.de_dy += self.de_dx


class Dense(GenericLayer):
    def __init__(self,
                 n_outputs,
                 activation,
                 initializer=None,
                 previous_layer=None,
                 optimizer=None,
                 dropout_rate=0,
                 ):
        self.type = 'Dense'
        self.previous_layer = previous_layer
        self.n_outputs = int(n_outputs)
        self.m_inputs = self.previous_layer.y.size # infer number of input nodes
        self.activation = activation
        self.initializer = initializer
        self.dropout_rate = dropout_rate

        if not initializer:
            self.initializer = Glorot()
        else:
            self.initializer = initializer

        if not optimizer:
            self.optimizer = SGD(learning_rate=0.001)
        else:
            self.optimizer = optimizer

        # Use an Initializer object to set the initial weights;
        # Add the bias node to the initializer inputs
        self.weights = initializer.initialize(self.m_inputs+1, self.n_outputs)

        self.regularizers = []
        self.reset()

    def reset(self):
        self.x = self.de_dx = np.zeros((1, self.m_inputs))
        self.y = self.de_dy = np.zeros((1, self.n_outputs))

    def add_regularizer(self, new_regularizer):
        self.regularizers.append(new_regularizer)

    def forward_propagate(self, is_evaluation=False, **kwargs):
        if self.previous_layer is not None:
            self.x += self.previous_layer.y

        if is_evaluation:
            dropout_rate = 0
        else:
            dropout_rate = self.dropout_rate

        # Determine random nodes to drop out
        self.i_dropout = np.zeros(self.x.size, dtype=bool)
        random_dropout_idx = (
                np.where(
                    np.random.uniform(size=self.x.size) < dropout_rate
        ))
        self.i_dropout[random_dropout_idx] = True
        # All dropout nodes get a value of 0, while remaining nodes will
        # increase by a factor to compensate for the dropped nodes
        self.x[:, self.i_dropout] = 0
        self.x[:, np.logical_not(self.i_dropout)] *= 1 / (1 - dropout_rate)

        # Include the bias node in the forward propagation
        bias = np.ones((1,1))
        x_w_bias = np.concatenate((self.x, bias), axis=1)
        v = x_w_bias @ self.weights # dot product of inputs and weights
        self.y = self.activation.calc(v)

    def back_propagate(self):
        bias = np.ones((1, 1))
        x_w_bias = np.concatenate((self.x, bias), axis=1)

        # From de_dy, get de_dx by calculating the dot product of
        # de_dy and the transpose of the weights matrix.
        dy_dv = self.activation.calc_derivative(self.y)
        dv_dw = x_w_bias.transpose()
        dv_dx = self.weights.transpose()

        dy_dw = dv_dw @ dy_dv
        self.de_dw = self.de_dy * dy_dw

        # regularize the weights:
        # L2 - penalize large weights
        # L1 - gradually regularize towards 0
        # Use L1 and L2 before optimization because they operate on the gradient
        for regularizer in self.regularizers:
            regularizer.pre_optim_update(self)

        self.optimizer.update(self)

        # Use the Limit regularizer after optimization since it operates
        # on the weights
        for regularizer in self.regularizers:
            regularizer.post_optim_update(self)

        self.de_dx = (self.de_dy * dy_dv) @ dv_dx

        # Do not backpropagate to the dropped out nodes
        de_dx_no_bias = self.de_dx[:, :-1]
        de_dx_no_bias[:, self.i_dropout] = 0

        # Do not backpropagate values from the bias node
        self.previous_layer.de_dy += de_dx_no_bias


class RangeNormalization(GenericLayer):
    def __init__(self, training_data, previous_layer=None):
        self.type = 'RangeNormalization'
        self.previous_layer = previous_layer
        self.range_min = 1e10
        self.range_max = -1e10
        n_range_test = 100

        # Get samples from training_data and determine the min and max values
        for _ in range(n_range_test):
            sample = next(training_data())

            if np.min(sample) < self.range_min:
                self.range_min = np.min(sample)
            if np.max(sample) > self.range_max:
                self.range_max = np.max(sample)

        self.scale_factor = self.range_max - self.range_min
        self.size = sample.size
        self.reset()

    def forward_propagate(self, **kwargs):
        if self.previous_layer is not None:
            self.x += self.previous_layer.y

        # Scale the values to values between -0.5 and 0.5
        self.y = (self.x - self.range_min) / self.scale_factor - 0.5

    def back_propagate(self):
        # Scale the gradient
        self.de_dx = self.de_dy / self.scale_factor

        if self.previous_layer is not None:
            self.previous_layer.de_dy += self.de_dx

    def denormalize(self, transformed_values):
        pass
        # (transformed_values + 0.5) * (self.scale_factor + self.range_min)


class Difference(GenericLayer):
    # This calculates the difference between the output of the first layer
    # and the output of the previous layer
    def __init__(self, previous_layer, diff_layer):
        self.type = 'Difference'
        self.previous_layer = previous_layer
        self.diff_layer = diff_layer

        # All layers involved in this operation should have the same size
        assert self.previous_layer.y.size == self.diff_layer.y.size

        self.size = self.previous_layer.y.size

    def forward_propagate(self, **kwargs):
        self.y = self.previous_layer.y - self.diff_layer.y

    def back_propagate(self):
        self.previous_layer.de_dy += self.de_dy
        self.diff_layer.de_dy -= self.de_dy
