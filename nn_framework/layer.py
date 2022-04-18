import numpy as np

class Dense():
    def __init__(self,
                 m_inputs,
                 n_outputs,
                 activation,
                 learning_rate=0.001,
                 dropout_rate=0):
        self.m_inputs = int(m_inputs)
        self.n_outputs = int(n_outputs)

        # set initial random weights between -1 and 1
        self.weights = (np.random.sample(
                            size=(self.m_inputs+1, self.n_outputs))
                        * 2 - 1)
        self.x = np.zeros((1, self.m_inputs+1)) # +1 for the bias
        self.y = np.zeros((1, self.n_outputs))
        self.activation = activation
        self.learning_rate = learning_rate
        self.regularizers = []
        self.dropout_rate = dropout_rate
        self.i_dropout = None

    def add_regularizer(self, new_regularizer):
        self.regularizers.append(new_regularizer)

    def forward_propagate(self, inputs, is_evaluation=False):
        if is_evaluation:
            self.dropout_rate = 0

        # Determine random nodes to drop out
        self.i_dropout = np.zeros(self.x.size, dtype=bool)
        random_dropout_idx = (
                np.where(
                    np.random.uniform(size=self.x.size) < self.dropout_rate
        ))
        self.i_dropout[random_dropout_idx] = True
        # All dropout nodes get a value of 0, while remaining nodes will
        # increase by a factor to compensate for the dropped nodes
        self.x[:, self.i_dropout] = 0
        self.x[:, np.logical_not(self.i_dropout)] *= (
                                1 / (1 - self.dropout_rate))


        # Include the bias node in the forward propagation
        bias = np.ones((1,1))
        self.x = np.concatenate((inputs, bias), axis=1)
        v = self.x @ self.weights # dot product of inputs and weights
        self.y = self.activation.calc(v)
        return self.y

    def back_propagate(self, de_dy):
        # From de_dy, get de_dx by calculating the dot product of
        # de_dy and the transpose of the weights matrix.
        dy_dv = self.activation.calc_derivative(self.y)
        dy_dw = self.x.transpose() @ dy_dv
        de_dw = de_dy * dy_dw

        # adjust the weights to minimise the error
        self.weights -= de_dw * self.learning_rate

        # regulirize the weights:
        # L2 - penalize large weights
        # L1 - gradually regularize towards 0
        for regularizer in self.regularizers:
            self.weights = regularizer.update(self)

        de_dx = (de_dy * dy_dv) @ self.weights.transpose()
        # Do not backpropagate to the dropped out nodes
        de_dx[:, self.i_dropout] = 0
        return de_dx[:, :-1]









