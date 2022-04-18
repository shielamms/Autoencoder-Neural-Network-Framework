import numpy as np


class GenericOptimizer():
    def __init__(self, **kwargs):
        # The properties of this class here were copied from
        # https://gitlab.com/brohrer/cottonwood/-/blob/v5/core/optimizers.py
        default_adam_beta_1 = .9
        default_adam_beta_2 = .999
        default_epsilon = 1e-8
        default_learning_rate = 1e-3
        default_minibatch_size = 1  # technically means no mini-batch
        default_momentum_amount = .9
        default_scaling_factor = 1e-3

        self.adam_beta_1 = kwargs.get(
            "adam_beta_1", default_adam_beta_1)
        self.adam_beta_2 = kwargs.get(
            "adam_beta_2", default_adam_beta_2)
        self.epsilon = kwargs.get(
            "epsilon", default_epsilon)
        self.learning_rate = kwargs.get(
            "learning_rate ", default_learning_rate)
        self.minibatch_size = kwargs.get(
            "minibatch_size ", default_minibatch_size)
        self.momentum_amount = kwargs.get(
            "momentum_amount", default_momentum_amount)
        self.scaling_factor = kwargs.get(
            "scaling_factor", default_scaling_factor)

        self.i_minibatch = 0
        self.de_dw_total = None

    def update_minibatch(self, layer):
        if self.de_dw_total is None:
            self.de_dw_total = np.zeros(layer.de_dw.shape)
        self.de_dw_total += layer.de_dw
        self.i_minibatch += 1

        de_dw_batch = None
        if self.i_minibatch >= self.minibatch_size:
            de_dw_batch = self.de_dw_total / self.minibatch_size
            self.de_dw_total = None
            self.i_minibatch = 0

        return de_dw_batch


class SGD(GenericOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, layer):
        # Get the average of a batch of gradients
        de_dw_batch = self.update_minibatch(layer)

        if not de_dw_batch:
            return

        layer.weights -= de_dw_batch * self.learning_rate


class Momentum(GenericOptimizer):
    # Momentum takes bigger and bigger steps in a particular direction
    # by carrying over a fraction of previous weight adjustments
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_adjustment = None

    def update(self, layer):
        de_dw_batch = self.update_minibatch(layer)

        if de_dw_batch is None:
            return

        if self.previous_adjustment is None:
            self.previous_adjustment = np.zeros(layer.weights.shape)

        new_adjustment = (
            self.previous_adjustment * self.momentum_amount
            + de_dw_batch * self.learning_rate
        )
        layer.weights -= new_adjustment
        self.previous_adjustment = new_adjustment


class Adam(GenericOptimizer):
    # Adam TBC, not entirely clear how it works :/
    def __init__(self):
        super().__init__(**kwargs)

    def update(self, layer):
        pass
