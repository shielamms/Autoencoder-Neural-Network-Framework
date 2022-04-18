import numpy as np


class GenericRegularizer():
    def __init__(self):
        pass

    def pre_optim_update(self, layer):
        pass

    def post_optim_update(self, layer):
        pass


# LASSO Regularization
class L1(GenericRegularizer):
    def __init__(self, regularization_amount=1e-2):
        """
        regularization_amount: a constant for the magnitude of the regularizer
        """
        self.regularization_amount = regularization_amount

    def pre_optim_update(self, layer):
        layer.de_dw += np.sign(layer.weights) * self.regularization_amount


# Ridge Regularization
class L2(GenericRegularizer):
    def __init__(self, regularization_amount=1e-2):
        """
        regularization_amount: a constant for the magnitude of the regularizer
        """
        self.regularization_amount = regularization_amount

    def pre_optim_update(self, layer):
        layer.de_dw += 2 * layer.weights * regularization_amount


# Custom Regularizer
class Limit(GenericRegularizer):
    def __init__(self, weight_limit=1):
        self.weight_limit = weight_limit

    def post_optim_update(self, layer):
        # Update the weights such that values larger than the positive limit
        # is pulled back to the positive limit, while values less than the
        # negative limit is pulled back to the negative limit. All other
        # values in between remain the same.
        layer.weights = np.minimum(self.weight_limit, layer.weights)
        layer.weights = np.maximum(-self.weight_limit, layer.weights)
