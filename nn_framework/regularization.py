import numpy as np


# LASSO Regularization
class L1():
    def __init__(self, regularization_amount=1e-2):
        """
        regularization_amount: a constant for the magnitude of the regularizer
        """
        self.regularization_amount = regularization_amount

    def update(self, layer):
        weights = layer.weights
        delta = self.regularization_amount * layer.learning_rate

        # Update the weights such that very small values close to 0 are 0,
        # values less than 0 increase by delta,
        # and values more than 0 decrease by delta
        weights[np.where(np.abs(weights) < delta)] = 0
        weights[np.where(weights < 0)] += delta
        weights[np.where(weights > 0)] -= delta
        return weights


# Ridge Regularization
class L2():
    def __init__(self, regularization_amount=1e-2):
        """
        regularization_amount: a constant for the magnitude of the regularizer
        """
        self.regularization_amount = regularization_amount


    def update(self, layer):
        # Update the weights such that very large values are scaled down
        # and subtracted from the old weights
        weights = layer.weights
        adjustment = (
            2 * layer.learning_rate * self.regularization_amount * weights
        )
        return weights - adjustment


# Custom Regularizer
class Limit():
    def __init__(self, weight_limit=1):
        self.weight_limit = weight_limit

    def update(self, layer):
        # Update the weights such that values larger than the positive limit
        # is pulled back to the positive limit, while values less than the
        # negative limit is pulled back to the negative limit. All other
        # values in between remain the same.
        weights = layer.weights
        weights[np.where(weights > self.weight_limit)] = self.weight_limit
        weights[np.where(weights < -self.weight_limit)] = -self.weight_limit
        return weights
