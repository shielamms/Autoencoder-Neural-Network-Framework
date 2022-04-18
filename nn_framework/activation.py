import numpy as np

class Tanh():

    @staticmethod
    def calc(value):
        return np.tanh(value)

    @staticmethod
    def calc_derivative(value):
        return 1 - np.tanh(value)**2

    @staticmethod
    def __str__():
        return 'Hyperbolic Tangent (Tanh)'


class Logistic():

    @staticmethod
    def calc(value):
        return 1 / (1 + np.exp(-value))

    @staticmethod
    def calc_derivative(value):
        return calc(value) * (1 - calc(value))

    @staticmethod
    def __str__():
        return 'Logistic'


class ReLU():

    @staticmethod
    def calc(value):
        return np.maximum(0, value)

    @staticmethod
    def calc_derivative(value):
        if value > 0:
            return 1
        else:
            return 0

    @staticmethod
    def __str__():
        return 'ReLU'
