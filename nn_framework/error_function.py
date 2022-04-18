import numpy as np

class SqrErr():
    @staticmethod
    def calc(x):
        return np.mean(x**2)

    @staticmethod
    def calc_derivative(x):
        return 2 * x

    @staticmethod
    def __str__():
        return 'Mean Squared Error'


class AbsErr():
    @staticmethod
    def calc(x):
        return np.mean(np.abs(x))

    @staticmethod
    def calc_derivative(x):
        return np.sign(x)

    @staticmethod
    def __str__():
        return 'Mean Absolute Error'
