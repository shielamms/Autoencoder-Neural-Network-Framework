import numpy as np


class Glorot():
    @staticmethod
    def __str__():
        return "Glorot"

    @staticmethod
    def initialize(n_rows, n_cols):
        return np.random.normal(
            scale = np.sqrt(2 / (n_rows + n_cols)), # standard deviation
            size = (n_rows, n_cols),
        )


class He():
    @staticmethod
    def __str__():
        return "He"

    @staticmethod
    def initialize(n_rows, n_cols):
        # This initializer is a function of the number of inputs
        return np.random.uniform(
            low = np.sqrt(6 / n_rows),
            high = np.sqrt(6 / n_rows),
            size = (n_rows, n_cols),
        )
