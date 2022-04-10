from data import elder_futhark as ef
import matplotlib.pyplot as plt
import numpy as np


def get_data_sets():
    data = list(ef.runes.values())

    for example in data:
        plt.figure()
        plt.imshow(example, cmap='bone')
        plt.show()

    def training_set():
        choice = np.random.choice(len(data))
        yield data[choice]

    def evaluation_set():
        choice = np.random.choice(len(data))
        yield data[choice]

    return training_set, evaluation_set
