import numpy as np
import matplotlib.pyplot as plt

def get_data_sets():
  examples_2x2 = [
      np.array([
            [0, 1],
            [1, 1]
      ]),
      np.array([
            [1, 1],
            [0, 0]
      ]),
      np.array([
            [0, 0],
            [1, 1]
      ]),
      np.array([
            [1, 1],
            [0, 1]
      ]),
      np.array([
            [1, 1],
            [1, 1]
      ]),
      np.array([
            [1, 1],
            [1, 0]
      ]),
      np.array([
            [0, 0],
            [0, 1]
      ]),
      np.array([
            [0, 1],
            [1, 0]
      ]),
      np.array([
            [0, 0],
            [0, 0]
      ]),
      np.array([
            [1, 0],
            [0, 0]
      ]),
  ]

  examples_3x3 = [
  	np.array([
            [0, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
      ]),
  	np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
      ]),
  	np.array([
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
      ]),
  	np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
      ]),
  	np.array([
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1],
      ]),
  	np.array([
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
      ]),
  	np.array([
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
      ]),
  	np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
      ]),
  	np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
      ]),
  	np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
      ]),
  	np.array([
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
      ]),
  	np.array([
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
      ]),
  ]

  examples = examples_3x3

  for example in examples:
  	plt.figure()
  	plt.imshow(example, cmap='bone')
  	plt.show()

  def training_set():
    choice = np.random.choice(len(examples))
    yield examples[choice]

  def evaluation_set():
    choice = np.random.choice(len(examples))
    yield examples[choice]

  return training_set, evaluation_set

if __name__ == '__main__':
	get_data_sets()
