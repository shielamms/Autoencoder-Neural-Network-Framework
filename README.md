## Autoencoder Neural Network Framework

This repository contains my solution to the exercises in the "312. Build a Neural Network Framework" module of the [End-to-End Machine Learning website by Brandon Rohrer, PhD](https://end-to-end-machine-learning.teachable.com/). If you would like to learn how to build a neural network framework from scratch, head on to his course and get coding! I had so much fun doing the exercises so I highly recommend taking the course!

---

### What is an autoencoder?

An **Autoencoder** is a neural network that learns the features of its input data so that it can be used to reconstruct that data. This means that the output of the neural network aims to be as close to its input as it can. After the model has learned the optimal weights that minimise the error between the output and the input, the output can be discarded and only the resulting encoder model is used for further purposes, like classifying images.

The test data for this framework is the Nordic Runes dataset (also provided by Brandon Rohrer in [this repository](https://github.com/brohrer/nn_framework/blob/master/elder_futhark.py)). It is a set of 7x7 pixel images of the runes of the Elder Futhark, an old Germanic alphabet. You can visualise these images by running `python data/elder_futhark.py`.

In `main.py`, you'll find two data loaders: The first (`data_loader`) contains simple 2x2-pixel and 3x3-pixel image datasets. The second one (`data_loader_nordic_runes`) contains the 7x7-pixel image dataset. I specified only 1 hidden layer with 41 nodes for this, but you can practically try any number of hidden layers and nodes.

---

### How to run the code

I used a Python 3.8 environment. I recommend creating a virtual environment on your local machine and installing the required libraries inside that virtual environment before running the code.

1. Checkout this repository

2. Create a virtual environment in the root directory of this project. If you have pyenv, it would be something like:

```
pyenv local 3.8.8
python -m virtualenv venv
source venv/bin/activate
```

3. Install the requirements

```
python -m pip install -r requirements.txt
```

4. Run:

```
python main.py
```

During training, a directory called `nn_images` will be created. This will contain snapshots of the neural network at particular i-th iterations. The `reports` directory will contain an image named `nn_performance_report.png`, which is a pyplot of the error history for the entire training duration. A decreasing error over time would be a good indicator of improving performance of the neural network on the dataset.
