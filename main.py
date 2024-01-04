from keras.datasets import mnist
#!/usr/bin/env python3

from tensorflow.keras.utils import to_categorical
import numpy as np

a = 1


def sigmoid(x):  # Activation function
    return 1 / (1 + np.exp(-x))


def forward_propagation(X):
    Z1 = np.dot(X, weights_input_hidden) + bias_hidden
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, weights_hidden_output) + bias_output
    A2 = sigmoid(Z2)
    return A1, A2


def prc():  # main code
    # Input data

    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train shape: (60000, 28, 28)
    # y_train shape: (60000,)
    # x_test shape: (10000, 28, 28)
    # y_test shape: (10000,)

    # Normalize the input data
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(type(x_train), type(x_test))

    # Model preparation

    np.random.seed(42)  # for reproducibility

    input_size = 784  # number of input neurons (MNIST images are 28x28)
    hidden_size = 64  # number of neurons in the hidden layer
    output_size = 10  # number of output neurons (digits 0-9)

    # Initialize weights and biases
    weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
    weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
    bias_hidden = np.zeros((1, hidden_size))
    bias_output = np.zeros((1, output_size))

    print(weights_input_hidden.shape, weights_hidden_output.shape,
          bias_hidden.shape, bias_output.shape)


if __name__ == "__main__":
    # main code that calls some_function
    prc()
