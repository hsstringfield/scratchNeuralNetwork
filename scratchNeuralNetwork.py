# used for arrays
import numpy as np

# used for data set
import nnfs
from nnfs.datasets import spiral_data


# instead of seed, used to recreate results
nnfs.init()


# creation of layers
class Layer_Dense:
    # initilize random weights and zeros for biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward through layer
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation function
class Activation_ReLU:
    # forward through activation function
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# softmax activation function
class Activation_Softmax:
    # forward through activation function, including normalizing values
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# calculate with loss class
class Loss:
    # calculate loss for batch
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


# loss with categorical crossentropy
class Loss_CategoricalCrossentropy(Loss):
    # forward method for categorical crossentropy
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # is scalar?
        if(len(y_true.shape) == 1):
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # is one hot encoded?
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
        # find loss
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood


# given data from nnfs
X, y = spiral_data(100, 3)


# set up layers with activation functions
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()


# forward through layers and activation functions
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)


# loss function with categorical crossentropy
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)


# print results
print(activation2.output[:5])

print("Loss:", loss)