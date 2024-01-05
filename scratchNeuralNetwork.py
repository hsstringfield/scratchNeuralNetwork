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
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # backward through layer
    def backward(self, dvalues):
        # gradients for inputs, weights, and biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU activation function
class Activation_ReLU:
    # forward through activation function
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # backward through activation function
    def backward(self, dvalues):
        # copy of inputs
        self.dinputs = dvalues.copy()
        # gradient for negative inputs
        self.dinputs[self.inputs < 0] = 0


# softmax activation function
class Activation_Softmax:
    # forward through activation function, including normalizing values
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # backward through activation function
    def backward(self, dvalues):
        # uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # calculate gradients and add to dinputs
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


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

    # backward method for categorical crossentropy
    def backward(self, dvalues, y_true):
        # determine shape using first sample
        samples = len(dvalues)
        labels = len(dvalues[0])

        # sparse turn into one-hot
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate and normalize gradient
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


# combined softmax and categorical ce for speed on backward
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # forward pass through using previously implemented
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    # backward pass optimized, compare to previous
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


'''
# given practice outputs
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

# given class targets
class_targets = np.array([0, 1, 1])

# example backpropagation optimized
softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs

# example backpropagation not optimized
activation = Activation_Softmax()
activation.output = softmax_outputs
loss = Loss_CategoricalCrossentropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

# print results
print('Gradients: combined loss and activation:')
print(dvalues1)
print('Gradients: combined loss and activation:')
print(dvalues2)
'''

# given data from nnfs
X, y = spiral_data(100, 3)


# set up layers with activation functions
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
# changed to use optimized function
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()


# forward through layers and activation functions
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)


# replaced with optimized function
loss = loss_activation.forward(dense2.output, y)

# print output of some samples
print(loss_activation.output[:5])
print('loss:', loss)


# calculate accuracy from output
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis = 1)
accuracy = np.mean(predictions == y)

# print accuracy
print('Accuracy:', accuracy)


# backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)


# print resulting gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)