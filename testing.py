import numpy as np
import struct
from mnist import MNIST
import random
import matplotlib.pyplot as plt


'''
mndata = MNIST('datasets')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
'''











'''
# simple neural net https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
from numpy import exp, array, random, dot
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1
for iteration in xrange(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights))))
'''







'''
# more beautiful

from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        #random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        #self.synaptic_weights = 2 * random.random((3, 1)) - 1
        self.synaptic_weights = array([[0.1, -0.8, 0.5]])

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 1)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    # Test the neural network with a new situation.
    print "Considering new situation [1, 0, 0] -> ?: "
    print neural_network.think(array([1, 0, 0]))

'''

'''
#confirming type and looked up display method
#''uses threshold of 200 when in fact image has variable greyness

print type(train_images[1])
print train_images[1]
print train_labels[1]
print(mndata.display(train_images[1]))

# using negative, so writing is white on black background
np_image = np.array(train_images[1], dtype='float')
pixels = np_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
'''




''' confirming python-mnist package works!
print len(test_images)
print len(test_labels)
print len(train_images)
print len(train_labels)

index = random.randrange(0, len(test_images))  # choose an index ;-)
print(mndata.display(test_images[index]))
'''




''' silly first attempts at reading files on my own, somehow kept getting
17301504
1625948160
as first two integers, but second at least should have been 10k or 60k I thought...
decided to find mnist conversion help...


fn_test_labels = 'datasets/t10k-labels-idx1-ubyte'
fn_test_images = 'datasets/t10k-images-idx3-ubyte'
fn_train_labels = 'datasets/train-labels-idx1-ubyte'
fn_train_images = 'datasets/train-images-idx3-ubyte'

f_test_labels = open(fn_test_labels, 'r')
f_test_images = open(fn_test_images, 'r')
f_train_labels = open(fn_train_labels, 'r')
f_train_images = open(fn_train_images, 'r')


TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.



print str(np.frombuffer(f_train_labels.read(), np.uint32, offset=4)[0:10])


print struct.unpack('I', f_train_labels.read(4))[0]
print struct.unpack('I', f_train_labels.read(4))[0]
print struct.unpack('b', f_train_labels.read(1))[0]
print struct.unpack('b', f_train_labels.read(1))[0]
print struct.unpack('b', f_train_labels.read(1))[0]

test_labels = np.frombuffer(f_test_labels.read(), np.uint8, offset=0)[10:-1]
train_labels = np.frombuffer(f_train_labels.read(), np.uint8, offset=0)[10:-1]

print str(test_labels[0:100])
print str(train_labels[0:100])

test_images = np.frombuffer(f_test_images.read(), np.uint8, offset=0)

print str(test_images[0:600])'''
