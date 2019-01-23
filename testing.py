import numpy as np
import struct
from mnist import MNIST
import random
import matplotlib.pyplot as plt

mndata = MNIST('datasets')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()




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
