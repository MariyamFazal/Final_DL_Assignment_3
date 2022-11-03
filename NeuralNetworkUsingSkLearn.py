import tensorflow as tf
from keras.utils import np_utils

# loadin CIFAR10 images dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# training dataset
x_train = x_train.reshape(x_train.shape[0], 32*32*3)
x_train = x_train.astype('float32')
x_train /= 255

# encoding output
y_train = np_utils.to_categorical(y_train)

# testing dataset
x_test = x_test.reshape(x_test.shape[0], 32*32*3)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

from sklearn.neural_network import MLPClassifier
NeuralNetwork = MLPClassifier(hidden_layer_sizes = (20,20,20), max_iter=1000, learning_rate_init=0.05,activation = 'relu')

# training on 1000 samples
NeuralNetwork.fit(x_train[0:1000], y_train[0:1000])

# test on 3 samples
y_predicted = NeuralNetwork.predict(x_test[0:3])
print("\n")
print("Predicted Values : ")
print(y_predicted, end="\n")
print("True Values : ")
print(y_test[0:3])

