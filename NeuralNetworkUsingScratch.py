from network import Network
from fc_layer import FCLayer
from activation_layer import activationlayer
from activations import tanh, tanh_prime
from losses import Mean_Squared_Error, Mean_Squared_Error_p
import tensorflow as tf
from keras.utils import np_utils

# loading CIFAR10 images dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# training dataset
x_train = x_train.reshape(x_train.shape[0], 1 , 32*32*3)
x_train = x_train.astype('float32')
x_train /= 255

# encoding output
y_train = np_utils.to_categorical(y_train)

# testing dataset
x_test = x_test.reshape(x_test.shape[0], 1, 32*32*3)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Network
net = Network()
net.add(FCLayer(32*32*3, 100))                # input_shape=(1, 32*32*3)    ;   output_shape=(1, 100)
net.add(activationlayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(activationlayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(activationlayer(tanh, tanh_prime))

# training on 100 samples

net.use(Mean_Squared_Error, Mean_Squared_Error_p)
net.fit(x_train[0:100], y_train[0:100], iterations=60, learning_rate=0.1)

# test on 3 samples
y_predict = net.predict(x_test[0:3])
print("\n")
print("Predicted Values : ")
print(y_predict, end="\n")
print("True Values : ")
print(y_test[0:3])

