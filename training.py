import numpy as np

# neural networks library for use on top of tensorflow
from keras import layers

# Input() is used to instantiate a Keras tensor.
# A Keras tensor is a tensor object from the underlying backend (Theano or TensorFlow),
# which we augment with certain attributes that allow us
# to build a Keras model just by knowing the inputs and outputs of the model

# Dense - regular densely-connected NN layer

# Activation - Activation function to be used as a layer or activation argument in other layers

# ZeroPadding2D - This layer can add rows and columns of zeros at the top, bottom, left and right side
# of an image tensor

# BatchNormalization - Normalize the activations of the previous layer at each batch,
# i.e. applies a transformation that maintains the mean activation close to 0 and
# the activation standard deviation close to 1

# Flatten - flatten the input

# Conv2D - 2D convolution layer (e.g. spatial convolution over images)
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

# AveragePooling2D -  Its function is to progressively reduce the spatial size of the representation to
# reduce the amount of parameters and computation in the network, and hence to also control overfitting

# MaxPooling2D - Max pooling is done by applying a max filter to (usually)
# non-overlapping subregions of the initial representation.

# Dropout - ignoring units (i.e. neurons) during the training phase of certain set of neurons which is chosen at random
# GlobalMaxPooling2D - MaxPooling2D with pool size = input size
# GlobalAveragePooling2D - AveragePooling2D with pool size = input size
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

# keras utility functions
from keras.utils import np_utils
# Sequential - linear stack of layers
from keras.models import Sequential
# callback is a set of functions to be applied at given stages of the training procedure
# ModelCheckpoint save model after every epoch
from keras.callbacks import ModelCheckpoint
import pandas as pd

# pandas - data manipulation and analysis
import keras.backend as K


# image_x, image_y - dimensions of image
def keras_model(image_x, image_y):
    # number of classes among which to classify
    num_of_classes = 12
    # set up a sequential model and add cnn layers
    model = Sequential()
    # Parameters -
    # filters - number of outputs from conv2d layer
    # kernel size
    # image shape
    # activation function - relu
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    # units - number of neurons in the layer
    model.add(Dense(1024, activation='relu'))
    # 60% dropout rate
    model.add(Dropout(0.6))
    # final softmax layer to classify image
    model.add(Dense(num_of_classes, activation='softmax'))

    # configure the model for training
    # optimizer - adam
    # loss - categorical crossentropy
    # metrics - accuracy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "emojinator.h5"
    # save model in filepath
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list


def main():
    # read csv for creating train and test data
    data = pd.read_csv("train_foo.csv")
    dataset = np.array(data)
    np.random.shuffle(dataset)
    X = dataset
    Y = dataset

    X = X[:, 1:2501]
    Y = Y[:, 0]

    X_train = X[0:12000, :]
    X_train = X_train / 255.
    X_test = X[12000:13201, :]
    X_test = X_test / 255.

    # Reshape
    Y = Y.reshape(Y.shape[0], 1)
    Y_train = Y[0:12000, :]
    Y_train = Y_train.T
    Y_test = Y[12000:13201, :]
    Y_test = Y_test.T

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    image_x = 50
    image_y = 50

    # Converts a class vector (integers) to binary class matrix
    train_y = np_utils.to_categorical(Y_train)
    test_y = np_utils.to_categorical(Y_test)
    train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
    test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
    X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
    X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
    print("X_train shape: " + str(X_train.shape))
    print("X_test shape: " + str(X_test.shape))

    model, callbacks_list = keras_model(image_x, image_y)
    model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=10, batch_size=64,
              callbacks=callbacks_list)
    scores = model.evaluate(X_test, test_y, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    model.save('emojinator.h5')


main()
