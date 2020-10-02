import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import Callback
from keras import backend as K


# Define a generic function to add resent layer
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation=True,
                 batch_normalization=True):
    # print("Adding Conv2D - filter:", num_filters, " kernel_size:", kernel_size, " strides:", strides)
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation:
        x = Activation('relu')(x)

    return x


def resnet_add_indentity_block(inputs, num_filters):
    # The main path
    y = resnet_layer(inputs, num_filters=num_filters)
    y = resnet_layer(y, num_filters=num_filters, activation=False)

    x = keras.layers.add([inputs, y])
    x = Activation('relu')(x)

    return x


def resnet_add_convulution_block(inputs, num_filters):
    # The main path
    y = resnet_layer(inputs, num_filters=num_filters, strides=2)
    y = resnet_layer(y, num_filters=num_filters, activation=False)

    # The shortcut - this has to match the dimensions, hence stride is 2
    x = resnet_layer(inputs, num_filters=num_filters, kernel_size=1, strides=2, activation=False)

    x = keras.layers.add([x, y])
    x = Activation('relu')(x)

    return x

#define RESTNET
def resnet_v1():

    #define the input shape
    inputs=Input(shape=(224, 224, 3))
    #inputs = Input(shape=input_shape)

    num_filters_in = 32

    x = resnet_layer(inputs=inputs, num_filters=num_filters_in) #RF: 3

    #Stage 1
    x = resnet_add_indentity_block(x, num_filters_in) #RF: 3, 7
    x = resnet_add_indentity_block(x, num_filters_in) #RF: 3, 7, 11
    x = resnet_add_indentity_block(x, num_filters_in) #RF: 3, 7, 11, 15

    #Stage 2
    num_filters_in *= 2
    x = resnet_add_convulution_block(x, num_filters_in) #RF: 3, 7, 11, 15, 21
    x = resnet_add_indentity_block(x, num_filters_in) #RF: 3, 7, 11, 15, 21, 29
    x = resnet_add_indentity_block(x, num_filters_in) #RF: 3, 7, 11, 15, 21, 29, 37

    #Stage 3
    num_filters_in *= 2
    x = resnet_add_convulution_block(x, num_filters_in) #RF: 3, 7, 11, 15, 21, 29, 49
    x = resnet_add_indentity_block(x, num_filters_in) #RF: 3, 7, 11, 15, 21, 29, 49, 65
    x = resnet_add_indentity_block(x, num_filters_in) #RF: 3, 7, 11, 15, 21, 29, 49, 81
    x = resnet_add_indentity_block(x, num_filters_in) #RF: 3, 7, 11, 15, 21, 29, 49, 97

    #Stage 4
    num_filters_in *= 2
    x = resnet_add_convulution_block(x, num_filters_in) #RF: 3, 7, 11, 15, 21, 29, 49, 97, 105
    x = resnet_add_indentity_block(x, num_filters_in) #RF: 3, 7, 11, 15, 21, 29, 49, 97, 105, 137
    x = resnet_add_indentity_block(x, num_filters_in) #RF: 3, 7, 11, 15, 21, 29, 49, 97, 105, 137, 169
    x = resnet_add_indentity_block(x, num_filters_in) #RF: 3, 7, 11, 15, 21, 29, 49, 97, 105, 137, 169, 201
    x = resnet_add_indentity_block(x, num_filters_in) #RF: 3, 7, 11, 15, 21, 29, 49, 97, 105, 137, 169, 233

    #Add classifier on top.
    #v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)

    return x, inputs


