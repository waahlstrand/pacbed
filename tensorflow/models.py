# Import libraries.
import numpy as np
import time
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Conv2D, AveragePooling2D, SpatialDropout2D, Flatten, RandomRotation, RandomFlip, CenterCrop, Resizing
from tensorflow.keras import optimizers, losses, activations, regularizers
from tensorflow.keras.regularizers import *
from tensorflow.keras.callbacks import CSVLogger
import processing
import os


def augmentation(n_pixels: int, eta: float, crop: int, name: str, scaling="linear"):

    input_shape = (n_pixels, n_pixels, 1)

    inputs = Input(shape=input_shape)
    x = RandomRotation(0.5)(inputs)
    x = RandomFlip()(x)
    x = CenterCrop(height=crop, width=crop)(x)
    x = Resizing(n_pixels, n_pixels)(x)
    x = processing.AddNoise(eta=eta)(x)
    x = processing.Normalize(scaling=scaling)(x)

    return Model(inputs=inputs, outputs=x, name=name) 

def naiveConvolutional(n_pixels: int, name: str):
    
    input_shape = (n_pixels, n_pixels, 1)
    
    kernel_size = (3, 3)
    number_of_filters = 16

    inputs = Input(shape=input_shape)
    x = Conv2D(number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same",)(inputs)
    x = Conv2D(number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same")(x)
    x = AveragePooling2D(pool_size = (2, 2))(x)
    
    x = Conv2D(2 * number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same")(x)
    x = Conv2D(2 * number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same")(x)
    x = AveragePooling2D(pool_size = (2, 2))(x)

    x = Conv2D(4 * number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same")(x)
    x = Conv2D(4 * number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same")(x)
    x = AveragePooling2D(pool_size = (2, 2))(x)

    x = Conv2D(8 * number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same")(x)
    x = Conv2D(8 * number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same")(x)
    x = AveragePooling2D(pool_size = (2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation = 'elu')(x)
    x = Dense(192, activation = 'elu')(x)
    x = Dense(128, activation = 'elu')(x)
    x = Dense(64, activation = 'elu')(x)
    x = Dense(1, activation = 'linear')(x)

    return Model(inputs=inputs, outputs=x, name=name)







def NaiveConvolutional(number_of_pixels):
    input_shape = (number_of_pixels, number_of_pixels, 1)
    
    kernel_size = (3, 3)
    number_of_filters = 16
    
    model = Sequential()

    model.add(Conv2D(number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same", input_shape = input_shape))
    model.add(Conv2D(number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same"))
    model.add(AveragePooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(2 * number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same"))
    model.add(Conv2D(2 * number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same"))
    model.add(AveragePooling2D(pool_size = (2, 2)))

    model.add(Conv2D(4 * number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same"))
    model.add(Conv2D(4 * number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same"))
    model.add(AveragePooling2D(pool_size = (2, 2)))

    model.add(Conv2D(8 * number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same"))
    model.add(Conv2D(8 * number_of_filters, kernel_size = kernel_size, activation = 'elu', padding = "same"))
    model.add(AveragePooling2D(pool_size = (2, 2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation = 'elu'))
    model.add(Dense(192, activation = 'elu'))
    model.add(Dense(128, activation = 'elu'))
    model.add(Dense(64, activation = 'elu'))
    model.add(Dense(1, activation = 'linear'))
    
    return model