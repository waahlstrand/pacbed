# Import libraries.
import numpy as np
import time
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, AveragePooling2D, SpatialDropout2D, Flatten
from tensorflow.keras import optimizers, losses, activations, regularizers
from tensorflow.keras.regularizers import *
from tensorflow.keras.callbacks import CSVLogger
import os

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