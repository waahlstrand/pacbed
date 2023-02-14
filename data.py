import numpy as np
import tensorflow as tf


def process_pacbed_data_from_file(x, number_of_samples, number_of_pixels):

    x = np.reshape(x, (number_of_pixels, number_of_pixels, number_of_samples, 1)) # Saved data format
    x = np.swapaxes(x, 0, 2) # Tensorflow requires a (B, H, W, C) format
    x = np.swapaxes(x, 1, 2)

    # Target is simply the thickness, equivalent to index + 1
    y = np.arange(1, number_of_samples+1).reshape(number_of_samples, 1)

    return x, y


def load_multiple_pacbed_datasets(files, number_of_samples, number_of_pixels):

    data = [np.fromfile(file, dtype = np.float32) for file in files]
    data = [process_pacbed_data_from_file(d, number_of_samples, number_of_pixels) for d in data]
    xs, ys = list(zip(*data))

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    dataset = tf.data.Dataset.from_generator(   lambda: zip(iter(x), iter(y)), 
                                                output_shapes = ((number_of_pixels, number_of_pixels, 1), (1)), 
                                                output_types = (tf.float32, tf.float32))

    return dataset


def load_pacbed_dataset(path_data, number_of_samples, number_of_pixels):


    x = np.fromfile(path_data, dtype = np.float32)
    x, y = process_pacbed_data_from_file(x, number_of_samples, number_of_pixels)

    dataset = tf.data.Dataset.from_generator(   lambda: zip(iter(x), iter(y)), 
                                                output_shapes = ((number_of_pixels, number_of_pixels, 1), (1)), 
                                                output_types = (tf.float32, tf.float32))

    return dataset