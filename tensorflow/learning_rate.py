# Import libraries.
import numpy as np
import tensorflow as tf

def schedule(current_epoch, log10_learning_rate_vector, number_of_epochs_vector):
   
    log10_learning_rate = log10_learning_rate_vector[0]
    for current_step in range(1, number_of_epochs_vector.size):
        if current_epoch >= np.cumsum(number_of_epochs_vector)[current_step-1]:
            log10_learning_rate = log10_learning_rate_vector[current_step]


    lr = 10 ** log10_learning_rate
    tf.summary.scalar('learning rate', data=lr, step=current_epoch)

    return lr