#%%
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import models
import learning_rate
import datasets
import processing
import matplotlib.pyplot as plt

#%%
# Constants
CONVERGENCE_ANGLES = [6.85, 13.32, 19.08, 26.69, 34.35]
N_CROP_PIXELS = [616, 510, 420, 358, 266]
ETAS = [0.00, 0.25, 0.50, 0.75, 1.0]
DATA_ROOT = Path("./data/")
LOG_DIR   = Path("./logs/")
RESULTS_DIR = Path("./results/")
N_SAMPLES_PER_SET = 165
N_REPEAT = 10
N_PIXELS = 1040

#%%
# Hyperparameters
batch_size_train = 16
prefetch_size = 2
number_of_parallel_calls = 32
n_epochs = 10
lr = -6.0
momentum = 0.9
nesterov = False

log10_learning_rate_vector = np.array([lr])
number_of_epochs_vector    = np.array([n_epochs])
number_of_epochs = np.sum(number_of_epochs_vector)

#%% Data
# Increase size, shuffle, map augmentation, and finally batch
files = list(DATA_ROOT.rglob("*.bin"))

# Load data from multiple files
dataset = datasets.load_pacbed_dataset(files[0], N_SAMPLES_PER_SET, N_PIXELS)
# augment = processing.Augment(eta=0.50, crop=616, size=N_PIXELS)
# images, labels = tuple(zip(*dataset))
#%%
# Repeat data by a constant factor, increasing size
dataset = dataset.repeat(N_REPEAT).shuffle(1000, reshuffle_each_iteration=False)
# dataset_size = len(list(files))*N_SAMPLES_PER_SET*N_REPEAT

#%%
# # Split dataset for validation and testing
# train, val, test = data.split_dataset(dataset, dataset_size, {'train': 0.7, 'val': 0.2, 'test': 0.1})

# val = val.map(lambda x, y: (x,
#                 processing.label(y)), 
#                 num_parallel_calls = 32)\
#          .batch(batch_size_train)\
#          .prefetch(prefetch_size)

# test = test.batch(batch_size_train)\
#             .map(lambda x, y: (x,
#                 processing.label(y)), 
#                 num_parallel_calls = 32)\
#            .prefetch(prefetch_size)

# train = train.map(lambda x, y: (
#                 processing.augment(x, 
#                                     N_PIXELS, 
#                                     N_CROP_PIXELS,
#                                     ETAS
#                                     ), 
#                 processing.label(y)), 
#                 num_parallel_calls = 32)\
#             .batch(batch_size_train)\
#             .prefetch(prefetch_size)

train = dataset.batch(batch_size_train).prefetch(prefetch_size)

#%% Logging
path_losses = RESULTS_DIR / "losses.csv"
path_parameters = RESULTS_DIR / "parameters.dat"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
file_writer = tf.summary.create_file_writer(str(LOG_DIR / "metrics"))
file_writer.set_as_default()

callback_csv_logger = tf.keras.callbacks.CSVLogger(path_losses, 
                                    append = True, 
                                    separator = ';')

#%%
# Callbacks
callback_learning_rate = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: learning_rate.schedule( epoch, log10_learning_rate_vector, number_of_epochs_vector)
    ) 

#%%
sgd = tf.keras.optimizers.SGD(momentum = momentum, nesterov = nesterov)
# model = tf.keras.models.Sequential([ augment, models.NaiveConvolutional(N_PIXELS) ])

from models import augmentation, naiveConvolutional

model = naiveConvolutional(N_PIXELS, name="test")
augment = augmentation(N_PIXELS, eta=0.50, crop=616, name="augment")

augmented_model = tf.keras.Model(augment.input, model(augment.output), "augmented")

#%%
# Train model
model.compile(
    loss = tf.keras.losses.mean_squared_error, 
    optimizer = sgd, 
    run_eagerly=True)

#%%
results = model.fit(train, 
                    epochs = number_of_epochs, 
                    # validation_data = val,
                    verbose = 1,
                    callbacks = [ 
                        callback_learning_rate, 
                        callback_csv_logger, 
                        tensorboard_callback ],
                    )


# %%
