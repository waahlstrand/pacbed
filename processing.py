import tensorflow as tf
import tensorflow_addons as tfa
import math as m
from typing import *

CROP_N_PIXELS = [616, 510, 420, 358, 266]
PI = tf.constant(m.pi)

@tf.function
def random_choice(x: tf.Tensor, k: int) -> tf.Tensor:
    
    idxs        = tf.range(tf.size(x))
    random_idxs = tf.random.shuffle(idxs)[:k]
    choice      = tf.gather(x, random_idxs)

    choice = tf.squeeze(choice)
    return choice

@tf.function
def augment(x: tf.Tensor, n_pixels: int, n_crop_pixels_list: List[int], eta_list: List[float], k: int = 1):
    """
    Series of augmentation applied per batch, or per image if the batch size is 1.

    To ensure randomized augmentation, make sure dataset is batched after applying the augmentation mapping.
    """
    # Randomly choose constants
    # n_crop_pixels_list  = tf.convert_to_tensor(n_crop_pixels_list)
    # eta_list            = tf.convert_to_tensor(eta_list)
    
    n_crop_pixels   = random_choice(n_crop_pixels_list, k)
    eta             = random_choice(eta_list, k)
    angle           = tf.random.uniform([1], minval=0, maxval=2*PI)

    # Rotate image
    x = tfa.image.rotate(x, angles=angle, interpolation="bilinear")

    # With certain probability, flip image
    p = tf.random.uniform((1, 1))
    x = tf.cond(p[0] <= 0.5, lambda: tf.image.flip_left_right(x), lambda: x)

    # Central crop of image with certain ratio
    # Resize to original size
    ratio = (n_crop_pixels / 2) / n_pixels
    box = [[ratio, 1-ratio, 1-ratio, ratio]] # Box coordinates, equivalent to indexes in range [0, 1] of height, width
    x = tf.reshape(x, (1, *x.shape))
    x = tf.image.crop_and_resize(x, boxes=box, box_indices=[0], crop_size=[n_pixels, n_pixels], method="bilinear")
    x = tf.squeeze(x, 0)

    # Add log-normal noise
    x = add_noise(x, eta)

    # Normalize by absolute value or sqrt(abs)
    x = normalize(x)

    return x


@tf.function
def normalize(x, scaling="linear"):

    # Normalize to mean intensity 0.25.
    x = 0.25 * x / tf.math.reduce_mean( tf.math.abs(x) )

    if scaling == "linear":
        # Absolute value of intensity.
        x = tf.math.abs(x)
    else:
        # Square root of absolute value of intensity.
        x = tf.math.sqrt( tf.math.abs(x) )

    return x

@tf.function
def label(y):

    # Log transformation.
    y = tf.math.log(y)

    return y

@tf.function
def add_noise(input, eta):

    # Noise model parameters.
    c1_log_intensity = -7.60540384562294
    c0_log_intensity = 28.0621318839493
    c2_log_s_signal = 0.0349329915028267
    c1_log_s_signal = -0.304984702105353
    c0_log_s_signal = 6.86126419242947
    c1_log_s_background = -6.99617784594964
    c0_log_s_background = 12.448421647627

    # Rescale intensity. Note that before this operation, the sum of x is 1.
    log_intensity   = c1_log_intensity * eta + c0_log_intensity
    intensity       = tf.math.exp(log_intensity)
    x               = tf.math.multiply(intensity, input)
    safety_noise    = tf.random.uniform(x.shape, minval=1e-5, maxval=1e-4) 
    x               += safety_noise
    # Compute lognormal distribution parameters (for each pixel).
    m_signal        = x

    # tf.print(m_signal)
    # tf.debugging.assert_non_negative(m_signal)
    # tf.debugging.assert_greater(m_signal, 0.0)

    m2_signal       = tf.math.multiply(x, x)
    log_s_signal    = c2_log_s_signal * tf.math.multiply(tf.math.log(m_signal), tf.math.log(m_signal)) + c1_log_s_signal * tf.math.log(m_signal) + c0_log_s_signal
    s_signal        = tf.math.exp(log_s_signal)
    s2_signal       = tf.math.multiply(s_signal, s_signal)

    mu_signal       = tf.math.log(m_signal) - 0.5 * tf.math.log(1.0 + tf.math.divide(s2_signal, m2_signal))
    sigma_signal    = tf.math.sqrt(tf.math.log(1.0 + tf.math.divide(s2_signal, m2_signal)))

    # Add lognormal noise.
    x = tf.math.exp(tf.random.normal(x.shape, mean = mu_signal, stddev = sigma_signal, dtype = tf.float32))

    # Compute normal distribution parameters.
    m_background        = 0.0
    log_s_background    = c1_log_s_background * eta + c0_log_s_background
    s_background        = tf.math.exp(log_s_background)

    # Add normal noise.
    x = tf.math.add(x, tf.random.normal(x.shape, mean = m_background, stddev = s_background, dtype = tf.float32))

    return x
