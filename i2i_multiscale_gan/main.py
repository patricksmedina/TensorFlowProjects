import tensorflow as tf
import numpy as np


def local_enhancer(images, features):
    conv = tf.layers.conv2d(inputs=images, filters=32, kernel_size=7, stride=1, padding='same', activations=tf.nn.elu,
                            name="conv2d_1")
    conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=3, stride=2, padding='same', activations=tf.nn.elu,
                            name="conv2d_2")
    conv = tf.add(images, features)

    conv = dense_conv_block(inputs=conv, filters=64, kernel_size=3, activation=tf.nn.elu, name="dense_conv_1")
    conv = dense_conv_block(inputs=conv, filters=64, kernel_size=3, activation=tf.nn.elu, name="dense_conv_2")
    conv = dense_conv_block(inputs=conv, filters=64, kernel_size=3, activation=tf.nn.elu, name="dense_conv_3")

    # TODO: upsample 1-layer

    conv = tf.layers.conv2d(inputs=conv, filters=1, kernel_size=7, padding='same', activations=tf.nn.elu,
                            name="conv2d_3")

    return conv


def global_network(images):
    conv = tf.layers.conv2d(inputs=images, filters=64, kernel_size=7, stride=1, padding='same', activations=tf.nn.elu)


def dense_conv_block(inputs, filters, kernel_size, activation, name="dense_block"):
    """Implementation of a dense convolution block."""

    with tf.variable_scope(name):
        # NOTE: It is assumed filters is divisible by 4.
        internal_filters = filters // 4

        # initial projection
        net_0 = tf.layers.conv2d(inputs=inputs, filters=internal_filters, kernel_size=1, padding='same',
                                 activation=activation, name="conv2d_0")

        # dense group 1
        net_1_0 = tf.layers.conv2d(inputs=net_0, filters=internal_filters, kernel_size=kernel_size, padding='same',
                                   activation=activation, name="conv2d_1")
        net_1_1 = tf.concat([net_0, net_1_0], axis=-1)

        # dense group 2
        net_2_0 = tf.layers.conv2d(inputs=net_1_1, filters=internal_filters, kernel_size=kernel_size, padding='same',
                                   activation=activation, name="conv2d_2")
        net_2_1 = tf.concat([net_0, net_1_0, net_2_0], axis=-1)

        # dense group 3
        net_3_0 = tf.layers.conv2d(inputs=inputs, filters=internal_filters, kernel_size=kernel_size, padding='same',
                                 activation=activation, name="conv2d_3")
        net_3_1 = tf.concat([net_0, net_1_0, net_2_0, net_3_0], axis=-1)

    return net_3_1

def predict_image(features):
    pass


def compute_losses(predicted_images, true_images):
    pass


# temporary home for the input parameters
learning_rate_s1 = 1e-4
learning_rate_s2 = 1e-4

# BUILD THE GRAPH
tf_synthetic_image = tf.placeholder(shape=[None, 128, 128, 1], dtype=tf.float32, name="synthetic_image")
tf_collected_image = tf.placeholder(shape=[None, 128, 128, 1], dtype=tf.float32, name="collected_image")

# downsample the input images
tf_synthetic_image_scale_2 = tf.layers.max_pooling2d(inputs=tf_synthetic_image, pool_size=2, strides=2, padding="same",
                                                     name="synthetic_image_scale_2")
tf_collected_image_scale_2 = tf.layers.max_pooling2d(inputs=tf_collected_image, pool_size=2, strides=2, padding="same",
                                                     name="collected_image_scale_2")


# extract the features from each scale; the features from scale 2 are combined with features in scale 1
with tf.variable_scope("global_network"):
    tf_features_scale_2 = global_network(images=tf_synthetic_image_scale_2)
    tf_predicted_image_scale_2 = predict_image(features=tf_features_scale_2)
    tf_scale_2_loss = compute_losses(predicted_images=tf_predicted_image_scale_2, true_images=tf_collected_image_scale_2)

with tf.variable_scope("local_enhancer"):
    tf_features_scale_1 = local_enhancer(images=tf_synthetic_image, features=scale_2_features)
    tf_predicted_image_scale_1 = predict_image(features=tf_features_scale_1)
    tf_scale_1_loss = compute_losses(predicted_images=tf_predicted_image_scale_1, true_images=tf_collected_image)


# define the optimizers and training steps
# the two optimizers allow for pre-training the
# networks at a lower scale separately.

# TODO: Make the learning rate an input parameter
ao_1 = tf.train.AdamOptimizer(learning_rate=1e-4)
ao_2 = tf.train.AdamOptimizer(learning_rate=1e-4)

s2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'scale_2')
s1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'scale_1')

tf_train_step_scale_1 = ao_1.minimize(scale_1_loss, var_list=s1_vars)
tf_train_step_scale_2 = ao_2.minimize(scale_2_loss, var_list=s2_vars)
