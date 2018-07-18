import tensorflow as tf
import numpy as np


def model(images, features):
    pass


def model_s2(images):


def residual_layer(inputs, filters, kernel_size, name="residual_layer"):
    with tf.variable_scope(name):
        net = tf.layers.conv2d(inputs=inputs, filters=filter, kernel_size=kernel_size, )

def predict_image(features):
    pass


def compute_losses(predicted_images, true_images):
    pass


# BUILD THE GRAPH
tf_synthetic_image = tf.placeholder(shape=[None, 128, 128, 1], dtype=tf.float32, name="synthetic_image")
tf_collected_image = tf.placeholder(shape=[None, 128, 128, 1], dtype=tf.float32, name="collected_image")

# downsample the input images
tf_synthetic_image_scale_2 = tf.layers.max_pooling2d(inputs=tf_synthetic_image, pool_size=2, strides=2, padding="same",
                                                     name="synthetic_image_scale_2")
tf_collected_image_scale_2 = tf.layers.max_pooling2d(inputs=tf_collected_image, pool_size=2, strides=2, padding="same",
                                                     name="collected_image_scale_2")


# extract the features from each scale; the features from scale 2 are combined with features in scale 1
with tf.variable_scope("scale_2"):
    tf_features_scale_2 = model_2(images=tf_synthetic_image_scale_2)
    tf_predicted_image_scale_2 = predict_image(features=tf_features_scale_2)
    tf_scale_2_loss = compute_losses(predicted_images=tf_predicted_image_scale_2, true_images=tf_collected_image_scale_2)

with tf.variable_scope("scale_1"):
    tf_features_scale_1 = model(images=tf_synthetic_image, features=scale_2_features)
    tf_predicted_image_scale_1 = predict_image(features=tf_features_scale_1)
    tf_scale_1_loss = compute_losses(predicted_images=tf_predicted_image_scale_1, true_images=tf_collected_image)


# define the optimizers and training steps
ao_1 = tf.train.AdamOptimizer(learning_rate=1e-4)
ao_2 = tf.train.AdamOptimizer(learning_rate=1e-4)

s2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'scale_2')
s1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'scale_1')

tf_train_step_scale_1 = ao_1.minimize(scale_1_loss, var_list=s1_vars)
tf_train_step_scale_2 = ao_2.minimize(scale_2_loss, var_list=s2_vars)