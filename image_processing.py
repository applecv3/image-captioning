import tensorflow.compat.v1 as tf


def scale_and_resize(image, size):

    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    scale_h = tf.cast(size / height, tf.float32)
    scale_w = tf.cast(size / width, tf.float32)

    scale = tf.minimum(scale_h, scale_w)

    scaled_h = tf.cast(scale * tf.cast(height, tf.float32), tf.int32)
    scaled_w = tf.cast(scale * tf.cast(width, tf.float32), tf.int32)

    scaled_image = tf.image.resize_images(image, [scaled_h, scaled_w])
    padded_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, size, size)

    return padded_image


def random_flip(image):

    is_flip = tf.greater_equal(tf.random.uniform([]), 0.5)

    fliped_image = tf.cond(is_flip, lambda: tf.image.flip_left_right(image), lambda: image)

    return fliped_image


def image_processing(image, size, is_training):

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = scale_and_resize(image, size)

    if is_training:
        image = random_flip(image)

    return image