import tensorflow.compat.v1 as tf
import image_processing


class Dataset:

    def __init__(self, path, batch, shards, params, is_training=False):

        self.config = params
        self.path = path
        self.n_batch = batch
        self.n_shards = shards
        self.is_training = is_training
        self.iterator = None
        self.build()

    def image_decode(self, img):

        image = tf.io.decode_image(img, channels=3)
        image.set_shape([None, None, 3])

        return image

    def truncate(self, caption):

        true_value_indices = tf.where(tf.not_equal(caption, 0))
        true_value_length = tf.random.uniform([], minval=1, maxval=tf.shape(true_value_indices)[0], dtype=tf.int32)
        mask = tf.ones(true_value_length, tf.int32)
        mask = tf.pad(mask, [[0, self.config.max_length - true_value_length]])
        truncated_caption = caption * mask

        return truncated_caption

    def caption_processing(self, caption, is_training):

        caption = tf.cast(caption, tf.int32)

        if is_training:

            is_truncate = tf.greater_equal(tf.random.uniform([]), 0.5)
            caption = tf.cond(is_truncate, lambda: self.truncate(caption), lambda: caption)

        return caption

    def parse_function(self, data):

        features = tf.parse_single_example(data, features={'image': tf.FixedLenFeature([], tf.string),
                                                           'caption': tf.FixedLenFeature([self.config.max_length], tf.int64)})

        image = self.image_decode(features['image'])
        image = image_processing.image_processing(image, self.config.img_size, self.is_training)
        caption = self.caption_processing(features['caption'], self.is_training)

        return image, caption

    def build(self):

        filenames = tf.data.Dataset.list_files(self.path, shuffle=self.is_training)
        dataset = filenames.apply(
            tf.data.experimental.parallel_interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=self.n_shards, sloppy=self.is_training))
        dataset = dataset.repeat()
        dataset = dataset.map(map_func=self.parse_function, num_parallel_calls=20)
        dataset = dataset.batch(self.n_batch, drop_remainder=not self.is_training).prefetch(self.n_batch)
        self.iterator = tf.data.make_one_shot_iterator(dataset)

    def get_batch(self):

        return self.iterator.get_next()


