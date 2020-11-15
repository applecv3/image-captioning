import tensorflow.compat.v1 as tf


class Model:

    def __init__(self, path):

        self.path = path

        self.image_to_process = None
        self.processed_image = None
        self.caption = None
        self.image_input = None
        self.context = None
        self.encoded_image = None
        self.probs = None
        self.sorted_indices = None

        self.build()
        self.sess = tf.Session()

    def build(self):

        with tf.gfile.GFile(self.path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.image_to_process, self.caption, self.processed_image, self.image_input,\
        self.context, self.encoded_image, self.probs, self.sorted_indices = tf.import_graph_def(graph_def,
                                                                              return_elements=['image_to_process:0',
                                                                                               'caption:0',
                                                                                               'processed_image:0',
                                                                                               'image_input:0',
                                                                                               'context:0',
                                                                                               'encoded_image:0',
                                                                                               'probs:0',
                                                                                               'sorted_indices:0'])

    def process_image(self, img):

        return self.sess.run(self.processed_image, feed_dict={self.image_to_process: img})

    def get_context(self, img):

        return self.sess.run(self.context, feed_dict={self.image_input: img})

    def get_probs_and_indices(self, encoded_image, caption):

        return self.sess.run([self.probs, self.sorted_indices], feed_dict={self.encoded_image: encoded_image, self.caption: caption})








