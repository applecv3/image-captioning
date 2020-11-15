import tensorflow.compat.v1 as tf
import json
import transformer
import config
import efficientnet_builder
import image_processing
import preprocess
import collections

flags = tf.flags.FLAGS

tf.flags.DEFINE_string('pickle', './word2idx.pickle', 'word2idx pickle path')
tf.flags.DEFINE_string('model', 'b3', 'efficient net level')

tf.flags.DEFINE_string('ckpt', './output/efficientnet_b3/', 'path to save the trained model')
tf.flags.DEFINE_string('save', './frozen_graph.pb', 'path to save the trained model')


def build_input(params, image_to_process, caption):

    processed_image = tf.identity(image_processing.image_processing(image_to_process, params.img_size, False), name='processed_image')

    embedding_matrix = tf.get_variable('embedding_weights', [params.vocab_size, params.embedding_dim])
    embedded_caption = tf.nn.embedding_lookup(embedding_matrix, caption)
    position_encoding = transformer.position_encoding(params.embedding_dim, params.max_length)
    caption_input = embedded_caption + position_encoding

    return processed_image, caption_input


def build_encoder(image, is_training):

    context = tf.identity(efficientnet_builder.build_model_base(image, 'efficientnet-' + flags.model, is_training), name='context')

    return context


def build_decoder(encoded_image, caption_input, params, is_training):

    decoder_output = transformer.decoder_layer(caption_input, encoded_image, params, is_training)  # n,20,512
    logit = tf.layers.dense(decoder_output, params.vocab_size)  # n,16,5000
    probs = tf.nn.softmax(logit, name='probs')
    sorted_indices = tf.identity(tf.argsort(probs, axis=-1, direction='DESCENDING'), name='sorted_indices')

    return probs, sorted_indices


def get_image_and_answer(annotation_path):

    data = json.load(open(annotation_path, 'r'))
    processor = preprocess.Processor(data, flags.pickle, False)

    converted_captions = [processor.converter(words) for words in processor.selected_caption]
    image_names = [processor.img_idx2file[idx] for idx in processor.selected_image_id]

    img_name2answers = collections.defaultdict(list)

    for img_name, answer in zip(image_names, converted_captions):
        img_name2answers[img_name].append(answer)

    img_names, answers = list(img_name2answers.keys()), list(img_name2answers.values())

    return img_names, answers


def main(argv):

    del argv

    params = config.Config()
    #  get all the inputs

    image_to_process = tf.placeholder(tf.uint8, [None, None, 3], name='image_to_process')
    caption = tf.placeholder(tf.int32, [None, params.max_length], name='caption')

    processed_image, caption_input = build_input(params, image_to_process, caption)

    #  build encoder

    image_input = tf.placeholder(tf.float32, [None, params.img_size, params.img_size, 3], name='image_input')
    is_training = tf.placeholder_with_default(False, shape=[])

    context = build_encoder(image_input, is_training)

    #  build decoder

    context_shape = context.shape
    encoded_image = tf.placeholder(tf.float32, [None, context_shape[1], context_shape[-1]], name='encoded_image')

    logit, sorted_indice = build_decoder(encoded_image, caption_input, params, is_training)

    restorer = tf.train.Saver()

    with tf.Session() as sess:

        restorer.restore(sess, flags.ckpt)

        outputs = ['image_to_process', 'caption', 'processed_image', 'image_input',
                   'context', 'encoded_image', 'probs', 'sorted_indices']

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, outputs)

        with tf.gfile.GFile(flags.save, 'wb') as f:
            f.write(frozen_graph.SerializeToString())


if __name__ == '__main__':
    tf.disable_v2_behavior()
    tf.app.run(main)