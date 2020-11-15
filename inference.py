import os
import tensorflow.compat.v1 as tf
from PIL import Image
import config
import numpy as np
import pickle
import matplotlib.pyplot as plt
import caption_generator

flags = tf.flags.FLAGS

tf.flags.DEFINE_string('img_dir', './val2014', 'image directory path')
tf.flags.DEFINE_string('pickle', './word2idx.pickle', 'word2idx pickle path')
tf.flags.DEFINE_string('save', './captioned_images', 'path to save captioned images')
tf.flags.DEFINE_string('pb', './frozen_graph.pb', 'path to saved pb file')

tf.flags.DEFINE_integer('n_beam', '3', 'beam size')


def main(argv):

    del argv

    params = config.Config()
    generator = caption_generator.Model(flags.pb)

    with open(flags.pickle, 'rb') as f:
        word2idx = pickle.load(f)

    idx2word = {idx: word for word, idx in word2idx.items()}

    if not tf.gfile.IsDirectory(flags.save):
        tf.gfile.MakeDirs(flags.save)

    filenames = os.listdir(flags.img_dir)

    for filename in filenames:

        full_path = os.path.join(flags.img_dir, filename)
        img = np.array(Image.open(full_path).convert('RGB'))

        processed_img = generator.process_image(img)
        context = generator.get_context([processed_img])

        #  beam search

        beam = []
        complete_sentence = None

        for step in range(params.max_length):

            if step == 0:

                initial_word = [1]+[0]*(params.max_length-(step+1))
                softmax, indices = generator.get_probs_and_indices(context, [initial_word])

                probs = softmax[0][step]
                top_k_words = indices[0][step][:flags.n_beam]
                top_k_probs = probs[top_k_words]

                for i in range(flags.n_beam):
                    beam.append([top_k_probs[i], [top_k_words[i]]])

            else:

                current_beam = []
                is_done = False

                for i in range(flags.n_beam):

                    prob, word = beam[i][0], beam[i][1]
                    last_word = word[-1]

                    if last_word == 3:
                        is_done = not is_done
                        complete_sentence = beam[i][1]
                        break

                    softmax, indices = generator.get_probs_and_indices(context, [[1]+beam[i][1]+[0]*(params.max_length-(step+1))])

                    probs = softmax[0][step]
                    top_k_words = indices[0][step][:flags.n_beam]
                    top_k_probs = probs[top_k_words]

                    for j in range(flags.n_beam):

                        current_beam.append([prob * top_k_probs[j], word + [top_k_words[j]]])

                if is_done:
                    break

                beam = sorted(current_beam, key=lambda x: -x[0])[:3]

        caption = complete_sentence if complete_sentence else beam[0][1]

        if caption[-1] == 3:
            caption.pop()

        sentence = [idx2word[w_idx] for w_idx in caption]
        sentence = ' '.join(sentence)
        sentence = sentence[0].upper() + sentence[1:] + '.'

        plt.imshow(img)
        plt.axis('off')
        plt.title(sentence)
        plt.savefig(os.path.join(flags.save, filename))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.disable_v2_behavior()
    tf.app.run(main)