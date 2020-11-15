import os
import tensorflow.compat.v1 as tf
import json
from PIL import Image
import config
import caption_generator
import preprocess
import collections
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

flags = tf.flags.FLAGS

tf.flags.DEFINE_string('img_dir', './', 'image directory path')
tf.flags.DEFINE_string('pickle', './word2idx.pickle', 'word2idx pickle path')
tf.flags.DEFINE_string('ann_dir', './', 'path to annotation folder')
tf.flags.DEFINE_string('pb', './frozen_graph.pb', 'path to saved pb file')

tf.flags.DEFINE_integer('n_batch', '200', 'batch size to eval')
tf.flags.DEFINE_integer('n_beam', '3', 'beam size')


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
    generator = caption_generator.Model(flags.pb)

    filenames, answers = get_image_and_answer(flags.ann_dir)

    c = 0
    n_images = len(filenames)

    print('Evaluating %d images' % n_images)

    score1, score2, score3, score4, score_total = 0, 0, 0, 0, 0

    for start_idx in range(0, n_images, flags.n_batch):

        end_idx = start_idx + flags.n_batch

        if end_idx > n_images:
            end_idx = n_images

        batch_img_files = filenames[start_idx: end_idx]
        batch_references = answers[start_idx: end_idx]
        batch_images = []

        for img_file in batch_img_files:
            c += 1
            full_path = os.path.join(flags.img_dir, img_file)
            img = np.array(Image.open(full_path).convert('RGB'))
            batch_images.append(generator.process_image(img))
            print(img_file, c)

        batch_context = generator.get_context(batch_images)

        #  *** Start BeamSearch ***

        actual_n_batch = end_idx - start_idx

        beam_batch = [[] for _ in range(actual_n_batch)]
        complete_sentence_batch = [[] for _ in range(actual_n_batch)]

        for step in range(params.max_length):

            if step == 0:

                batch_initial_word = [[1]+[0]*(params.max_length-(step+1)) for _ in range(actual_n_batch)]
                batch_softmax, batch_indice = generator.get_probs_and_indices(batch_context, batch_initial_word)

                for batch_idx in range(actual_n_batch):

                    probs = batch_softmax[batch_idx][step]
                    top_k_words = batch_indice[batch_idx][step][:flags.n_beam]
                    top_k_probs = probs[top_k_words]

                    for i in range(flags.n_beam):
                        beam_batch[batch_idx].append([top_k_probs[i], [top_k_words[i]]])

            else:

                current_beam_batch = [[] for _ in range(actual_n_batch)]

                for i in range(flags.n_beam):

                    batch_prob, batch_word = [], []

                    for batch_idx in range(actual_n_batch):

                        batch_prob.append(beam_batch[batch_idx][i][0])
                        last_word = beam_batch[batch_idx][i][1][-1]

                        if last_word == 3:
                            complete_sentence_batch[batch_idx].append(beam_batch[batch_idx][i][1])

                        batch_word.append([1]+beam_batch[batch_idx][i][1]+[0]*(params.max_length-(step+1)))

                    batch_softmax, batch_indice = generator.get_probs_and_indices(batch_context, batch_word)

                    for batch_idx in range(actual_n_batch):

                        probs = batch_softmax[batch_idx][step]
                        top_k_words = batch_indice[batch_idx][step][:flags.n_beam]
                        top_k_probs = probs[top_k_words]

                        for j in range(flags.n_beam):

                            current_beam_batch[batch_idx].append([batch_prob[batch_idx] * top_k_probs[j],
                                                                  beam_batch[batch_idx][i][1] + [top_k_words[j]]])

                beam_batch = [sorted(beam, key=lambda x: -x[0])[:3] for beam in current_beam_batch]

        batch_candidate = [complete_sentence[0] if complete_sentence else beam_batch[idx][0][1] for idx, complete_sentence in
                  enumerate(complete_sentence_batch)]

        for reference, candidate in zip(batch_references, batch_candidate):

            if candidate[-1] == 3:
                candidate.pop()

            score1 += sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            score2 += sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
            score3 += sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
            score4 += sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
            score_total += sentence_bleu(reference, candidate)

    score1 /= n_images
    score2 /= n_images
    score3 /= n_images
    score4 /= n_images
    score_total /= n_images

    print('1-gram BLEU score: %f' % score1)
    print('2-gram BLEU score: %f' % score2)
    print('3-gram BLEU score: %f' % score3)
    print('4-gram BLEU score: %f' % score4)
    print('BLEU score: %f' % score_total)


if __name__ == '__main__':
    tf.disable_v2_behavior()
    tf.app.run(main)