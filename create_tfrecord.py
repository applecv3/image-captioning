import os
import tensorflow.compat.v1 as tf
import json
import preprocess


flags = tf.flags.FLAGS

tf.flags.DEFINE_string('img_dir', '', 'path to image folder')
tf.flags.DEFINE_string('ann_dir', '', 'path to annotation folder')
tf.flags.DEFINE_string('pickle', './word2idx.pickle', 'path to save or saved word2idx dict')

tf.flags.DEFINE_bool('is_training', 'True', 'train data if True else False')

tf.flags.DEFINE_integer('shards', '200', 'number of tfrecord files')


def _bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def main(argv):

    del argv

    c = 0  # to count how many images it's gone through
    s = 0  # to count how many images it's skipped

    data = json.load(open(flags.ann_dir, 'r'))
    processor = preprocess.Processor(data, flags.pickle, flags.is_training)

    number_of_data = len(processor.selected_caption)
    number_of_data_per_shard = number_of_data // flags.shards

    save_dir = './train_tfrecords/' if flags.is_training else './eval_tfrecords/'

    if not tf.gfile.IsDirectory(save_dir):
        tf.gfile.MakeDirs(save_dir)

    writers = [tf.io.TFRecordWriter(save_dir +'tfrecord_'+ str(i)) for i in range(flags.shards)]
    print('Processing %d images' % number_of_data)
    for idx, items in enumerate(zip(processor.selected_caption, processor.selected_image_id)):

        caption, image_id = items

        img_path = os.path.join(flags.img_dir, processor.img_idx2file[image_id])

        try:
            img = tf.io.gfile.GFile(img_path, 'rb').read()
        except:
            s += 1
            print(img_path, "has been skipped")
            continue

        c += 1

        converted_sentence = processor.converter(caption)
        caption_data = processor.padding(converted_sentence)

        features = {'image': _bytes_feature([img]), 'caption': _int_feature(caption_data)}
        example = tf.train.Example(features=tf.train.Features(feature=features))

        shard_idx = idx // number_of_data_per_shard

        if shard_idx >= flags.shards:
            shard_idx -= 1

        writers[shard_idx].write(example.SerializeToString())

    print('%d images in total' % c)
    print('%d images have been skipped' % s)


if __name__ == '__main__':
    tf.disable_v2_behavior()
    tf.app.run(main=main)
