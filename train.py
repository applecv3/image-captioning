import tensorflow.compat.v1 as tf
import dataloader
import transformer
import config
import efficientnet_builder

flags = tf.flags.FLAGS

tf.flags.DEFINE_string('train', '', 'path to train tfrecord data set')
tf.flags.DEFINE_string('eval', '', 'path to train tfrecord data set')
tf.flags.DEFINE_string('model', 'b3', 'efficient net level')
tf.flags.DEFINE_string('ckpt', './efficientnet_weights/efficientnet-b3/efficientnet-b3/model.ckpt', 'path to imagenet pretrained weights')
tf.flags.DEFINE_string('save', './output/efficientnet_b3/', 'path to save the trained model')

tf.flags.DEFINE_integer('batch_train', '', 'batch size to train')
tf.flags.DEFINE_integer('batch_eval', '', 'batch size to train')
tf.flags.DEFINE_integer('n_eval', '', 'number of eval data')
tf.flags.DEFINE_integer('n_shards', '200', 'number of tfrecord shards')
tf.flags.DEFINE_integer('n_epoch', '', 'total steps for training')
tf.flags.DEFINE_integer('n_iter', '', 'iter interval to evaluate')
tf.flags.DEFINE_integer('decay', '100', 'step interval(n_iter * this value) to decay lr')

tf.flags.DEFINE_float('lr', '0.00005', 'initial learning rate')
tf.flags.DEFINE_float('decay_rate', '0.95', 'initial learning rate')
tf.flags.DEFINE_float('ema_rate', '0.99', 'initial learning rate')


def build_input(params):

    train_dataset = dataloader.Dataset(flags.train, flags.batch_train, flags.n_shards, params, True)
    eval_dataset = dataloader.Dataset(flags.eval, flags.batch_eval, flags.n_shards, params)

    is_training = tf.placeholder_with_default(False, shape=[])
    # get image and caption batch
    image, caption_label = tf.cond(is_training, lambda: train_dataset.get_batch(), lambda: eval_dataset.get_batch())
    # create mask
    mask = tf.where(tf.equal(caption_label, 0), tf.zeros_like(caption_label, tf.float32),
                    tf.ones_like(caption_label, tf.float32))
    # shift one step to the right
    dynamic_n_batch = tf.shape(caption_label)[0]
    start_idx = tf.ones([dynamic_n_batch, 1], dtype=tf.int32)
    sliced_caption = tf.slice(caption_label, [0, 0], [dynamic_n_batch, params.max_length - 1])
    caption_input = tf.concat([start_idx, sliced_caption], axis=-1)

    # embedding and positional encoding
    embedding_matrix = tf.get_variable('embedding_weights', [params.vocab_size, params.embedding_dim])
    embedded_caption = tf.nn.embedding_lookup(embedding_matrix, caption_input)
    position_encoding = transformer.position_encoding(params.embedding_dim, params.max_length)
    caption_input = embedded_caption + position_encoding

    return image, caption_input, caption_label, mask, is_training


def build_model(image, caption_input, params, is_training):

    context = efficientnet_builder.build_model_base(image, 'efficientnet-'+flags.model, is_training)
    decoder_output = transformer.decoder_layer(caption_input, context, params, is_training)
    logit = tf.layers.dense(decoder_output, params.vocab_size)

    return logit


def build_accuracy(logit, label, mask):

    prediction = tf.argmax(logit, axis=-1)
    correct = tf.cast(tf.equal(prediction, label), tf.float32) * mask
    accuracy = tf.reduce_mean(tf.reduce_sum(correct, axis=-1) / tf.reduce_sum(mask, axis=-1))

    return accuracy


def build_loss(logit, label, mask, params):

    label = tf.one_hot(label, params.vocab_size, axis=-1)
    entropy = tf.losses.softmax_cross_entropy(
        label, logit, label_smoothing=0.1, reduction=tf.losses.Reduction.NONE
    ) * mask

    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([tf.reduce_mean(entropy)]+reg_loss)

    return total_loss


def build_optimizer(loss):

    step = tf.Variable(0, trainable=False)
    tvars = tf.trainable_variables()

    lr = tf.train.exponential_decay(flags.lr, step, flags.n_iter*flags.decay, flags.decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=step)

    ema = tf.train.ExponentialMovingAverage(flags.ema_rate, step)
    bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(bn_ops + [optimizer]):
        train_op = ema.apply(tvars)

    ema_restorer = tf.group([tf.assign(var, ema.average(var)) for var in tvars])

    return train_op, step, ema_restorer


def main(argv):

    del argv

    params = config.Config()
    #  get all the inputs
    image, caption_input, caption_label, mask, is_training = build_input(params)
    #  build model
    logit = build_model(image, caption_input, params, is_training)
    #  build accuracy
    accuracy = build_accuracy(logit, caption_label, mask)
    #  build loss
    loss = build_loss(logit, caption_label, mask, params)
    #  build optimizer
    optimizer, global_step, ema_restorer = build_optimizer(loss)

    gvars = tf.global_variables()
    vars_to_restore = []

    for var in gvars:  # restore imagenet pretrained EfficientNet weights
        vname = var.name.split(':')[0]
        try:
            var_to_restore = tf.train.load_variable(flags.ckpt, vname)
            if var_to_restore.shape == var.shape:
                vars_to_restore.append(var)
                print(var, "is going to be restored from ckpt")
        except:
            print(var, "not found in ckpt")

    restorer = tf.train.Saver(vars_to_restore)
    saver = tf.train.Saver()
    best_acc = 0

    eval_step = flags.n_eval // flags.batch_eval

    if not tf.io.gfile.exists(flags.save):
        tf.io.gfile.makedirs(flags.save)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        restorer.restore(sess, flags.ckpt)

        for _ in range(flags.n_epoch):

            for iter in range(flags.n_iter):
                sess.run(optimizer, feed_dict={is_training: True})

            sess.run(ema_restorer)
            step = sess.run(global_step)
            acc = 0

            for _ in range(eval_step):

                acc += sess.run(accuracy)

            acc /= eval_step

            if best_acc < acc:
                best_acc = acc
                saver.save(sess, flags.save)
                print('*SAVED STEP %d accuracy: %f' % (step, best_acc))
            else:
                print('STEP %d accuracy: %f' % (step, best_acc))


if __name__ == '__main__':
    tf.disable_v2_behavior()
    tf.app.run(main)