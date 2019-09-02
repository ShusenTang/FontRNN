import json
import os
import six
import time
import numpy as np
import tensorflow as tf

import utils
from model import FontRNN, get_default_hparams, copy_hparams


tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass key=value pairs such as '
    '\'data_dir=../data, batch_size=128\' '
    '(no whitespace) to be read into the HParams object defined in model.py')


def reset_graph():
    """Close current graph and reset graph."""
    sess = tf.get_default_session()
    if sess:
        sess.close()
    tf.reset_default_graph()


def load_env(data_dir, model_dir):
    """Loads environment for inference mode, used in jupyter notebook."""
    model_params = get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_params.parse_json(f.read())
    return load_dataset(data_dir, model_params, testing_mode=True)


def load_model(model_dir):
    """Loads model for inference mode, used in jupyter notebook."""
    model_params = get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_params.parse_json(f.read())

    model_params.batch_size = 1  # only sample one at a time
    eval_model_params = copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 0
    sample_model_params = copy_hparams(eval_model_params)
    sample_model_params.max_seq_len = 1  # sample one point at a time
    return [model_params, eval_model_params, sample_model_params]


def load_dataset(data_dir, model_params, testing_mode=False):
    """Loads the .npz file, and splits the set into train/valid/test."""
    # normalizes the x and y columns using scale_factor.

    dataset = model_params.data_set
    data_filepath = os.path.join(data_dir, dataset)
    data = np.load(data_filepath, allow_pickle=True, encoding='latin1')

    # target data 
    train_strokes = data['train']
    valid_strokes = data['valid']
    test_strokes = data['test']
    all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
    
    # standard data (reference data in paper)
    std_train_strokes = data['std_train'] 
    std_valid_strokes = data['std_valid']
    std_test_strokes = data['std_test']
    all_std_trokes = np.concatenate((std_train_strokes, std_valid_strokes, std_test_strokes))

    print('Dataset combined: %d (train=%d/validate=%d/test=%d)' % (
        len(all_strokes), len(train_strokes), len(valid_strokes),len(test_strokes)))


    # calculate the max strokes we need.
    max_seq_len = utils.get_max_len(all_strokes) 
    max_std_seq_len = utils.get_max_len(all_std_trokes)
    # overwrite the hps with this calculation.
    model_params.max_seq_len = max(max_seq_len, max_std_seq_len)
    print('model_params.max_seq_len set to %d.' % model_params.max_seq_len)

    eval_model_params = copy_hparams(model_params)
    eval_model_params.rnn_dropout_keep_prob = 1.0
    eval_model_params.is_training = True

    if testing_mode:  # for testing 
        eval_model_params.batch_size = 1
        eval_model_params.is_training = False # sample mode


    train_set = utils.DataLoader(
        train_strokes,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor,
        augment_stroke_prob=model_params.augment_stroke_prob)
    normalizing_scale_factor = model_params.scale_factor
    train_set.normalize(normalizing_scale_factor)  

    valid_set = utils.DataLoader(
        valid_strokes,
        eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    valid_set.normalize(normalizing_scale_factor)

    test_set = utils.DataLoader(
        test_strokes,
        eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    test_set.normalize(normalizing_scale_factor)

    # process the reference dataset
    std_train_set = utils.DataLoader(
        std_train_strokes,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor,
        augment_stroke_prob=model_params.augment_stroke_prob)
    std_train_set.normalize(normalizing_scale_factor) 

    std_valid_set = utils.DataLoader(
        std_valid_strokes,
        eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    std_valid_set.normalize(normalizing_scale_factor)

    std_test_set = utils.DataLoader(
        std_test_strokes,
        eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    std_test_set.normalize(normalizing_scale_factor)

    result = [
        train_set, valid_set, test_set,
        std_train_set, std_valid_set, std_test_set,
        model_params, eval_model_params
    ]
    return result


def evaluate_model(sess, model, data_set, std_data_set):
    """Returns the avg validate loss"""
    avg_Loss, avg_Ld, avg_Lc = 0.0, 0.0, 0.0
    for batch in range(data_set.num_batches):
        unused_orig_x, x, s = data_set.get_batch(batch)
        unused_orig_std_x, std_x, std_s = std_data_set.get_batch(batch)

        feed = {model.enc_input_data: std_x,
                model.enc_seq_lens: std_s,
                model.dec_input_data: x,
                model.dec_seq_lens: s}
        (Loss, Ld, Lc) = sess.run([model.Loss, model.Ld, model.Lc], feed)
        avg_Loss += Loss
        avg_Ld += Ld
        avg_Lc += Lc

    bn = data_set.num_batches
    avg_Loss /= bn
    avg_Ld /= bn
    avg_Lc /= bn
    return (avg_Loss, avg_Ld, avg_Lc)


def load_checkpoint(sess, checkpoint_path):
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('Loading model %s.' % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


def save_model(sess, saver, model_save_path, global_step):
    checkpoint_path = os.path.join(model_save_path, 'vector')
    print('Saving model to %s...' % checkpoint_path)
    saver.save(sess, checkpoint_path, global_step=global_step)
    print("Done.")


def train(sess, model, eval_model, train_set, valid_set, std_train_set, std_valid_set):
    """Train a FontRNN model."""
    # Setup summary writer.
    train_summary_writer = tf.summary.FileWriter(model.hps.log_root + "/train_log", sess.graph)
    valid_summary_writer = tf.summary.FileWriter(model.hps.log_root + "/valid_log")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # conut the trainable parameters
    t_vars = tf.trainable_variables()
    count_t_vars = 0
    for var in t_vars:
        num_param = np.prod(var.get_shape().as_list())
        count_t_vars += num_param
        print('%s %s %i' % (var.name, str(var.get_shape()), num_param))
    print('Total trainable variables %d.' % count_t_vars)

    best_valid_cost = 100000000.0  # set a large init value
    hps = model.hps 
    for step in range(hps.num_steps):
        start = time.time()
        curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                              (hps.decay_rate) ** step + hps.min_learning_rate)

        # get batch data
        idx = np.random.permutation(range(0, len(train_set.strokes)))[0:hps.batch_size]
        _, x, s = train_set._get_batch_from_indices(idx) # target data
        _, std_x, std_s = std_train_set._get_batch_from_indices(idx) # reference data

        feed = {
            model.enc_input_data: std_x,  
            model.enc_seq_lens: std_s,
            model.dec_input_data: x,
            model.dec_seq_lens: s, 
            model.lr: curr_learning_rate,
        }

        # train
        (_, Ld, Lc, Loss, summ_str) = sess.run([model.train_op, model.Ld, model.Lc, model.Loss, model.summ],feed)
        train_summary_writer.add_summary(summ_str, step)
        train_summary_writer.flush()

        # print log
        if step % 10 == 0 and step > 0:
            print("Train step %d, lr:%.6f, Ld: %.4f, Lc: %.4f, Loss: %.4f; Step time: %.4fs" % (
                step, curr_learning_rate, Ld, Lc, Loss, time.time() - start))

        # validation
        if step % hps.save_every == 0 and step > 0:
            print(model.hps.log_root, " validating...")
            (valid_Loss, valid_Ld, valid_Lc) = evaluate_model(sess, eval_model, valid_set, std_valid_set)

            valid_Loss_summ = tf.summary.Summary()
            valid_Loss_summ.value.add(tag='valid_Loss', simple_value=float(valid_Loss))
            valid_Ld_summ = tf.summary.Summary()
            valid_Ld_summ.value.add(tag='valid_Ld', simple_value=float(valid_Ld))
            valid_Lc_summ = tf.summary.Summary()
            valid_Lc_summ.value.add(tag='valid_Lc', simple_value=float(valid_Lc))

            print("Best valid Loss: %.4f, curr Loss: %.4f, Ld: %.4f, Lc: %.4f" % (
                min(best_valid_cost, valid_Loss), valid_Loss, valid_Ld, valid_Lc))

            valid_summary_writer.add_summary(valid_Loss_summ, step)
            valid_summary_writer.add_summary(valid_Ld_summ, step)
            valid_summary_writer.add_summary(valid_Lc_summ, step)
            valid_summary_writer.flush()

            # better valid cost: saving model
            if valid_Loss < best_valid_cost:
                best_valid_cost = valid_Loss
                save_model(sess, saver, model.hps.log_root, step)


def trainer(model_params):
    """Train a FontRNN model."""
    print('Training a FontRNN model:')
    print('Hyperparams:')
    for key, val in six.iteritems(model_params.values()):
        print('%s = %s' % (key, str(val)))
    print('Loading data files:')
    datasets = load_dataset(model_params.data_dir, model_params)

    train_set = datasets[0]
    valid_set = datasets[1]
    test_set = datasets[2]
    std_train_set = datasets[3]
    std_valid_set = datasets[4]
    std_test_set = datasets[5]

    model_params = datasets[6]
    eval_model_params = datasets[7]

    reset_graph()
    model = FontRNN(model_params)

    eval_model = FontRNN(eval_model_params, reuse=True)
    assert eval_model.hps.rnn_dropout_keep_prob == 1.0

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Write config file to json file.
    tf.gfile.MakeDirs(model_params.log_root)
    with tf.gfile.Open(os.path.join(model_params.log_root, 'model_config.json'), 'w') as f:
        json.dump(model_params.values(), f, indent=True)
    train(sess, model, eval_model, train_set, valid_set, std_train_set, std_valid_set)


def main(unused_argv):
    """Load model params, save config file and start training."""
    model_params = get_default_hparams()  # load defualt params
    if FLAGS.hparams:
        model_params.parse(FLAGS.hparams)  # reload params
    trainer(model_params)


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
