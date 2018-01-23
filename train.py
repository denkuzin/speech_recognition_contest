# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import numpy as np
from six.moves import xrange
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile
import time
import pandas as pd
import datetime
from utils import send2telegramm


data_index = 0
epoch_number = 1
def next_batch(batch_size, X, y):
    global data_index, epoch_number
    to_shuffle = 0
    batch = X[data_index:data_index + batch_size, :]
    labels = y[data_index:data_index + batch_size]
    if data_index + batch_size > len(X):
        tf.logging.info("{} epoch is done".format(epoch_number))
        epoch_number += 1
        to_shuffle = 1
    data_index = (data_index + batch_size) % len(X)
    return (batch, labels, to_shuffle)


def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def main(_):
  global data_index, epoch_number

  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.is_debug:
      FLAGS.learning_rate = '0.001'
      FLAGS.how_many_training_steps = ','.join(['3']*len(FLAGS.learning_rate.split(',')))
      FLAGS.testing_percentage = 0.01
      FLAGS.validation_percentage = 0.01
      FLAGS.eval_step_interval = 2
      FLAGS.save_step_interval = 2
      FLAGS.train_dir = "/home/dkuzin/files/tensorflow_speech_recognition/saved_models/debug"
      FLAGS.summaries_dir = "/home/dkuzin/files/tensorflow_speech_recognition/saved_models/debug/board"

  if FLAGS.do_train is False:
      assert FLAGS.start_checkpoint != "", "please, specify FLAGS.start_checkpoint"


  #save current configuration:
  for key,value in FLAGS.__dict__.items():
      tf.logging.info(" {} = {}".format(key, value))

  # Start a new TensorFlow session.
  config = tf.ConfigProto(log_device_placement=False,
          intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
  config.gpu_options.allow_growth = True
  sess = tf.InteractiveSession(config=config)

  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)

  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings, predict_dir=FLAGS.predict_dir)
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))
  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  logits, dropout_prob = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=True)

  # Define loss and optimizer
  ground_truth_input = tf.placeholder(
      tf.int64, [None], name='groundtruth_input')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate_input).minimize(cross_entropy_mean)

  confes = tf.reduce_max(tf.nn.softmax(logits), reduction_indices=[1])
  predicted_indices = tf.argmax(logits, 1)
  correct_prediction = tf.equal(predicted_indices, ground_truth_input)
  confusion_matrix = tf.confusion_matrix(
      ground_truth_input, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)

  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  saver = tf.train.Saver(tf.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                       sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
  tf.global_variables_initializer().run()
  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)

  tf.logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                       FLAGS.model_architecture + '.pbtxt')

  # Save list of words.
  with gfile.GFile(
      os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
      'w') as f:
    f.write('\n'.join(audio_processor.words_list))


  # Upload all procesed train data
  # (it speeds up calculations)
  tf.logging.info('start to upload train data')
  t = time.time()
  if FLAGS.is_debug:
      num_examles = 3*FLAGS.batch_size
  else:
      num_examles = -1 # means upload full dataset
  train_fingerprints_all, train_ground_truth_all = audio_processor.get_data(
      num_examles , 0, model_settings, FLAGS.background_frequency,
      FLAGS.background_volume, time_shift_samples, 'training', sess)
  tf.logging.info('train data is uploaded, {:.2f} sec'.format(time.time()-t))


  # Training loop.
  training_steps_max = np.sum(training_steps_list)
  if FLAGS.do_train is False:
      training_steps_max = 0 # skip training part
  for training_step in xrange(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    training_steps_sum = 0
    for i in range(len(training_steps_list)):
      training_steps_sum += training_steps_list[i]
      if training_step <= training_steps_sum:
        learning_rate_value = learning_rates_list[i]
        break
    # Pull the audio samples we'll use for training.
    train_fingerprints, train_ground_truth, to_shuffle = next_batch(
                    FLAGS.batch_size, train_fingerprints_all, train_ground_truth_all)
    if to_shuffle:
        p = np.random.permutation(len(train_fingerprints_all))
        train_fingerprints_all = train_fingerprints_all[p]
        train_ground_truth_all = train_ground_truth_all[p]
    # Run the graph with this batch of training data.
    train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        [
            merged_summaries, evaluation_step, cross_entropy_mean, train_step,
            increment_global_step
        ],
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            dropout_prob: FLAGS.dropout_prob
        })
    train_writer.add_summary(train_summary, training_step)
    tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                    (training_step, learning_rate_value, train_accuracy * 100,
                     cross_entropy_value))
    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      # without imput preprocessing
      tf.logging.info("validation on NOT preprocessed data")
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_conf_matrix = None
      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                     0.0, 0, 'validation', sess))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy, conf_matrix = sess.run(
            [merged_summaries, evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: validation_fingerprints,
                ground_truth_input: validation_ground_truth,
                dropout_prob: 1.0
            })
        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
        else:
          total_conf_matrix += conf_matrix
      tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
      tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, set_size))
      send2telegramm('Step %d: Validation accuracy without input preproc = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, set_size))

    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      # with input preprocessing
      tf.logging.info("validation on preprocessed data")
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_conf_matrix = None
      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, FLAGS.background_frequency,
                    FLAGS.background_volume, time_shift_samples, 'validation', sess))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy, conf_matrix = sess.run(
            [merged_summaries, evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: validation_fingerprints,
                ground_truth_input: validation_ground_truth,
                dropout_prob: 1.0
            })
        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
        else:
          total_conf_matrix += conf_matrix
      tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
      tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, set_size))
      send2telegramm('Step %d: Validation accuracy, with input proc = %.1f%% (N=%d)' %
                   (training_step, total_accuracy * 100, set_size))

    # Save the model checkpoint periodically.
    if (training_step % FLAGS.save_step_interval == 0 or is_last_step):
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     FLAGS.model_architecture + '.ckpt')
      tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)

  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  if FLAGS.do_train is False:
      set_size=0
  for i in xrange(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
            dropout_prob: 1.0
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                           set_size))


  # PREDICT with NO preprocessing
  set_size = audio_processor.set_size('predict')
  tf.logging.info('predict set_size=%d', set_size)

  res = []
  if FLAGS.is_debug:
      set_size = 3*FLAGS.batch_size
  for i in xrange(0, set_size, FLAGS.batch_size):
    predict_fingerprints, file_names = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'predict', sess)
    confidences, predictions = sess.run([confes, predicted_indices],
             feed_dict={
                 fingerprint_input: predict_fingerprints,
                 #ground_truth_input: predict_ground_truth,
                 dropout_prob: 1.0
             })
    short_file_names = [os.path.basename(name) for name in file_names]
    human_predictions = [audio_processor.words_list[ind] for ind in predictions.flatten()]

    if i == 0: #check
        tf.logging.info("short_file_names = {}".format(short_file_names))
        tf.logging.info("human_predictions = {}".format(human_predictions))
        tf.logging.info("confidences = {}".format(confidences))

    res.extend(list(zip(short_file_names, human_predictions, confidences)))
  #atleast, save all
  df = pd.DataFrame(res, columns=['name', 'pred', 'score'])
  df.loc[:, 'pred'] = df.pred.apply(lambda x: x.strip('_'))
  suffix = datetime.datetime.today().isoformat().split(':')[0]
  path_to_save_pred_1 = os.path.join(FLAGS.train_dir, 'prediction_for_submit_{}.txt'.format(suffix))
  path_to_save_pred_2 = os.path.join(FLAGS.train_dir, 'prediction_with_score_{}.txt'.format(suffix))
  df.to_csv(path_to_save_pred_1, sep=',',
            columns=['name', 'pred'], index=False,
            header=['fname', 'label'])
  df.to_csv(path_to_save_pred_2, sep=',',
            columns=['name', 'pred', 'score'], index=False,
            header=['fname', 'label', 'score'])


  # PREDICT with preprocessing
  set_size = audio_processor.set_size('predict')
  tf.logging.info('predict set_size=%d', set_size)

  res = []
  if FLAGS.is_debug:
      set_size = 3*FLAGS.batch_size
  for i in xrange(0, set_size, FLAGS.batch_size):
    predict_fingerprints, file_names = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, time_shift_samples, 'predict', sess)
    confidences, predictions = sess.run([confes, predicted_indices],
             feed_dict={
                 fingerprint_input: predict_fingerprints,
                 #ground_truth_input: predict_ground_truth,
                 dropout_prob: 1.0
             })
    short_file_names = [os.path.basename(name) for name in file_names]
    human_predictions = [audio_processor.words_list[ind] for ind in predictions.flatten()]

    if i == 0: #check
        tf.logging.info("short_file_names = {}".format(short_file_names))
        tf.logging.info("human_predictions = {}".format(human_predictions))
        tf.logging.info("confidences = {}".format(confidences))

    res.extend(list(zip(short_file_names, human_predictions, confidences)))
  #atleast, save all
  df = pd.DataFrame(res, columns=['name', 'pred', 'score'])
  df.loc[:, 'pred'] = df.pred.apply(lambda x: x.strip('_'))
  suffix = datetime.datetime.today().isoformat().split(':')[0]
  path_to_save_pred_1 = os.path.join(FLAGS.train_dir,
                'prediction_for_submit_with_preprocess{}.txt'.format(suffix))
  path_to_save_pred_2 = os.path.join(FLAGS.train_dir,
                'prediction_with_score_with_preprocess{}.txt'.format(suffix))
  df.to_csv(path_to_save_pred_1, sep=',',
            columns=['name', 'pred'], index=False,
            header=['fname', 'label'])
  df.to_csv(path_to_save_pred_2, sep=',',
            columns=['name', 'pred', 'score'], index=False,
            header=['fname', 'label', 'score'])

  sess.close()
  tf.reset_default_graph()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/home/dkuzin/files/tensorflow_speech_recognition/train/audio',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='15000,3000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/home/dkuzin/files/tensorflow_speech_recognition/saved_models/calc_defaut',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv3',
      help='What model architecture to use')
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  parser.add_argument(
      '--dropout_prob',
      type=float,
      default=0.5,
      help='dropout probability for all layers')
  parser.add_argument(
      '--is_debug',
      type=bool,
      default=False,
      help='rum mode')
  parser.add_argument(
      '--do_train',
      type=bool,
      default=True,
      help='rum mode')
  parser.add_argument(
      '--predict_dir',
      type=str,
      default='/home/dkuzin/files/tensorflow_speech_recognition/test/audio',
      help='Directory of filed for prediction')



  FLAGS, unparsed = parser.parse_known_args()
  if unparsed:
      raise Exception('Check unparsed args, I found this one' + str(unparsed))
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)