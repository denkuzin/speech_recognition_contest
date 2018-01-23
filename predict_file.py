from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import input_data
import models
from tensorflow.python.framework import graph_util

from tqdm import tqdm
from os.path import join, basename
import pandas as pd
import datetime

FLAGS = None

def read_all_data(path):
    """
    upload all data in binary format
    :param path:
    :return: data list and short names of files
    """
    full_pathes = [join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
    short_names = [basename(f) for f in full_pathes]
    data = []
    for full_path in tqdm(full_pathes[:5]):
        with open(full_path, 'rb') as wav_file:
            wav_data = wav_file.read()
            data.append(wav_data)
    return data, short_names


def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def create_inference_graph(wanted_words, sample_rate, clip_duration_ms,
                           clip_stride_ms, window_size_ms, window_stride_ms,
                           dct_coefficient_count, model_architecture):

    graph = tf.Graph()
    with graph.as_default():
        words_list = input_data.prepare_words_list(wanted_words.split(','))
        model_settings = models.prepare_model_settings(
            len(words_list), sample_rate, clip_duration_ms, window_size_ms,
            window_stride_ms, dct_coefficient_count)
        runtime_settings = {'clip_stride_ms': clip_stride_ms}

        wav_data_placeholder = tf.placeholder(
            tf.string, [], name='wav_data')
        decoded_sample_data = contrib_audio.decode_wav(
            wav_data_placeholder,
            desired_channels=1,
            desired_samples=model_settings['desired_samples'],
            name='decoded_sample_data')
        spectrogram = contrib_audio.audio_spectrogram(
            decoded_sample_data.audio,
            window_size=model_settings['window_size_samples'],
            stride=model_settings['window_stride_samples'],
            magnitude_squared=True)
        fingerprint_input = contrib_audio.mfcc(
            spectrogram,
            decoded_sample_data.sample_rate,
            dct_coefficient_count=dct_coefficient_count)
        fingerprint_frequency_size = model_settings['dct_coefficient_count']
        fingerprint_time_size = model_settings['spectrogram_length']
        reshaped_input = tf.reshape(fingerprint_input, [
            -1, fingerprint_time_size * fingerprint_frequency_size
        ])

        logits = models.create_model(
            reshaped_input, model_settings, model_architecture,
            is_training=False, runtime_settings=runtime_settings)

        # Create an output to use for inference.
        tf.nn.softmax(logits, name='labels_softmax')
    return graph


def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)

    # save current configuration:
    for key, value in FLAGS.__dict__.items():
        tf.logging.info(" {} = {}".format(key, value))

    data, short_names = read_all_data(FLAGS.data2predict)
    labels_list = load_labels(FLAGS.labels)

    graph = create_inference_graph(
            FLAGS.wanted_words, FLAGS.sample_rate,
            FLAGS.clip_duration_ms, FLAGS.clip_stride_ms,
            FLAGS.window_size_ms, FLAGS.window_stride_ms,
            FLAGS.dct_coefficient_count, FLAGS.model_architecture)

    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        models.load_variables_from_checkpoint(sess, FLAGS.model_checkpoint)
        output_layer_name = "labels_softmax:0"
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
        res = []
        for i, binary in tqdm(enumerate(data), total=len(data)):
            predictions = sess.run(softmax_tensor,
                                   feed_dict={'wav_data:0': binary})
            predictions = predictions[0]  # only 1 element in batch
            imd_max = predictions.argmax()
            human_string = labels_list[imd_max]
            file_name = short_names[i]
            score = predictions[imd_max]
            res.append((file_name, human_string, score))
    df = pd.DataFrame(res, columns=['name','pred','score'])
    df.loc[:, 'pred'] = df.pred.apply(lambda x: x.strip('_'))
    df.to_csv(FLAGS.output_file, sep=',', columns=['name', 'pred'], index=False,
                      header=['fname', 'label'])
    tf.logging.info('Predictions are saved to %s', FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs', )
    parser.add_argument(
        '--clip_stride_ms',
        type=int,
        default=30,
        help='How often to run recognition. Useful for models with cache.', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long the stride is between spectrogram timeslices', )
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint', )
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        default='/home/dkuzin/files/tensorflow_speech_recognition/saved_models/conv.ckpt-9',
        help='What model to use?')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='conv',
        help='What model architecture to use')
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)')
    parser.add_argument(
        '--output_file', type=str, help='Where to save the frozen graph.')
    parser.add_argument(
        '--data2predict',
        type=str,
        default='/home/dkuzin/files/tensorflow_speech_recognition/test/audio',
        help='Words to use (others will be added to an unknown label)')
    parser.add_argument(
        '--labels', type=str, default='', help='Path to file containing labels.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
