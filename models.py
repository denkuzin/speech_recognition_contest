import tensorflow as tf
import math

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
    """Calculates common settings needed for all models.
    Returns:
    Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
    }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
    """
    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
    """
    if model_architecture == 'single_fc':
        return create_single_fc_model(fingerprint_input, model_settings,
                                      is_training)
    elif model_architecture == 'conv':
        return create_conv_model(fingerprint_input, model_settings, is_training)
    elif model_architecture == 'conv3':
        return create_conv_model_3_conv_lauers(fingerprint_input, model_settings,
                                            is_training)
    elif model_architecture == 'low_latency_conv':
        return create_low_latency_conv_model(fingerprint_input, model_settings,
                                             is_training)
    elif model_architecture == 'low_latency_svdf':
        return create_low_latency_svdf_model(fingerprint_input, model_settings,
                                             is_training, runtime_settings)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                        '" not recognized, should be one of "single_fc", "conv",' +
                        ' "low_latency_conv, or "low_latency_svdf"')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.

    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)

def create_single_fc_model(fingerprint_input, model_settings, is_training):
    pass
def create_low_latency_conv_model(fingerprint_input, model_settings, is_training):
    pass
def create_low_latency_svdf_model(fingerprint_input, model_settings, is_training):
    pass


def create_conv_model(fingerprint_input, model_settings, is_training):
    """Builds a standard convolutional model.

    This is roughly the network labeled as 'cnn-trad-fpool3' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

    Here's the layout of the graph:

    (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

    This produces fairly good quality results, but can involve a large number of
    weight parameters and computations. For a cheaper alternative from the same
    paper with slightly less accuracy, see 'low_latency_conv' below.

    During training, dropout nodes are introduced after each relu, controlled by a
    placeholder.

    Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

    Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
    """

    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = 20
    first_filter_count = 64
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    second_filter_width = 4
    second_filter_height = 10
    second_filter_count = 64
    second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
    second_bias = tf.Variable(tf.zeros([second_filter_count]))
    second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias
    second_relu = tf.nn.relu(second_conv)
    if is_training:
        second_dropout = tf.nn.dropout(second_relu, dropout_prob)
    else:
        second_dropout = second_relu
    second_conv_shape = second_dropout.get_shape()
    second_conv_output_width = second_conv_shape[2]
    second_conv_output_height = second_conv_shape[1]
    second_conv_element_count = int(
        second_conv_output_width * second_conv_output_height *
        second_filter_count)
    flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
    label_count = model_settings['label_count']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_conv_element_count, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_conv_model_3_conv_lauers(fingerprint_input, model_settings, is_training):
    """Builds a standard convolutional model.

     This is roughly the network labeled as 'cnn-trad-fpool3' in the
     'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
     http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

     Here's the layout of the graph:

     (fingerprint_input)
           v               -----------------------
       [Conv2D]<-(weights)   |
           v                 |
       [BiasAdd]<-(bias)     |
           v                 |   1 convolutional layer
         [Relu]              |
           v                 |
       [MaxPool]             |
           v               -----------------------
       [Conv2D]<-(weights)   |
           v                 |
       [BiasAdd]<-(bias)     |
           v                 |   2 convolutional layer
         [Relu]              |
           v                 |
       [MaxPool]             |
           v               ------------------------
       [Conv2D]<-(weights)   |
           v                 |
       [BiasAdd]<-(bias)     |
           v                 |   3 convolutional layer
         [Relu]              |
           v                 |
       [MaxPool]             |
           v               -------------------------
       [MatMul]<-(weights)   |
           v                 |    densely connected layer
       [BiasAdd]<-(bias)     |
           v               -------------------------

     This produces fairly good quality results, but can involve a large number of
     weight parameters and computations. For a cheaper alternative from the same
     paper with slightly less accuracy, see 'low_latency_conv' below.

     During training, dropout nodes are introduced after each relu, controlled by a
     placeholder.

     Args:
     fingerprint_input: TensorFlow node that will output audio feature vectors.
     model_settings: Dictionary of information about the model.
     is_training: Whether the model is going to be used for training.

     Returns:
     TensorFlow node outputting logits results, and optionally a dropout
     placeholder.
     """

    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 16
    first_filter_height = 40
    first_filter_count = 64
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                              'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    second_filter_width = 8
    second_filter_height = 20
    second_filter_count = 64
    second_weights = tf.Variable(
        tf.truncated_normal(
            [
                second_filter_height, second_filter_width, first_filter_count,
                second_filter_count
            ],
            stddev=0.01))
    second_bias = tf.Variable(tf.zeros([second_filter_count]))
    second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                               'SAME') + second_bias
    second_relu = tf.nn.relu(second_conv)
    if is_training:
        second_dropout = tf.nn.dropout(second_relu, dropout_prob)
    else:
        second_dropout = second_relu


    max_pool2 = tf.nn.max_pool(second_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    third_filter_width = 4
    third_filter_height = 10
    third_filter_count = 64
    third_weights = tf.Variable(
        tf.truncated_normal(
            [
                third_filter_height, third_filter_width, second_filter_count,
                third_filter_count
            ],
            stddev=0.01))
    third_bias = tf.Variable(tf.zeros([third_filter_count]))
    third_conv = tf.nn.conv2d(max_pool2, third_weights, [1, 1, 1, 1],
                               'SAME') + third_bias
    third_relu = tf.nn.relu(third_conv)
    if is_training:
        third_dropout = tf.nn.dropout(third_relu, dropout_prob)
    else:
        third_dropout = third_relu
    third_conv_shape = third_dropout.get_shape()
    third_conv_output_width = third_conv_shape[2]
    third_conv_output_height = third_conv_shape[1]
    third_conv_element_count = int(
        third_conv_output_width * third_conv_output_height *
        third_filter_count)
    flattened_third_conv = tf.reshape(third_dropout,
                                       [-1, third_conv_element_count])
    label_count = model_settings['label_count']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [third_conv_element_count, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattened_third_conv, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc