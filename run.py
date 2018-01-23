#Configuration
from argparse import Namespace
from os.path import join
import train
from itertools import product
from collections import OrderedDict
import importlib
import tensorflow as tf
from distutils.dir_util import copy_tree

FLAGS = Namespace()

calc_number = 4
FLAGS.train_dir = ('/home/dkuzin/files/tensorflow_speech_recognition/'
                      'saved_models/calc_{}'.format(calc_number))
FLAGS.do_train = True  # False in case you want to do only predict
FLAGS.start_checkpoint = ""
FLAGS.model_architecture = "conv3"
FLAGS.batch_size = 100
FLAGS.background_volume = (0.1, 0.3) 
FLAGS.background_frequency = (0.4, 0.8) 
FLAGS.silence_percentage = 10.0
FLAGS.unknown_percentage = (10.0, 20.0)
FLAGS.time_shift_ms = 100   
FLAGS.testing_percentage = 5
FLAGS.validation_percentage = 5
FLAGS.sample_rate = 16000
FLAGS.clip_duration_ms = 1000
FLAGS.window_size_ms = 30.0
FLAGS.window_stride_ms = 10.0
FLAGS.dct_coefficient_count = 40
FLAGS.how_many_training_steps = '25000,25000'
FLAGS.learning_rate = '0.001,0.0001'
FLAGS.eval_step_interval = 500
FLAGS.summaries_dir = join(FLAGS.train_dir, 'board')
FLAGS.wanted_words = 'yes,no,up,down,left,right,on,off,stop,go'
FLAGS.save_step_interval = 500
FLAGS.check_nans = False
FLAGS.dropout_prob = (0.6, 0.9)
FLAGS.is_debug = False
FLAGS.data_url = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
FLAGS.data_dir = '/home/dkuzin/files/tensorflow_speech_recognition/train/audio'
FLAGS.predict_dir = '/home/dkuzin/files/tensorflow_speech_recognition/test/audio'


# implement custom greed search:

#remember FLAGS.train_dir:
FLAGS_train_dir_basic = FLAGS.train_dir

num_calculations = 1
attr_to_iterate = OrderedDict()
for key, value in FLAGS.__dict__.items():
    if type(value) == tuple:
        attr_to_iterate[key] = value
        num_calculations *= len(value)
if attr_to_iterate:
    combinations = list(product(*attr_to_iterate.values()))
    print("The RUN requeres {} calculations".format(num_calculations))
    print(f"combinations = {combinations}")
    for i, combination in enumerate(combinations):
        #change FLAGS
        for attr, value in zip(list(attr_to_iterate), combination):
            FLAGS.__setattr__(attr, value)
        FLAGS.train_dir = FLAGS_train_dir_basic + '/{}'.format(i)

        train.FLAGS = FLAGS
        open('nohup.out', 'w').close() # clear nohup.out file
        train.main("let's do it, mf")
        _ = copy_tree('../tensorflow_speech_recognition', FLAGS.train_dir + "/tensorflow_speech_recognition")
        tf.logging.info("the all scripts is copied to {}".format(FLAGS.train_dir + "/tensorflow_speech_recognition"))

else:  #in case if there is only 1 combination
    train.FLAGS = FLAGS
    open('nohup.out', 'w').close()  # clear nohup.out file
    train.main("let's do it, mf")
    _ = copy_tree('../tensorflow_speech_recognition', FLAGS.train_dir + "/tensorflow_speech_recognition")
    tf.logging.info("the all scripts is copied to {}".format(FLAGS.train_dir + "/tensorflow_speech_recognition"))

