from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

import os
import sys
import numpy as np
import scipy
import scipy.misc

from capture import Im2TxtShow

from ambient.tools.utils import CVStreamFactory, CVModel, CVWriter

import math
import os
import sys


import tensorflow as tf
import cv2

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

# CUSTOM
from ambient.tools.utils import CVStreamFactory, CVModel


if __name__ == "__main__":
    '''Process video file or camera stream
    '''
    identifier = 0
    FLAGS = tf.flags.FLAGS

    tf.flags.DEFINE_string("checkpoint_path", "",
                           "Model checkpoint file or directory containing a "
                           "model checkpoint file.")
    tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
    tf.flags.DEFINE_string("input_files", "",
                           "File pattern or comma-separated list of file patterns "
                           "of image files.")
    if FLAGS.input_files:
        identifier = FLAGS.input_files
        print("Reading video from: {}".format(identifier))
    
    im2txt_model = Im2TxtShow(FLAGS.checkpoint_path, FLAGS.vocab_file)
    output_file = '/home/vikesh/data/ambient/bbs/im2txt.bin'
    cv_writer = CVWriter(im2txt_model, output_file)
    # cv_writer.write(identifier, resume=True, save_every=100, frame_range=(2520, 3900))
    
    # Test reconstruct
    cv_writer.reconstruct(identifier, video_file='/home/vikesh/data/ambient/videos/im2txt.avi', fps=60.0, dim=(1920, 1080))
