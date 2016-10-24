# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Generate captions for images using default beam search parameters."""

import math
import os
import sys
import random

import tensorflow as tf
import cv2

# CUSTOM
from ambient.tools.utils import CVStreamFactory
from repository.models.im2txt.im2txt.models import Im2TxtStubModel

# PERF: Press this key to toggle recognition
REC_KEY = 'r'

def main(_):
    '''Process video file or camera stream
    '''
    should_recognize = True
    identifier = 0
    
    if FLAGS.input_files:
        identifier = FLAGS.input_files
    
    print("Reading video from: {}".format(identifier))
    streamer = CVStreamFactory.get(identifier)
    # Must match server.py
    im2txt_model = Im2TxtStubModel('localhost', 9003)


    # Create model
    for ret, frame in streamer.read():
        if ret:
            # PERF: Run expensive ops in recognition mode (for interesting parts in video)
            if not identifier or should_recognize:
                im2txt_model.make(frame)

            cv2.imshow('Video', frame)
        else:
            # ret is false
            print("Failed to read frame: {}".format(frame))
            # break

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord(REC_KEY):
            should_recognize = not should_recognize
            print("Switching to should_recognize:{}".format(should_recognize))

    streamer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    FLAGS = tf.flags.FLAGS

    tf.flags.DEFINE_string("checkpoint_path", "",
                           "Model checkpoint file or directory containing a "
                           "model checkpoint file.")
    tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
    tf.flags.DEFINE_string("input_files", "",
                           "File pattern or comma-separated list of file patterns "
                       "of image files.")
    tf.app.run()
