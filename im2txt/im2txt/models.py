'''
NO bazel dependency
THIS FILE WILL be included by the main repo. No Bazel imports
'''
import cv2
import math

import numpy as np
import tensorflow as tf

from ambient.serving.client import TFModel
from ambient.serving.interface import CVShow, CVClient


class Im2TxtStubModel(TFModel):
    '''Create a stub TFModel to allow orchestrator to work
    '''
    # HIJACK init to not call at()
    def __init__(self, host, port):
        self.client = CVClient(host, port)
        self.frame_num = 0

    def get(self, frame):
        '''Model object is the most likely sentence
        '''
        self.frame_num += 1
        # HAX: Encode frame
        filename = '/tmp/im2txt/frame' + str(self.frame_num % 100) + '.jpg'
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Hack: Encode frame
        cv2.imwrite(filename, frame)
        
        # Read encoded file
        with tf.gfile.GFile(filename, "r") as f:
            image = f.read()
        
        return self.client.get(image)

    def set(self, sentence, frame):
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, height - 100), (width, height), (153, 122, 0), -1)
        cv2.putText(frame, sentence.title(), (5, height-75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return frame
