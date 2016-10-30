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

from ambient.search.metadata import EntityMetadata

# HOST = 'localhost'
HOST = 'ec2-54-244-39-243.us-west-2.compute.amazonaws.com'
PORT = 9003


class Im2TxtStubModel(TFModel):
    '''Create a stub TFModel to allow orchestrator to work
    '''
    # HIJACK init to not call at()
    def __init__(self):
        self.frame_num = 0

    def at(self, host, port):
        self.client = CVClient(host, port)
        return self

    def get(self, frame):
        '''Model object is the most likely sentence
        '''
        self.frame_num += 1
        # HAX: Encode frame
        # filename = '/tmp/im2txt/frame' + str(self.frame_num % 5000) + '.jpg'
        # Hack: Encode frame
        # cv2.imwrite(filename, frame)
        
        # Read encoded file
        # with tf.gfile.GFile(filename, "r") as f:
        #     image = f.read()
        
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90] 
        result, image = cv2.imencode('.jpg', frame, encode_param)
        bstr = image.tostring()
        print("Len: {} bytes".format(len(bstr)))
        assert result, 'Could not encode'
        sentence = self.client.get(bstr)
        height, width = frame.shape[:2]
        return [EntityMetadata('scene', (0, height - 50, width, 50), 
            keywords=sentence.title().split(), custom=sentence)]

    def set(self, entities, frame):
        height, width = frame.shape[:2]
        # There's only one entity for im2txt
        sentence = entities[0].custom
        x, y, w, h = entities[0].roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (153, 122, 0), -1)
        cv2.putText(frame, sentence.title(), (5, height-25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return frame

def get_model():
    return Im2TxtStubModel().at(HOST, PORT)
