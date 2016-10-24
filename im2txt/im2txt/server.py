from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import random

import tensorflow as tf
import cv2

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

from ambient.serving.interface import CVServer
from ambient.serving.resource import ResourceHandler

# CUSTOM
from ambient.serving.interface import CVModel


class Im2TxtModel(CVModel):
    def __init__(self, checkpoint_path, vocab_file):
        # Build the inference graph.
        g = tf.Graph()
        with g.as_default():
            model = inference_wrapper.InferenceWrapper()
            restore_fn = model.build_graph_from_config(configuration.ModelConfig(), checkpoint_path)

        g.finalize()

        # Create the vocabulary.
        self.vocab = vocabulary.Vocabulary(vocab_file)
        self.sess = tf.Session(graph=g)

        # Load the model from checkpoint.
        restore_fn(self.sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        self.generator = caption_generator.CaptionGenerator(model, self.vocab)
        self.last_caption = None

    def get(self, encoded_image):
        '''Inference from the binary data
        '''
        captions = self.generator.beam_search(self.sess, encoded_image)

        # Find most likely caption
        sorted_captions = sorted(captions, key=lambda caption: math.exp(caption.logprob), reverse=True)
        # TODO: REMOVE
        '''
        for i, caption in enumerate(captions)
            # Ignore begin and end words.
            sentence = [self.vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            cv2.rectangle(frame, (50, 50), (100, 1000), (125,125,125), -1)
            cv2.putText(frame, sentence + ' : %.4f' % math.exp(caption.logprob), (50, 10*i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        '''
        caption = sorted_captions[0]
        sentence = [self.vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        return sentence


    def inference_fn(self, request, context):
        return self.get(request.input)

if __name__ == "__main__":
    rh = ResourceHandler('im2txt', 'default')
    checkpoint_path = rh.get('model.ckpt')
    vocab_file = rh.get('vocab')

    im2txt_model = Im2TxtModel(checkpoint_path, vocab_file)
    CVServer('localhost', 9003).start(im2txt_model.inference_fn)
