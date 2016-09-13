'''
Created on Sep 10, 2016

@author: yy1
'''
from collections import OrderedDict
from itertools import chain
import itertools
import optparse
import os

from pygments.lexer import words

from crf_ner import add_pos_tags, CRF_NER
from loader import augment_with_pretrained
from loader import update_tag_scheme, prepare_dataset
from loader import word_mapping, char_mapping, tag_mapping
import loader
import numpy as np
from utils import create_input
from utils import models_path, evaluate, eval_script, eval_temp
import nltk,pycrfsuite

def add_pos_tag(sent):
    '''
        example sents = [[u'breaking'], [u'news'], [u'dealing']..]
    '''
    words = [part[0] for part in sent]
    word_tags = nltk.pos_tag(words)
    return word_tags

if __name__ == '__main__':
    # Read parameters from command line
    
    nltk.download('averaged_perceptron_tagger')
    tagger = pycrfsuite.Tagger()
    tagger.open('model1.crfsuite')
    while True:
        sent = raw_input("Type a query (type \"exit\" to exit):\n")
        words = sent.rstrip().split()
        words_pos = add_pos_tag(words)
        
        if sent == 'exit':
            break
        if sent == "":
            continue
        
        pred_tags = tagger.tag(CRF_NER.sent2features(words_pos))
        assert len(pred_tags) == len(words)
        print '\n'
        for w, pred in zip(words, pred_tags):
            print w + " " + pred
    
    
    
    

    
    


    
    
    
    
    
    
    