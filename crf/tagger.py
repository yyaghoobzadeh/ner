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
import nltk


if __name__ == '__main__':
    # Read parameters from command line
    optparser = optparse.OptionParser()
    optparser.add_option(
        "-T", "--train", default="",
        help="Train set location"
    )
    optparser.add_option(
        "-d", "--dev", default="",
        help="Dev set location"
    )
    optparser.add_option(
        "-t", "--test", default="",
        help="Test set location"
    )
    optparser.add_option(
        "-s", "--tag_scheme", default="iob",
        help="Tagging scheme (IOB or IOBES)"
    )
    optparser.add_option(
        "-l", "--lower", default="1",
        type='int', help="Lowercase words (this will not affect character inputs)"
    )
    optparser.add_option(
        "-z", "--zeros", default="1",
        type='int', help="Replace digits with 0"
    )
    optparser.add_option(
        "-p", "--pre_emb", default="",
        help="Location of pretrained embeddings"
    )
    optparser.add_option(
        "-a", "--cap_dim", default="0",
        type='int', help="Capitalization feature dimension (0 to disable)"
    )
    optparser.add_option(
        "-f", "--crf", default="1",
        type='int', help="Use CRF (0 to disable)"
    )
    opts = optparser.parse_args()[0]

    # Parse parameters
    parameters = OrderedDict()
    parameters['tag_scheme'] = opts.tag_scheme
    parameters['lower'] = opts.lower == 1
    parameters['zeros'] = opts.zeros == 1
    parameters['pre_emb'] = opts.pre_emb
    parameters['cap_dim'] = opts.cap_dim
    parameters['crf'] = opts.crf == 1
    
    # Check parameters validity
    assert os.path.isfile(opts.train)
    assert os.path.isfile(opts.dev)
    assert os.path.isfile(opts.test)
    assert parameters['tag_scheme'] in ['iob', 'iobes']
#     assert not parameters['pre_emb'] or parameters['word_dim'] > 0

        # Check evaluation script / folders
    if not os.path.isfile(eval_script):
        raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
    if not os.path.exists(eval_temp):
        os.makedirs(eval_temp)
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    # Data parameters
    lower = parameters['lower']
    zeros = parameters['zeros']
    tag_scheme = parameters['tag_scheme']
        
    train_sents = loader.load_sentences(opts.train, lower, zeros)
    dev_sents = loader.load_sentences(opts.dev, lower, zeros)
    test_sents = loader.load_sentences(opts.test, lower, zeros)
    
    nltk.download('averaged_perceptron_tagger')
    train_sents = add_pos_tags(train_sents)
    dev_sents = add_pos_tags(dev_sents)
    test_sents = add_pos_tags(test_sents)
    train_sents.extend(dev_sents)
    
    crf_ner_obj = CRF_NER()
    crf_ner_obj.build_datasets(train_sents, test_sents)
    crf_ner_obj.train()
    crf_ner_obj.evaluate(test_sents)
    crf_ner_obj.tag(test_sents, 'test_outputs')
    
    
    
    

    
    


    
    
    
    
    
    
    