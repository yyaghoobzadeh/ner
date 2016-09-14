#!/usr/bin/env python

import os
import time
import codecs
import optparse
import numpy as np
from loader import prepare_sentence
from utils import create_input, zero_digits
from model_newdomain import Model


optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model_newdomain", default="",
    help="Model location"
)
opts = optparser.parse_args()[0]
assert os.path.isdir(opts.model)

# Load existing model_newdomain
print "Loading model_newdomain..."
model = Model(model_path=opts.model)
parameters = model.parameters

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

# Load the model_newdomain
_, f_eval = model.build(training=False, **parameters)
model.reload()


while True:
    sent = raw_input("Type a query (type \"exit\" to exit):\n")
    count = 0
    words = sent.rstrip().split()
    if sent == 'exit':
        break
    else:
        # Lowercase sentence
        if parameters['lower']:
            sent = sent.lower()
        # Replace all digits with zeros
        if parameters['zeros']:
            sent = zero_digits(sent)
            # Prepare input
        sentence = prepare_sentence(words, word_to_id, char_to_id,
                                        lower=parameters['lower'])
        input = create_input(sentence, parameters, False)
            # Decoding
        y_preds = f_eval(*input).argmax(axis=1)
        y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
            # Output tags in the IOB2 format
            # Write tags
        assert len(y_preds) == len(words)
        print('%s\n\n' % '\n'.join('%s%s%s' % (w, " ", y)
                                             for w, y in zip(words, y_preds)))

print '---- %i lines tagged in %.4fs ----' % (count, time.time() - start)
f_output.close()
