import os
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import cPickle

from utils import shared, set_values, get_name
from nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM
from optimization import Optimization


class Model_newdomain(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, models_path=None, model_path=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        self.new_id2tag = {}
        if model_path is None:
            assert parameters and models_path
            # Create a name based on the parameters
            self.parameters = parameters
            self.name = parameters['model_name']#get_name(parameters)
            # Model location
            model_path = os.path.join(models_path, self.name)
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            else:
                print "model exists: ", self.model_path
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                cPickle.dump(parameters, f)
        else:
            assert parameters is None and models_path is None
            # Model location
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = cPickle.load(f)
            self.reload_mappings()
        self.components = {}

    def save_mappings(self, id_to_word, id_to_char, id_to_tag, id_to_tag_old):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        self.id_to_tag_old = id_to_tag_old
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_char': self.id_to_char,
                'id_to_tag': self.id_to_tag,
                'id_to_tag_old': self.id_to_tag_old,
            }
            cPickle.dump(mappings, f)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = cPickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']
	if 'id_to_tag_old' in mappings:
	    self.id_to_tag_old = mappings['id_to_tag_old']
	else:
	    self.id_to_tag_old = self.id_to_tag

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self):
        """
        Write components values to disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self):
        """
        Load components values from disk.
        """
        ignore_shape = False
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if not os.path.exists(param_path):
                print "param :", param, "is not saved before to be loaded. "
                continue
            param_values = scipy.io.loadmat(param_path)
            if 'word_layer' in name: 
                ignore_shape = True
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name], ignore_size=ignore_shape)
            else:
                set_values(name, param, param_values[name], ignore_size=ignore_shape)

    def build(self,
              dropout,
              char_dim,
              char_lstm_dim,
              char_bidirect,
              word_dim,
              word_lstm_dim,
              word_bidirect,
              lr_method,
              pre_emb,
              crf,
              cap_dim,
              training=True,
              **kwargs
              ):
        """
        Build the network.
        """
        # Training parameters
        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)
        n_tags_loaded = len(self.id_to_tag_old)
        n_tags = len(self.id_to_tag)
        
        print "n_words: ", n_words, "n_chars: ", n_chars, "n_tags_loaded: ", n_tags_loaded, "n_tags(new ones): ", n_tags
        print self.id_to_tag
        print self.id_to_tag_old

        # Number of capitalization features
        if cap_dim:
            n_cap = 4

        # Network variables
        is_train = T.iscalar('is_train')
        word_ids = T.ivector(name='word_ids')
        char_for_ids = T.imatrix(name='char_for_ids')
        char_rev_ids = T.imatrix(name='char_rev_ids')
        char_pos_ids = T.ivector(name='char_pos_ids')
        tag_ids = T.ivector(name='tag_ids')
        if cap_dim:
            cap_ids = T.ivector(name='cap_ids')

        # Sentence length
        s_len = (word_ids if word_dim else char_pos_ids).shape[0]

        # Final input (all word features)
        input_dim = 0
        inputs = []

        #
        # Word inputs
        #
        if word_dim:
            input_dim += word_dim
            word_layer = EmbeddingLayer(n_words, word_dim, name='word_layer')
            word_input = word_layer.link(word_ids)
            inputs.append(word_input)
        #
        # Chars inputs
        #
        if char_dim:
            input_dim += char_lstm_dim
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')

            char_lstm_for = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_for')
            char_lstm_rev = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_rev')

            char_lstm_for.link(char_layer.link(char_for_ids))
            char_lstm_rev.link(char_layer.link(char_rev_ids))

            char_for_output = char_lstm_for.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]
            char_rev_output = char_lstm_rev.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]

            inputs.append(char_for_output)
            if char_bidirect:
                inputs.append(char_rev_output)
                input_dim += char_lstm_dim

        #
        # Capitalization feature
        #
        if cap_dim:
            input_dim += cap_dim
            cap_layer = EmbeddingLayer(n_cap, cap_dim, name='cap_layer')
            inputs.append(cap_layer.link(cap_ids))

        # Prepare final input
        if len(inputs) != 1:
            inputs = T.concatenate(inputs, axis=1)

        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            inputs = T.switch(T.neq(is_train, 0), input_train, input_test)

        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_rev')
        word_lstm_for.link(inputs)
        word_lstm_rev.link(inputs[::-1, :])
        word_for_output = word_lstm_for.h
        word_rev_output = word_lstm_rev.h[::-1, :]
        if word_bidirect:
            final_output = T.concatenate(
                [word_for_output, word_rev_output],
                axis=1
            )
            tanh_layer = HiddenLayer(2 * word_lstm_dim, word_lstm_dim,
                                     name='tanh_layer', activation='tanh')
            final_output = tanh_layer.link(final_output)
        else:
            final_output = word_for_output

        # Sentence to Named Entity tags - Score
        final_layer_init = HiddenLayer(word_lstm_dim, n_tags_loaded, name='final_layer',
                                  activation=(None))
        tags_loaded_scores = final_layer_init.link(final_output)
        
        print word_lstm_dim+n_tags_loaded
        final_layer = HiddenLayer(word_lstm_dim+n_tags_loaded, n_tags, name='final_layer_new',
                                  activation=('softmax'))
        final_out_new = T.concatenate([final_output, tags_loaded_scores], axis=1)
        tags_scores = final_layer.link(final_out_new)

        # No CRF
        cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()

        # Network parameters
        params = []
        if word_dim:
            self.add_component(word_layer)
            params.extend(word_layer.params)
        if char_dim:
            self.add_component(char_layer)
            self.add_component(char_lstm_for)
            params.extend(char_layer.params)
            params.extend(char_lstm_for.params)
            if char_bidirect:
                self.add_component(char_lstm_rev)
                params.extend(char_lstm_rev.params)
        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)
        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)
        if cap_dim:
            self.add_component(cap_layer)
            params.extend(cap_layer.params)
        self.add_component(final_layer)
        params.extend(final_layer.params)
        if word_bidirect:
            self.add_component(tanh_layer)
            params.extend(tanh_layer.params)

        # Prepare train and eval inputs
        eval_inputs = []
        if word_dim:
            eval_inputs.append(word_ids)
        if char_dim:
            eval_inputs.append(char_for_ids)
            if char_bidirect:
                eval_inputs.append(char_rev_ids)
            eval_inputs.append(char_pos_ids)
        if cap_dim:
            eval_inputs.append(cap_ids)
        train_inputs = eval_inputs + [tag_ids]

        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function
        print 'Compiling...'
        if training:
            updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters)
            f_train = theano.function(
                inputs=train_inputs,
                outputs=cost,
                updates=updates,
                givens=({is_train: np.cast['int32'](1)} if dropout else {})
            )
        else:
            f_train = None

        # Compile evaluation function
        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )

        return f_train, f_eval
