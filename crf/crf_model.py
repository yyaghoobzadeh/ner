
import pycrfsuite, nltk
from itertools import chain
import codecs
import os

dirname, _ = os.path.split(os.path.abspath(__file__))

models_path = os.path.join(dirname, "models")
eval_path = os.path.join(dirname, "evaluation")
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")

def add_pos_tags(sents):
    '''
        example sents[0] = [[u'breaking', u'B-NEWSTYPE'], [u'news', u'I-NEWSTYPE'], [u'dealing', u'O'], [u'with', u'O'], [u'prime', u'B-KEYWORDS'], [u'minister', u'I-KEYWORDS'], [u'of', u'I-KEYWORDS'], [u'greece', u'I-KEYWORDS'], [u's', u'I-KEYWORDS'], [u'visit', u'I-KEYWORDS'], [u'to', u'I-KEYWORDS'], [u'tunis', u'I-KEYWORDS'], [u'in', u'O'], [u'west', u'B-PROVIDER'], [u'shore', u'I-PROVIDER'], [u'news', u'I-PROVIDER']]
    '''
    sents_posadded = []
    for sent in sents:
        words = [part[0] for part in sent]
        ner_tags = [part[1] for part in sent]
        word_tags = nltk.pos_tag(words)
        word_pos_tags = [(word_pos[0],word_pos[1],nertag) for word_pos, nertag in zip(word_tags, ner_tags)]
        sents_posadded.append(word_pos_tags)
    return sents_posadded

class CRF_NER(object):
    '''
    NER with CRF
    '''
    def __init__(self):
        '''
        '''
    
    @staticmethod
    def bio_classification_report(y_true, y_pred):
        """
        Classification report for a list of BIO-encoded sequences.
        It computes token-level metrics and discards "O" labels.
        
        Note that it requires scikit-learn 0.15+ (or a version from github master)
        to calculate averages properly!
        """
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
            
        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
        
        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels = [class_indices[cls] for cls in tagset],
            target_names = tagset,
        )
        
    
    @staticmethod
    def word2features(sent, i):
        word = sent[i][0]
        postag = sent[i][1]
        features = [
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'postag=' + postag,
        ]
        if i > 1:
            word1 = sent[i-2][0]
            postag1 = sent[i-2][1]
            features.extend([
                '-2:word.lower=' + word1.lower(),
                '-2:postag=' + postag1,
            ])

	if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:postag=' + postag1,
            ])
        else:
            features.append('BOS')
        
        if i < len(sent)-2:
            word1 = sent[i+2][0]
            postag1 = sent[i+2][1]
            features.extend([
                '+2:word.lower=' + word1.lower(),
                '+2:postag=' + postag1,
            ])

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:postag=' + postag1,
            ])
        else:
            features.append('EOS')
        return features
    
    @staticmethod
    def sent2features(sent):
        return [CRF_NER.word2features(sent, i) for i in range(len(sent))]
    
    @staticmethod
    def sent2labels(sent):
        return [label for _, _, label in sent]
    
    @staticmethod
    def sent2tokens(sent):
        return [token for token, _, _ in sent]

    
    def build_datasets(self, train_sents, test_sents):
        print CRF_NER.sent2features(train_sents[0])[0]

        self.X_train = [CRF_NER.sent2features(s) for s in train_sents]
        self.y_train = [CRF_NER.sent2labels(s) for s in train_sents]
        
        self.X_test = [CRF_NER.sent2features(s) for s in test_sents]
        self.y_test = [CRF_NER.sent2labels(s) for s in test_sents]
    
    def train(self, model_path='model1.crfsuite'):
        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(self.X_train, self.y_train):
            trainer.append(xseq, yseq)
            
        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 100,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        
        trainer.params()
        trainer.train(model_path)
        print trainer.logparser.last_iteration
    
    def tag_eval(self, model_path, test_sents, outpath='output.txt'):
        '''
            sent example: [(u'wellness', 'JJ', u'B-SECTION'), (u'section', 'NN', u'I-SECTION'), (u'top', 'JJ', u'B-NEWSTYPE'), (u'news', 'NN', u'I-NEWSTYPE'), (u'stories', 'NNS', u'I-NEWSTYPE'), (u'from', 'IN', u'O'), (u'carroll', 'NN', u'B-PROVIDER'), (u'daily', 'JJ', u'I-PROVIDER'), (u'times', 'NNS', u'I-PROVIDER'), (u'herald', 'VBP', u'I-PROVIDER')]
        '''
        tagger = pycrfsuite.Tagger()
        tagger.open(model_path)
        mylines = []
        for sent in test_sents:
            pred_tags = tagger.tag(CRF_NER.sent2features(sent))
            for i, w in enumerate(sent):
                mylines.append(' '.join([w[0], w[2], pred_tags[i]]))
            mylines.append('')
        outpath = os.path.join(eval_temp, outpath+'.outputs')
        scorespath = os.path.join(eval_temp, outpath+'.scores')
        
        with codecs.open(outpath, 'w', 'utf8') as f:
            f.write("\n".join(mylines))
        os.system("%s < %s > %s" % (eval_script, outpath, scorespath))

        # CoNLL evaluation results
        eval_lines = [l.rstrip() for l in codecs.open(scorespath, 'r', 'utf8')]
        for line in eval_lines:
            print line
    
        # F1 on all entities
        print "F1 on all entities",  float(eval_lines[1].strip().split()[-1])
        
#     def evaluate(self, test_sents):
#         tagger = pycrfsuite.Tagger()
#         tagger.open(self.model_name)
#         
#         print("Predicted:", ' '.join(tagger.tag(CRF_NER.sent2features(test_sents[0]))))
#         print("Correct:  ", ' '.join(CRF_NER.sent2labels(test_sents[0])))
#         
#         y_pred = [tagger.tag(xseq) for xseq in self.X_test]
#         
#         print(CRF_NER.bio_classification_report(self.y_test, y_pred))
