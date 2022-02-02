import bert
from bert import run_classifier
from bert import tokenization
from bert import optimization
from bert import modeling
import tensorflow as tf
import numpy as np
import json
import itertools
from unidecode import unidecode
import re
from tokenization import pretokenize
from modelling import Model
from training import train

BERT_INIT_CHKPNT = '***'
BERT_CONFIG = '***'

with open('subjectivity-negative-bm.txt','r') as fopen:
    texts = fopen.read().split('\n')
labels = [0] * len(texts)

with open('subjectivity-positive-bm.txt','r') as fopen:
    positive_texts = fopen.read().split('\n')
labels += [1] * len(positive_texts)
texts += positive_texts

assert len(labels) == len(texts)


input_ids, input_masks, segment_ids = pretokenize(texts)

dimension_output = 2
learning_rate = 2e-5

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(
    dimension_output,
    learning_rate
)

sess.run(tf.global_variables_initializer())
var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'bert')
saver = tf.train.Saver(var_list = var_lists)
saver.restore(sess, BERT_INIT_CHKPNT)

epoch = 10
batch_size = 60
warmup_proportion = 0.1
num_train_steps = int(len(texts) / batch_size * epoch)
num_warmup_steps = int(num_train_steps * warmup_proportion)

from sklearn.cross_validation import train_test_split

train_input_ids, test_input_ids, train_input_masks, test_input_masks, train_segment_ids, test_segment_ids, train_Y, test_Y = train_test_split(
    input_ids, input_masks, segment_ids, labels, test_size = 0.2
)

model = train(train_input_ids, test_input_ids, train_input_masks, test_input_masks, train_segment_ids, test_segment_ids, train_Y, test_Y, sess, model)

