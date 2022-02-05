import bert
from bert import tokenization
from bert import optimization
from bert import modeling
import tensorflow as tf
import numpy as np
import json
import itertools
import re
from sklearn import metrics

from tokenization import pretokenize
from modelling import Model
from training import train, batch_size
from datasets import load_dataset
from dataset_utils import join_terra_rcb
from testing import inference

BERT_INIT_CHKPNT = 'model.ckpt-1445000'

ru_super_glue_terra = load_dataset("russian_super_glue", 'terra')
train_terra = ru_super_glue_terra['train']
valid_terra = ru_super_glue_terra['validation']
test_terra = ru_super_glue_terra['test']

train_Y = train_terra['label']
valid_Y = valid_terra['label']
test_Y = test_terra['label']

train_terra_texts = train_terra.map(join_terra_rcb)['text']
valid_terra_texts = valid_terra.map(join_terra_rcb)['text']
test_terra_texts = test_terra.map(join_terra_rcb)['text']

train_input_ids, train_input_masks, train_segment_ids = pretokenize(train_terra_texts)
valid_input_ids, valid_input_masks, valid_segment_ids = pretokenize(valid_terra_texts)
test_input_ids, test_input_masks, test_segment_ids = pretokenize(test_terra_texts)

dimension_output = 2
learning_rate = 2e-5
epoch = 2
num_train_steps = int((len(train_terra_texts) + len(valid_terra_texts)) / batch_size * epoch)

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(
    dimension_output,
    learning_rate,
    num_train_steps
)

sess.run(tf.global_variables_initializer())
var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'bert')
saver = tf.train.Saver(var_list = var_lists)
saver.restore(sess, BERT_INIT_CHKPNT)

model = train(train_input_ids, valid_input_ids, train_input_masks, valid_input_masks, train_segment_ids, valid_segment_ids, train_Y, valid_Y, sess, model)

predict_valid_Y = inference(valid_input_ids, valid_input_masks, valid_segment_ids, model, sess, batch_size)
print(
    metrics.classification_report(
        valid_Y, predict_valid_Y, target_names = ['negative', 'positive'], digits=5
    )
)
