import bert
from bert import run_classifier
from bert import tokenization
from bert import optimization
from bert import modeling
import tensorflow as tf
import numpy as np
import json
import itertools
import re
from tokenization import pretokenize
from modelling import Model
from training import train
from training import batch_size
from datasets import load_dataset

BERT_INIT_CHKPNT = 'model.ckpt-1445000'

def join_terra(examples):
    premise = examples["premise"]
    hypo = examples["hypothesis"]
    return {'text': f'{premise} [SEP] {hypo}'}

ru_super_glue_terra = load_dataset("russian_super_glue", 'terra')
train_terra = ru_super_glue_terra['train']
valid_terra = ru_super_glue_terra['validation']

train_Y = train_terra['label']
valid_Y = valid_terra['label']

train_terra_texts = train_terra.map(join_terra)['text']
valid_terra_texts = valid_terra.map(join_terra)['text']

train_input_ids, train_input_masks, train_segment_ids = pretokenize(train_terra_texts)
valid_input_ids, valid_input_masks, valid_segment_ids = pretokenize(valid_terra_texts)

dimension_output = 2
learning_rate = 2e-5
epoch = 10
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

