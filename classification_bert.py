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

from modelling import Model
from training import train, batch_size
from datasets import load_dataset
from testing import inference
from dataset_utils import *

BERT_INIT_CHKPNT = 'model.ckpt-1445000'

ru_super_glue_terra = load_dataset("russian_super_glue", 'terra')
ru_super_glue_rcb = load_dataset("russian_super_glue", 'rcb')
ru_super_glue_parus = load_dataset("russian_super_glue", 'parus')
ru_super_glue_muserc = load_dataset("russian_super_glue", 'muserc')
ru_super_glue_russe = load_dataset("russian_super_glue", 'russe')
ru_super_glue_rwsd = load_dataset("russian_super_glue", 'rwsd')
ru_super_glue_danetqa = load_dataset("russian_super_glue", 'danetqa')

train_terra, valid_terra, test_terra, train_Y, valid_Y, test_Y = extract_hf_data(ru_super_glue_terra)
input_ids, input_masks, segment_ids = preprocess_dataset_texts(train_terra, valid_terra, test_terra)

dimension_output = 2
learning_rate = 2e-5
epoch = 2
num_train_steps = int((len(input_ids['train']) + len(input_ids['valid'])) / batch_size * epoch)

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

model = train(input_ids['train'], input_ids['valid'], input_masks['train'], input_masks['valid'], segment_ids['train'], segment_ids['valid'], train_Y, valid_Y, sess, model)

predict_valid_Y = inference(input_ids['valid'], input_masks['valid'], segment_ids['valid'], model, sess, batch_size)
print(
    metrics.classification_report(
        valid_Y, predict_valid_Y, target_names = ['negative', 'positive'], digits=5
    )
)
