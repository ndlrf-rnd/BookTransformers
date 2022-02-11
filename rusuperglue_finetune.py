import bert
from bert import tokenization
from bert import optimization
from bert import modeling
import tensorflow as tf
import numpy as np
import json
import itertools
import re
import argparse
from sklearn import metrics

from modelling import Model
from training import train, batch_size
from inference import *
from dataset_utils import *

BERT_INIT_CHKPNT = 'model.ckpt-1445000'
ALBERT_INIT_CHKPNT = 'model.ckpt-1100000'

def main(args):
    ru_super_glue_datasets = load_all_rusglue_datasets()
    train_terra, valid_terra, test_terra, train_Y, valid_Y, test_Y = extract_hf_data(ru_super_glue_datasets['terra'])
    input_ids, input_masks, segment_ids = preprocess_dataset_texts(train_terra, valid_terra, test_terra)
    dimension_output = 2
    learning_rate = args.lr
    epoch = args.epochs
    num_train_steps = int((len(input_ids['train']) + len(input_ids['valid'])) / batch_size * epoch)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    print(args.model_name)
    model = Model(
        dimension_output,
        args.model_name,
        learning_rate,
        num_train_steps
    )

    sess.run(tf.global_variables_initializer())
    var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'bert')
    saver = tf.train.Saver(var_list = var_lists)


    if args.model_name == 'albert':
        saver.restore(sess, ALBERT_INIT_CHKPNT)
    elif args.model_name == 'bert':
        saver.restore(sess, BERT_INIT_CHKPNT)
    elif args.model_name == 'bigbird':
        saver.restore(sess, BIGBIRD_INIT_CHKPNT)

    model = train(input_ids['train'], input_ids['valid'], input_masks['train'], input_masks['valid'], segment_ids['train'], segment_ids['valid'], train_Y, valid_Y, sess, model)

    predict_valid_Y = inference(input_ids['valid'], input_masks['valid'], segment_ids['valid'], model, sess, batch_size)
    print(
        metrics.classification_report(
            valid_Y, predict_valid_Y, target_names = ['negative', 'positive'], digits=5
        )
    )
    predict_test_Y = inference(input_ids['test'], input_masks['test'], segment_ids['test'], model, sess, batch_size)
    terra_label_map = {1: 'entailment', 0: 'not_entailment'}
    pack_n_dump_predictions_jsonl(test_terra, predict_test_Y, terra_label_map, 'test.jsonl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main variables for finetuning')
    parser.add_argument('--epochs', type=int,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--model_name', type=str, default='bert')

    args = parser.parse_args()
    main(args)
