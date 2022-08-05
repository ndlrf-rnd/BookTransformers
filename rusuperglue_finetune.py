import bert
from bert import tokenization
from bert import optimization
from bert import modeling
import re
import json
import pickle
import argparse
import itertools
import collections
import numpy as np
import tensorflow as tf
from sklearn import metrics

from modelling import Model, Model_BigBird
from training import train, batch_size
from inference import *
from dataset_utils import *

BERT_INIT_CHKPNT = 'model.ckpt-1445000'
ALBERT_INIT_CHKPNT = 'model.ckpt-1100000'
BIGBIRD_INIT_CHKPNT = 'model.ckpt-450000'

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match('^(.*):\\d+$', name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        name_r = name.replace('bert/embeddings/LayerNorm', 'bert/encoder/LayerNorm')
        if name_r not in name_to_variable:
            continue
        if 'embeddings/position_embeddings' in name_r:
            continue
        assignment_map[name] = name_to_variable[name_r]
        initialized_variable_names[name_r] = 1
        initialized_variable_names[name_r + ':0'] = 1

    return (assignment_map, initialized_variable_names)


def main(args):
    if not args.pickledData:
        ru_super_glue_datasets = load_all_rusglue_datasets()
        train_terra, valid_terra, test_terra, train_Y, valid_Y, test_Y = extract_hf_data(ru_super_glue_datasets['terra'])
        input_ids, input_masks, segment_ids = preprocess_dataset_texts(train_terra, 
                                                                       valid_terra, 
                                                                       test_terra, None, args.pickledData)
    else:
        input_ids, input_masks, segment_ids = preprocess_dataset_texts(None, 
                                                                       None, 
                                                                       None, args.task, args.pickledData)
        task = args.task
        with open(f'./dsets/{args.task}_train_label.pkl', 'rb') as f:
            train_Y = pickle.load(f)
        with open(f'./dsets/{args.task}_val_label.pkl', 'rb') as f:
            valid_Y = pickle.load(f)
            
    # testing purposes only
    # input_ids['train'], input_masks['train'], segment_ids['train'] = input_ids['train'][:16], input_masks['train'][:16], segment_ids['train'][:16]
    # input_ids['valid'], input_masks['valid'], segment_ids['valid'] = input_ids['valid'][:16], input_masks['valid'][:16], segment_ids['valid'][:16]
    # input_ids['test'], input_masks['test'], segment_ids['test'] = input_ids['test'][:16], input_masks['test'][:16], segment_ids['test'][:16]
    
    
    if args.task == 'rcb':
        dimension_output = 3
    else:
        dimension_output = 2

    learning_rate = args.lr
    epoch = args.max_epochs
    num_train_steps = int((len(input_ids['train']) + len(input_ids['valid'])) / batch_size * epoch)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    if args.model_name.lower() == 'bigbird':
        model = Model_BigBird(
                dimension_output,
                num_train_steps,
                learning_rate
                )
    else:
        model = Model(
            dimension_output,
            args.model_name,
            learning_rate,
            num_train_steps
        )

    sess.run(tf.global_variables_initializer())
    if args.model_name == 'bigbird':
        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars,
                                                                                        BIGBIRD_INIT_CHKPNT)
        saver = tf.train.Saver(var_list = assignment_map)
    else:
        var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'bert')
        saver = tf.train.Saver(var_list = var_lists)



    if args.model_name == 'albert':
        saver.restore(sess, ALBERT_INIT_CHKPNT)
    elif args.model_name == 'bert':
        saver.restore(sess, BERT_INIT_CHKPNT)
    elif args.model_name == 'bigbird':
        saver.restore(sess, BIGBIRD_INIT_CHKPNT)

    model = train(input_ids, input_masks, segment_ids, train_Y, valid_Y, sess, model, args.task)

    predict_valid_Y = inference(input_ids['valid'], input_masks['valid'], segment_ids['valid'], model, sess, batch_size)
    print(
        metrics.classification_report(
            valid_Y, predict_valid_Y, target_names = ['negative', 'positive'], digits=5
        )
    )
    predict_test_Y = inference(input_ids['test'], input_masks['test'], segment_ids['test'], model, sess, batch_size)
    with open(f'./{args.task}_test_label_last.pkl', 'wb') as f:
        pickle.dump(f, predict_test_Y)
    # label_maps = construct_rusuperglue_label_maps()
    # pack_n_dump_predictions_jsonl(test_terra, predict_test_Y, label_maps['terra'], 'test.jsonl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main variables for finetuning')
    parser.add_argument('--max_epochs', type=int,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--pickledData', type=bool)
    parser.add_argument('--task', type=str)
    parser.add_argument('--model_name', type=str, default='bert')

    args = parser.parse_args()
    main(args)
