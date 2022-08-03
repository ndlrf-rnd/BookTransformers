from tokenization import pretokenize
from datasets import load_dataset
import pickle

def join_terra_rcb(examples):
    premise = examples["premise"]
    hypo = examples["hypothesis"]
    return {'text': f"{premise} [SEP] {hypo}"}

def join_parus(examples):
    premise = examples["premise"]
    q1 = examples["choice1"]
    q2 = examples["choice2"]
    question = examples["question"]
    return {'text': f"{premise} [SEP] {question} [SEP] {q1} [SEP] {q2}"}

def join_muserc(examples):
    premise = examples["paragraph"]
    question = examples["question"]
    answer = examples["answer"]
    return {'text': f"{premise} [SEP] {question} [SEP] {answer}"}

def join_russe(examples):
    premise = examples["word"]
    s1 = examples["sentence1"]
    s2 = examples["sentence2"]
    return {'text': f"{premise} [SEP] {s1} [SEP] {s2}"}

def join_rwsd(examples):
    premise = examples["text"]
    s1 = examples["span1_text"]
    s2 = examples["span2_text"]
    return {'text': f"{premise} [SEP] {s1} [SEP] {s2}"}

def join_danetqa(examples):
    premise = examples["passage"]
    question = examples["question"]
    return {'text': f"{premise} [SEP] {question}"}


def extract_hf_data(raw_dataset):
    train_ds = raw_dataset['train']
    valid_ds = raw_dataset['validation']
    test_ds = raw_dataset['test']

    train_Y = train_ds['label']
    valid_Y = valid_ds['label']
    test_Y = test_ds['label']
    return train_ds, valid_ds, test_ds, train_Y, valid_Y, test_Y

def preprocess_dataset_texts(train_dataset, valid_dataset, test_dataset, filenames):
    if filenames == '':
        train_dataset_texts = train_dataset.map(join_terra_rcb)['text']
        valid_dataset_texts = valid_dataset.map(join_terra_rcb)['text']
        test_dataset_texts = test_dataset.map(join_terra_rcb)['text']
    else:
        with open(f'./dsets/{filenames}_train_texts.pkl', 'rb') as f:
            train_dataset_texts = pickle.load(f)
        with open(f'./dsets/{filenames}_val_texts.pkl', 'rb') as f:
            valid_dataset_texts = pickle.load(f)
        with open(f'./dsets/{filenames}_test_texts.pkl', 'rb') as f:
            test_dataset_texts = pickle.load(f)

    train_input_ids, train_input_masks, train_segment_ids = pretokenize(train_dataset_texts)
    valid_input_ids, valid_input_masks, valid_segment_ids = pretokenize(valid_dataset_texts)
    test_input_ids, test_input_masks, test_segment_ids = pretokenize(test_dataset_texts)
    return {'train': train_input_ids, 'valid': valid_input_ids, 'test': test_input_ids},\
           {'train': train_input_masks, 'valid': valid_input_masks, 'test': test_input_masks},\
           {'train': train_segment_ids, 'valid': valid_segment_ids, 'test': test_segment_ids},\

def load_all_rusglue_datasets():
    ru_super_glue_terra = load_dataset("russian_super_glue", 'terra')
    ru_super_glue_rcb = load_dataset("russian_super_glue", 'rcb')
    ru_super_glue_parus = load_dataset("russian_super_glue", 'parus')
    ru_super_glue_muserc = load_dataset("russian_super_glue", 'muserc')
    ru_super_glue_russe = load_dataset("russian_super_glue", 'russe')
    ru_super_glue_rwsd = load_dataset("russian_super_glue", 'rwsd')
    ru_super_glue_danetqa = load_dataset("russian_super_glue", 'danetqa')
    return {'terra': ru_super_glue_terra, 'rcb': ru_super_glue_rcb, 'parus': ru_super_glue_parus,
            'muserc': ru_super_glue_muserc, 'russe': ru_super_glue_russe, 'rwsd': ru_super_glue_rwsd,
            'danetqa': ru_super_glue_danetqa}

def construct_rusuperglue_label_maps():
    terra_label_map = {1: 'entailment', 0: 'not_entailment'}
    return {'terra': terra_label_map}
