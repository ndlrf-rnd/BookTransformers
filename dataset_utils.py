from tokenization import pretokenize

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

def preprocess_dataset_texts(train_terra, valid_terra, test_terra):
    train_terra_texts = train_terra.map(join_terra_rcb)['text']
    valid_terra_texts = valid_terra.map(join_terra_rcb)['text']
    test_terra_texts = test_terra.map(join_terra_rcb)['text']

    train_input_ids, train_input_masks, train_segment_ids = pretokenize(train_terra_texts)
    valid_input_ids, valid_input_masks, valid_segment_ids = pretokenize(valid_terra_texts)
    test_input_ids, test_input_masks, test_segment_ids = pretokenize(test_terra_texts)
    return {'train': train_input_ids, 'valid': valid_input_ids, 'test': test_input_ids},\
           {'train': train_input_masks, 'valid': valid_input_masks, 'test': test_input_masks},\
           {'train': train_segment_ids, 'valid': valid_segment_ids, 'test': test_segment_ids},\
