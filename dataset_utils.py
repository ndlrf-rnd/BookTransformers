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

