import json
import re
from tqdm import tqdm
from transformers import BertTokenizer

#MAX_SEQ_LENGTH = 100
# bigbird
MAX_SEQ_LENGTH = 512

def pretokenize(texts):
    #tokenizer = Tokenizer(v)
    tokenizer = BertTokenizer.from_pretrained('./ruBookBertTokenizer', do_lower_case = False)
    input_masks, input_ids, segment_ids = [], [], []
    for text in tqdm(texts):
        input_id = tokenizer(text)['input_ids']
        #label = text['label']
        if len(input_id) > MAX_SEQ_LENGTH:
            input_id = input_id[:MAX_SEQ_LENGTH]
        tokens = input_id
        segment_id = [0] * len(tokens)
        input_mask = [1] * len(input_id)
        padding = [0] * (MAX_SEQ_LENGTH - len(input_id))
        input_id += padding
        input_mask += padding
        segment_id += padding
        
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
    return input_ids, input_masks, segment_ids
