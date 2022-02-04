import json
import re
from tqdm import tqdm
from transformers import BertTokenizer

#with open('***') as fopen:
#    v = fopen.read().split('\n')[:-1]
#v = [i.split('\t') for i in v]
#v = {i[0]: i[1] for i in v}

#class Tokenizer:
#    def __init__(self, v):
#        self.vocab = v
#        pass
#
#    def tokenize(self, string):
#        return encode_pieces(sp_model, string, return_unicode=False, sample=False)
#
#    def convert_tokens_to_ids(self, tokens):
#        return [sp_model.PieceToId(piece) for piece in tokens]
#
#    def convert_ids_to_tokens(self, ids):
#        return [sp_model.IdToPiece(i) for i in ids]

MAX_SEQ_LENGTH = 100

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
