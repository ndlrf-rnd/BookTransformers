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
tokenizer = BertTokenizer.from_pretrained('./ruBookBertTokenizer', do_lower_case = False)
print(tokenizer('Чертовы гуки, засели прямо на деревьях'))

def tokenize_function_terra_rcb(examples):
    premise = examples["premise"]
    hypo = examples["hypothesis"]
    return tokenizer(f"[CLS] {premise} [SEP] {hypo}", padding="max_length", max_length=512, truncation=True)

def pretokenize(texts):
    tokenizer = Tokenizer(v)
    input_masks, input_ids, segment_ids = [], [], []
    for text in tqdm(texts):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > MAX_SEQ_LENGTH - 2:
            tokens_a = tokens_a[:(MAX_SEQ_LENGTH - 2)]
        tokens = ["<cls>"] + tokens_a + ["<sep>"]
        segment_id = [0] * len(tokens)
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_id)
        padding = [0] * (MAX_SEQ_LENGTH - len(input_id))
        input_id += padding
        input_mask += padding
        segment_id += padding
        
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
    return input_ids, input_masks, segment_ids
