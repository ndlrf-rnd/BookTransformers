import numpy as np
from tqdm import tqdm
import pickle
import json

def inference(test_input_ids, test_input_masks, test_segment_ids, model, sess, batch_size):
    predict_Y = []
    pbar = tqdm(
        range(0, len(test_input_ids), batch_size), desc = 'inference minibatch loop'
    )
    for i in pbar:
        index = min(i + batch_size, len(test_input_ids))
        batch_x = test_input_ids[i: index]
        batch_masks = test_input_masks[i: index]
        batch_segment = test_segment_ids[i: index]
        predict_Y += np.argmax(sess.run(model.logits,
                feed_dict = {
                    model.X: batch_x,
                    model.segment_ids: batch_segment,
                    model.input_masks: batch_masks
                },
        ), 1, ).tolist()
    return predict_Y

def pack_n_dump_predictions_jsonl(test_dataset_split, labels, labels_map, dump_filename):
    mapped_label = [labels_map[value] for value in labels]
    for prem, hypo, idx, label in zip(test_dataset_split['premise'],
                                      test_dataset_split['hypothesis'],
                                      test_dataset_split['idx'],
                                      mapped_label):
        with open(dump_filename, 'w') as outfile:
            json.dump({'premise': prem, 'hypothesis': hypo, 'idx': idx, 'label': label}, outfile, ensure_ascii=False)
            outfile.write('\n')
