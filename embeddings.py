from tqdm import tqdm
import numpy as np

batch_size = 60

def embed_texts(train_input_ids, train_input_masks, train_segment_ids, sess, model, prefix, dump=True):
    pbar = tqdm(
        range(0, len(train_input_ids), batch_size), desc = 'embeddings extracting'
    )
    embeddings = []
    for i in pbar:
        index = min(i + batch_size, len(train_input_ids))
        #print('idxs', i, ':', index)
        batch_x = [train_input_ids[i: index]]
        batch_masks = [train_input_masks[i: index]]
        batch_segment = [train_segment_ids[i: index]]
        #print('batch_x len', len(batch_x[0]), ':', len(batch_x[0][0]))
        batch_embeddings = sess.run(
            [model.output_layer],
            feed_dict = {
                model.X: batch_x,
                model.segment_ids: batch_segment,
                model.input_masks: batch_masks
            },
        )
        embeddings += batch_embeddings
    if dump:
        with open(f'{prefix}_embeddings.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
    return embeddings
