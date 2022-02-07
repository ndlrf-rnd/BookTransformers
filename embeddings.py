from tqdm import tqdm
import numpy as np

batch_size = 60

def embed_texts(train_input_ids, train_input_masks, train_segment_ids, sess, model):
    pbar = tqdm(
<<<<<<< HEAD
        range(0, len(train_input_ids), batch_size), desc = 'embeddings extracting'
=======
        range(0, len(train_input_ids), batch_size), desc = 'embeddings processing'
>>>>>>> 59e3b83... BERT-like pooled_layer text embeddings extraction functions added
    )
    embeddings = []
    for i in pbar:
        index = min(i + batch_size, len(train_input_ids))
        batch_x = train_input_ids[i: index]
        batch_masks = train_input_masks[i: index]
        batch_segment = train_segment_ids[i: index]
<<<<<<< HEAD
        batch_embeddings = sess.run(
            [model.output_layer],
            feed_dict = {
=======
        #batch_y = train_Y[i: index]
        batch_embeddings = sess.run(
            [model.output_layer],
            feed_dict = {
               # model.Y: batch_y,
>>>>>>> 59e3b83... BERT-like pooled_layer text embeddings extraction functions added
                model.X: batch_x,
                model.segment_ids: batch_segment,
                model.input_masks: batch_masks
            },
        )
        embeddings += batch_embeddings
    return embeddings
