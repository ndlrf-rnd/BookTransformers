import numpy as np
from tqdm import tqdm

def inference(test_input_ids, test_input_masks, test_segment_ids, model, sess, batch_size):
    predict_Y = []
    pbar = tqdm(
        range(0, len(test_input_ids), batch_size), desc = 'validation minibatch loop'
    )
    for i in pbar:
        index = min(i + batch_size, len(test_input_ids))
        batch_x = test_input_ids[i: index]
        batch_masks = test_input_masks[i: index]
        batch_segment = test_segment_ids[i: index]
        #batch_y = test_Y[i: index]
        predict_Y += np.argmax(sess.run(model.logits,
                feed_dict = {
                    #model.Y: batch_y,
                    model.X: batch_x,
                    model.segment_ids: batch_segment,
                    model.input_masks: batch_masks
                },
        ), 1, ).tolist()
    return predict_Y
