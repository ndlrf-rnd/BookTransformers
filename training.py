import time
from tqdm import tqdm
import numpy as np

EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 3, 0, 0, 0
batch_size = 60

def train(train_input_ids, test_input_ids, train_input_masks, test_input_masks, train_segment_ids, test_segment_ids, train_Y, test_Y, sess, model):
    while True:
        lasttime = time.time()
        if CURRENT_CHECKPOINT == EARLY_STOPPING:
            print('break epoch:%d\n' % (EPOCH))
            break

        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
        pbar = tqdm(
            range(0, len(train_input_ids), batch_size), desc = 'train minibatch loop'
        )
        for i in pbar:
            index = min(i + batch_size, len(train_input_ids))
            batch_x = train_input_ids[i: index]
            batch_masks = train_input_masks[i: index]
            batch_segment = train_segment_ids[i: index]
            batch_y = train_Y[i: index]
            acc, cost, _ = sess.run(
                [model.accuracy, model.cost, model.optimizer],
                feed_dict = {
                    model.Y: batch_y,
                    model.X: batch_x,
                    model.segment_ids: batch_segment,
                    model.input_masks: batch_masks
                },
            )
            assert not np.isnan(cost)
            train_loss += cost
            train_acc += acc
            pbar.set_postfix(cost = cost, accuracy = acc)
        pbar = tqdm(range(0, len(test_input_ids), batch_size), desc = 'test minibatch loop')
        for i in pbar:
            index = min(i + batch_size, len(test_input_ids))
            batch_x = test_input_ids[i: index]
            batch_masks = test_input_masks[i: index]
            batch_segment = test_segment_ids[i: index]
            batch_y = test_Y[i: index]
            acc, cost = sess.run(
                [model.accuracy, model.cost],
                feed_dict = {
                    model.Y: batch_y,
                    model.X: batch_x,
                    model.segment_ids: batch_segment,
                    model.input_masks: batch_masks
                },
            )
            test_loss += cost
    return model
