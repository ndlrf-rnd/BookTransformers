import time
from tqdm import tqdm
import numpy as np
from inference import *

batch_size = 8

def train(input_ids, input_masks, segment_ids, train_Y, test_Y, sess, model, task):
    train_input_ids, test_input_ids, train_input_masks, test_input_masks, train_segment_ids, test_segment_ids = input_ids['train'], \
    input_ids['valid'], input_masks['train'], input_masks['valid'], segment_ids['train'], segment_ids['valid']
    EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
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
            test_acc += acc
        train_loss /= len(train_input_ids) / batch_size
        train_acc /= len(train_input_ids) / batch_size
        test_loss /= len(test_input_ids) / batch_size
        test_acc /= len(test_input_ids) / batch_size

        if test_acc > CURRENT_ACC:
            print(
                'epoch: %d, pass acc: %f, current acc: %f'
                % (EPOCH, CURRENT_ACC, test_acc)
            )
            CURRENT_ACC = test_acc
            CURRENT_CHECKPOINT = 0
            predict_test_Y = inference(input_ids['test'], input_masks['test'], segment_ids['test'], model, sess, batch_size)
            with open(f'./{task}_test_label_epoch{EPOCH}_acc{test_acc}.pkl', 'wb') as f:
                pickle.dump(predict_test_Y, f)
        else:
            CURRENT_CHECKPOINT += 1

        print('time taken:', time.time() - lasttime)
        print(
            'epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n'
            % (EPOCH, train_loss, train_acc, test_loss, test_acc)
        )
        EPOCH += 1
    return model
