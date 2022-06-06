import re
import os
import json
import kfserving
import tensorflow as tf
from typing import List, Dict
from modelling import Model_embed
from tensorflow.keras.preprocessing.sequence import pad_sequences

class KFServingExplainModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model = None
        self.gpu = True

    def load(self):
        BERT_INIT_CHKPNT = 'model.ckpt-1445000'
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        model = Model_embed()

        sess.run(tf.global_variables_initializer())
        var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'bert')
        saver = tf.train.Saver(var_list = var_lists)
        saver.restore(sess, BERT_INIT_CHKPNT)

    def predict(self, request: Dict) -> Dict:
        text = request['instances'][0]['text']
        tokenized_files = pretokenize_split_text(text)

        x = [train_input_ids[i: index]]
        x = pad_sequences(batch_x, padding='post')
        embeddings = sess.run(
            [model.output_layer],
            feed_dict = {
                model.X: x,
            },
        )
        return {"embedding": result}


if __name__ == "__main__":
    x = re.compile('(kfserving-\d+)').search(os.environ.get('HOSTNAME'))
    name = "kfserving-default"
    if x:
        name = x[0]

    model = KFServingExplainModel(name)
    model.load()
    kfserving.KFServer(workers=1).start([model])
