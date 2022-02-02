from bert import optimization
from bert import modeling
import tensorflow as tf

bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)

class Model:
    def __init__(
        self,
        dimension_output,
        learning_rate = 2e-5,
    ):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.segment_ids = tf.placeholder(tf.int32, [None, None])
        self.input_masks = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None])
        
        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=self.X,
            input_mask=self.input_masks,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        
        output_layer = model.get_pooled_output()
        self.logits = tf.layers.dense(output_layer, dimension_output)
        
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = self.logits, labels = self.Y
            )
        )
        
        self.optimizer = optimization.create_optimizer(self.cost, learning_rate, 
                                                       num_train_steps, num_warmup_steps, False)
        correct_pred = tf.equal(
            tf.argmax(self.logits, 1, output_type = tf.int32), self.Y
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
