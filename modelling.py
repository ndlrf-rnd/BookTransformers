from bert import optimization
from bert import modeling as modeling_bert
from albert import modeling as modeling_albert
from bigbird import modeling as modeling_bigbird
import tensorflow as tf

BERT_CONFIG = 'configs/baseBert_config.json'
SBER_BERT_CONFIG = 'sber_bert/config.json'
ALBERT_CONFIG = 'configs/baseAlbert_config.json'
bert_config = modeling_bert.BertConfig.from_json_file(BERT_CONFIG)
sber_bert_config = modeling_bert.BertConfig.from_json_file(SBER_BERT_CONFIG)
albert_config = modeling_albert.AlbertConfig.from_json_file(ALBERT_CONFIG)
bigbird_config = {
    'attention_probs_dropout_prob': 0.2,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.2,
    'hidden_size': 768,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 4096,
    'max_encoder_length': 512,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'type_vocab_size': 2,
    'scope': 'bert',
    'use_bias': True,
    'rescale_embedding': False,
    'vocab_model_file': None,
    'attention_type': 'block_sparse',
    'norm_type': 'postnorm',
    'block_size': 16,
    'num_rand_blocks': 3,
    'vocab_size': 32000,
}
#bigbird_config = {
#    'attention_probs_dropout_prob': 0.1,
#    'hidden_act': 'gelu',
#    'hidden_dropout_prob': 0.1,
#    'hidden_size': 512,
#    'initializer_range': 0.02,
#    'intermediate_size': 3072,
#    'max_position_embeddings': 4096,
#    'max_encoder_length': 512,
#    'num_attention_heads': 8,
#    'num_hidden_layers': 6,
#    'type_vocab_size': 2,
#    'scope': 'bert',
#    'use_bias': True,
#    'rescale_embedding': False,
#    'vocab_model_file': None,
#    'attention_type': 'block_sparse',
#    'norm_type': 'postnorm',
#    'block_size': 16,
#    'num_rand_blocks': 3,
#    'vocab_size': 32000,
#}
warmup_proportion = 0.1

def create_initializer(initializer_range=0.02):
    return tf.truncated_normal_initializer(stddev=initializer_range)

class Model_BigBird:
    def __init__(
        self,
        dimension_output,
        num_train_steps,
        learning_rate = 2e-5,
        training = True,
    ):
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None])
        # no need for this variables
        self.segment_ids = tf.placeholder(tf.int32, [None, None])
        self.input_masks = tf.placeholder(tf.int32, [None, None])
        # no need for this variables
        
        model = modeling_bigbird.BertModel(bigbird_config)
        sequence_output, pooled_output = model(self.X)
        
        output_layer = sequence_output
        output_layer = tf.layers.dense(
            output_layer,
            bigbird_config['hidden_size'],
            activation=tf.tanh,
            kernel_initializer=create_initializer())
        self.logits_seq = tf.layers.dense(output_layer, dimension_output,
                                         kernel_initializer=create_initializer())
        self.logits_seq = tf.identity(self.logits_seq, name = 'logits_seq')
        self.logits = self.logits_seq[:, 0]
        self.logits = tf.identity(self.logits, name = 'logits')
        
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

class Model:
    def __init__(
        self,
        dimension_output,
        model_name,
        learning_rate = 2e-5,
        num_train_steps = None,
    ):
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        self.X = tf.placeholder(tf.int32, [None, None])
        self.segment_ids = tf.placeholder(tf.int32, [None, None])
        self.input_masks = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None])
        
        model_name = model_name.lower()
        if model_name == 'bert':
            model = modeling_bert.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=self.X,
                input_mask=self.input_masks,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=False)
        elif model_name == 'albert':
            model = modeling_albert.AlbertModel(
                config = albert_config,
                is_training = False,
                input_ids = self.X,
                input_mask = self.input_masks,
                token_type_ids = self.segment_ids,
                use_one_hot_embeddings = False,
            )
        elif model_name == 'sber_bert':
            model = modeling_bert.BertModel(
                config = sber_bert_config,
                is_training = False,
                input_ids = self.X,
                input_mask = self.input_masks,
                token_type_ids = self.segment_ids,
                use_one_hot_embeddings = False,
            )
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
        

class Model_embed:
    def __init__(
        self,
    ):
        self.X = tf.placeholder(tf.int32, [None, None])
 #       self.segment_ids = tf.placeholder(tf.int32, [None, None])
 #       self.input_masks = tf.placeholder(tf.int32, [None, None])
 #       self.Y = tf.placeholder(tf.int32, [None])
        
        model = modeling_bert.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=self.X,
 #           input_mask=self.input_masks,
 #           token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        
        self.output_layer = model.get_pooled_output()
