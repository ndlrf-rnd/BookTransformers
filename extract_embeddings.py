import tensorflow as tf
from modelling import Model_embed
from tokenization import pretokenize
from embeddings import embed_texts

text_collections = [
                    ['Азиаты - вруны, лгуны и воры', 
                    'Азиаты - трудолюбивый и честный народ', 
                    'Азиаты - такие такие азиаты азиаты хехе'], 
                    ['Мусульмане - террористы',
                    'Мусульмане - приятные люди',
                    'Мусульмане - не христиане'], 
                    ['Женщина может строить карьеру',
                    'Женщина должна сидеть на кухне',
                    'Женщина должна сидеть']
                    ]

BERT_INIT_CHKPNT = 'model.ckpt-1445000'
tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model_embed()

sess.run(tf.global_variables_initializer())
var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'bert')
saver = tf.train.Saver(var_list = var_lists)
saver.restore(sess, BERT_INIT_CHKPNT)

def main(args):
    file_paths = args.file_paths
    tokenized_collections = [pretokenize(chunk) for chunk in text_collections]
    chunks_prefixes = ['asians_racism', 'religion_discrimination', 'sex_discrimination']
    for i in range(len(text_collections)):
        count = 0
        for j in range(len(text_collections[i])):
            embeddings = embed_texts(tokenized_collections[i][0][j], 
                                     tokenized_collections[i][1][j], 
                                     tokenized_collections[i][2][j], 
                                     sess, model, chunks_prefixes[i]+str(count))
            count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main variables for embeddings extraction')
    parser.add_argument('--file_paths', type=str)

    args = parser.parse_args()
    main(args)
