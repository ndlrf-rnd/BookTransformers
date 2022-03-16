import tensorflow as tf
from modelling import Model_embed
from tokenization import pretokenize, pretokenize_split_text
from embeddings import embed_texts
import argparse

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
    file_paths_list = file_paths.split(',')
    tokenized_files = {}
    for file_path in file_paths_list:
        with open(file_path) as f:
            text = f.read()
        tokenized_files[file_path] = pretokenize_split_text(text)
        for i in range(len(tokenized_files[file_path][0])):
            #print(tokenized_files[file_path][1][i], tokenized_files[file_path][0][i])
            embeddings = embed_texts(tokenized_files[file_path][0][i], 
                                     tokenized_files[file_path][1][i], 
                                     tokenized_files[file_path][2][i], 
                                     sess, model, file_path+str(i+1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main variables for embeddings extraction')
    parser.add_argument('--file_paths', type=str)

    args = parser.parse_args()
    main(args)
