# BookTransformers
Descriptions and links providing for trained BookTransformers

Список обученных моделей
Все модели обучены на текстах файла "ru-cc.txt.gz" (230 Gb книг)
- elmo : https://github.com/allenai/bilm-tf
(LSTM Hidden Size/Output size) - loss 
(1024/128) - ~30.0
(2048/256) - ~20.0
Пока не заводил модель на inference

* fasttext : https://github.com/facebookresearch/fastText
ru-cc_skipgram_fasttext_dim300_epoch6 и ru-cc_cbow_fasttext
skipgram и cbow претрейн векторов fasttext
Заводил как pretrained_vectors на задаче определения "Экстремизм/неЭкстремизм" - давало прирост в районе погрешности, не бенчмаркал

* gpt3-neo/weights : https://github.com/EleutherAI/gpt-neo (данные и токенайзер заготовлены по гайду) 
"./model.ckpt-*" - GPT-NEO XL (1.3B parameters)
"./300m/model.ckpt-*" - GPT-NEO medium (300M parameters)
"./125m/model.ckpt-..." - GPT-NEO small (125M parameters)

* bert : https://github.com/google-research/bert

* bigbird : https://github.com/google-research/bigbird
BERT и BigBird тестил с вот этой либой для саммаризации https://github.com/dmmiller612/bert-extractive-summarizer
Или сам метод саммаризации model-agnostic, или же модели просто хорошо отрабатывали на саммари
Также дообучал BERT на RussianSuperGLUE (https://russiansuperglue.com/leaderboard/2)
С помощью скрипта на pytorch-lightning ниже
Дообучил на все задачи кроме двух, качество так себе

* albert : https://github.com/google-research/albert
tiny/base/large sizes

* pegasus : https://github.com/google-research/pegasus
base/small sizes
Обычно эти модели применяют как декодеры в связке с BigBird в качестве энкодера, еще не пробовал

* w2v : https://radimrehurek.com/gensim/models/word2vec.html
что-то пошло не так, эмбеддинги при отображении с помощью tsne давали хаотично расположенные точки

Полезные ссылки
* tensorflow bert finetuning: https://github.com/huseinzol05/malaya/blob/master/pretrained-model/bert/how-to-classification.ipynb
* huggingface w/ pytorch-lightning finetuning: https://colab.research.google.com/drive/10aiF1rw2ijUvERJW13Zw4ByVELQ31rEO?usp=sharing
