# encoding: utf-8
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys

'''
    老版本，新版本见 github keras examples: lstm_text_generation.py
    Example script to generate text from Nietzsche's writings.
    At least 20 epochs are required before the generated text
    starts sounding coherent.
    It is recommended to run this script on GPU, as recurrent
    networks are quite computationally intensive.
    If you try this script on new data, make sure your corpus
    has at least ~100k characters. ~1M is better.
'''

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))   # 这个是字符 char 的个数，600901 个，而不是 word 的个数

chars = set(text)
print('total chars:', len(chars))    # 同样，是 unique char 的个数，59 个独立字符
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
step = 3
sentences = []
next_chars = []
# 从头开始到倒数第20个字符，每 3 个字符做一次循环
for i in range(0, len(text) - maxlen, step):
    # 从循环中的起始字符开始，取 20 个字符，作为输入
    sentences.append(text[i: i + maxlen])
    # 那么第 21 个作为输出，就是说，这个仍然是通过字符来推测下一个字符，而不是 word
    next_chars.append(text[i + maxlen])
# 这样，每 3 个字符得到一个 20字符 ==> 1个后续字符的输入输出对儿，共计 (600901 - 20) / 3 = 200394 对儿
print('nb sequences:', len(sentences))

print('Vectorization...')
# 输入向量共计 len(sentences) 个；每个向量为 maxlen=20 个字符；每个字符为一个 onehot 向量，字符数为 len(chars) = 59 个
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
# 输出向量共计 len(sentences) 个；每个向量都是输入向量的后续 1 个字符，同样是 onehot 向量，字符数为 len(chars) = 59 个
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
# 下面循环 sentences，对每对儿输入输出设置 onehot 中的 1
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    # 进行训练，会输出 Epoch 1/1 及训练进度条和 loss
    model.fit(X, y, batch_size=128, nb_epoch=1)    # 一个 batch 128 对儿输入输出

    # 开始生成 text，随机找一个位置开始的 20 个字符作为出发点，生成后续字符
    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        # 由出发点开始，对不同的 diversity (灵活度)，生成 400 个后续字符
        for iteration in range(400):
            # 向量化输入 sentence
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            # 预测，结果是一个 softmax，也就是说 59 个备选字符，每个字符的当选概率
            preds = model.predict(x, verbose=0)[0]
            # 根据灵活度取样得到预测字符是哪个，结果是个 index
            next_index = sample(preds, diversity)
            # 根据 index 找到对应字符
            next_char = indices_char[next_index]

            # 生成的句子中加入新预测的字符
            generated += next_char
            # 为了预测再下一个字符，那么把 20个字符的窗口移动一位，就是说去掉第一个，加入新预测的字符
            sentence = sentence[1:] + next_char

            # 输出结果
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
