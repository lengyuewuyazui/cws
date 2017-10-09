# encoding: utf-8
'''
LSTM 中文分词
Refer: https://mp.weixin.qq.com/s?__biz=MzA4OTk5OTQzMg==&mid=2449231335&idx=1&sn=d3ba98841e85b7cea0049cc43b3c16ca

设窗口长度为 7，那么从训练样本中切分 xxxoxxx，根据前三个 x，后三个 x 和 o 本身，训练 o 的 tag (B/E/M/S)
为了能够处理开头和结尾的字符，要在训练文本和测试文本的开头和结尾分别 padding 三个 '\01' 不可见字符，作为 START/END
要求训练文本和测试文本没有空格；这里默认已经处理过了，不需要本脚本再去做处理
重要：算法中默认测试集中没有训练文本中不存在的汉字，也就是说，要求训练文本中的汉字空间是完全覆盖的!!!!
这里没有把整个训练和测试流程化，只是提供了函数接口 pretraining, training, run_test
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np


def pretraining(train_text, train_tags, maxlen, step):
    """
        train_text: 待训练文本
        train_tags: 待训练tags，和train_text等长，且一一对应
        maxlen    : 切分窗口长度，要求是奇数；比如7，表示考察一个字的前3个字和后3个字
        step      : 切分窗口间隔
    """
    # 预处理：把样本中的所有字符映射为数字
    print('corpus length:', len(train_text))
    # 字符集中加入 '\01'，不可见字符，用于在 train_text 前后 padding
    half_window = maxlen / 2
    train_text = u'\01' * half_window + train_text + u'\01' * half_window
    chars = sorted(list(set(train_text)))
    print('chars length:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))

    # 要学习的分词标记，分别表示词开始、词结尾、词中间、单字成词
    tags = ['B', 'E', 'M', 'S']
    # 在 train_tags 前后 padding 'S' 单字 tag
    train_tags = 'S' * half_window + train_tags + 'S' * half_window
    print('total tags:', len(tags))
    tag_indices = dict((c, i) for i, c in enumerate(tags))

    # 开始切分，用某个字的前 half_window + 后 half_window 个字来预测该字的 tag
    windows = []
    next_tags = []
    end_pos = len(train_text) - half_window
    for i in range(half_window, end_pos, step):
        windows.append(train_text[i - half_window: i + half_window + 1])
        next_tags.append(train_tags[i])
    print('nb sequences:', len(windows))

    # 向量化，前面得到的是字符和标记数组，要转化为数字
    # X 和 Y 各自都有 len(windows) 组；每组中 X 为 maxlen 个字，每个字 len(chars) 维，one-hot； Y为 len(tags) 维
    print('Vectorization...')
    X = np.zeros((len(windows), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(windows), len(tags)), dtype=np.bool)
    for i, sentence in enumerate(windows):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, tag_indices[next_tags[i]]] = 1

    return X, y, chars, tags, char_indices


def training(X, y, maxlen, len_of_chars, len_of_tags, hidden_nodes=128, batch_size=128, nb_epoch=1):
    # build the model: 2 stacked LSTM
    model = get_lstm_model(hidden_nodes, (maxlen, len_of_chars), len_of_tags)
    # train
    for iteration in range(1, 61):
        print_iteration_sign(iteration)
        model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch)
        yield model


def run_test(model, test_text, diversity, maxlen, len_of_chars, char_indices, tags):
    """
        model: 用于预测的模型
        test_text: 待预测字符串
        diversity: 调整抽样概率
        maxlen: 窗口大小
        len_of_char: 用于确定 x 的维度 (one-hot)
        char_indices: 用于确定 x one-hot 字符维度上的值
        tags: 用于把预测的结果还原为 tag 字符
    """
    print()
    print('>>>>> diversity: ', diversity)
    half_window = maxlen / 2
    # padding with '\01'
    test_text = u'\01' * half_window + test_text + u'\01' * half_window
    # 初始化预测结果
    next_tags = ''
    end_pos = len(test_text) - half_window
    # 每个字都要预测，故此显然不要设置 step
    for i in range(half_window, end_pos):
        window = test_text[i - half_window: i + half_window + 1]
        x = np.zeros((1, maxlen, len_of_chars))
        for t, char in enumerate(window):
            x[0, t, char_indices[char]] = 1.
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_tags += tags[next_index]
    return next_tags


def get_lstm_model(hidden_nodes, input_shape, output_nodes):
    print('Build model...')
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=input_shape))
    # 输出 len(tags) 维度
    model.add(Dense(output_nodes))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def sample(preds, temperature=1.0):
    """
        给定一个多维预测结果，在其中抽样一个，并取出该抽样的 index
        temperature 用于调整输入的 preds
        temperature 为 1，则没有调整
        temparature < 1 会加大差距，比如 array([ 0.1 ,  0.15,  0.5 ,  0.25]) ---0.5---> array([ 0.02898551,  0.06521739,  0.72463768,  0.18115942])
        反之 temparature > 1 会均匀化，比如同样的数组，经过 1.3 变为  array([ 0.12757797,  0.17427316,  0.43999216,  0.2581567 ])
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # 抽样，从所有预测维度中，只抽取一个作为结果
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def print_iteration_sign(i):
    print()
    print('-' * 50)
    print('Iteration: ', i)
