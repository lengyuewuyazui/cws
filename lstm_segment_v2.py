# encoding: utf-8
'''
LSTM 中文分词
Refer: https://mp.weixin.qq.com/s?__biz=MzA4OTk5OTQzMg==&mid=2449231335&idx=1&sn=d3ba98841e85b7cea0049cc43b3c16ca

ver.1 的问题在于一次训练所有的 windows，导致内存过大，具体来说就是 X 太大了；
这里通过分 batch 训练来解决这个问题，分批放到 lstm model 的 fit 中，每次 X 只是一批 windows 的矢量化数据
注意，fit 函数本身还有个 batch_size，这个是训练时分批做参数更新用的，在这里我们称为 minibatch

另外还处理了测试文本中有字不在训练文本中的问题；在初始化为 np.zeros 的条件下，只处理在训练文本中的字，那么不在的字自然 one-hot 都是 0
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation   # , Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np


class LstmSegmentor(object):
    def __init__(self, maxlen, step, maxmem):
        """
            maxlen    : 切分窗口长度，要求是奇数；比如7，表示考察一个字的前3个字和后3个字
            step      : 切分窗口间隔
            maxmem    : 一个 batch 占用的最大内存，按每个 np.bool 1 个字节计算
        """
        self.maxlen = maxlen
        self.step = step
        self.maxmem = maxmem

        self.chars = []   # uniq chars in the corpus
        self.char_indices = {}   # char -> index
        self.tags = []    # tags to classify
        self.tag_indices = {}    # tags -> index

    def pretraining(self, train_text, train_tags):
        """
            train_text: 待训练文本
            train_tags: 待训练tags，和train_text等长，且一一对应
        """
        # 预处理：把样本中的所有字符映射为数字
        print('corpus length:', len(train_text))
        # 字符集中加入 '\01'，不可见字符，用于在 train_text 前后 padding
        half_window = self.maxlen / 2
        train_text = u'\01' * half_window + train_text + u'\01' * half_window
        self.chars = sorted(list(set(train_text)))
        print('chars length:', len(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))

        # 要学习的分词标记，分别表示词开始、词结尾、词中间、单字成词
        self.tags = ['B', 'E', 'M', 'S']
        # 在 train_tags 前后 padding 'S' 单字 tag
        train_tags = 'S' * half_window + train_tags + 'S' * half_window
        print('total tags:', len(self.tags))
        self.tag_indices = dict((c, i) for i, c in enumerate(self.tags))

        # 开始切分，用某个字的前 half_window + 后 half_window 个字来预测该字的 tag
        self.windows = []        # 把整个训练文本按长度 self.maxlen 间隔 self.step 切分的样本，相当于 X
        self.next_tags = []      # self.maxlen 长度的样本中间一个字所对应的 tag 的集合，相当于 y
        end_pos = len(train_text) - half_window
        for i in range(half_window, end_pos, self.step):
            self.windows.append(train_text[i - half_window: i + half_window + 1])
            self.next_tags.append(train_tags[i])
        print('nb sequences:', len(self.windows))

        # 计算 batch size, np.bool 占 1 个字节
        batchsize = self.maxmem / self.maxlen / len(self.chars) / 1
        # 计算一共多少个 batches
        batchnum = (len(self.windows) - 1) / batchsize + 1
        print ('batchsize based on mem: ', batchsize)
        print ('no. of batches: ', batchnum)

        return batchsize, batchnum

    def vectorize_per_batch(self, batchsize, idx):
        # 从 self.windows & self.next_tags 中，根据批次号 idx，取出对应的一批样本，然后向量化
        # 对于每个单个样本，x 为 self.maxlen * len(self.chars) 维，one-hot； Y为 len(tags) 维，相当于 self.tags 的 one-hot
        print('Vectorization...')
        spos = idx * batchsize
        epos = (idx + 1) * batchsize
        windows = self.windows[spos: epos]
        next_tags = self.next_tags[spos: epos]
        # 考虑到不一定整除，故此，下面不能轻率的使用 batchsize 替换 len(windows)
        self.X = np.zeros((len(windows), self.maxlen, len(self.chars)), dtype=np.bool)
        self.y = np.zeros((len(windows), len(self.tags)), dtype=np.bool)
        for i, sentence in enumerate(windows):
            for t, char in enumerate(sentence):
                self.X[i, t, self.char_indices[char]] = 1
            self.y[i, self.tag_indices[next_tags[i]]] = 1

        # return X, y

    def training(self, batchsize, batchnum, hidden_nodes=128, minibatch_size=128, nb_epoch=1):
        # build the model: 2 stacked LSTM
        model = self.get_lstm_model(hidden_nodes, (self.maxlen, len(self.chars)), len(self.tags))
        # train
        for iteration in range(1, 61):
            self.print_iteration_sign(iteration)
            for batch_idx in range(batchnum):
                print("   |__ batch: ", batch_idx)
                self.vectorize_per_batch(batchsize, batch_idx)
                model.fit(self.X, self.y, batch_size=minibatch_size, nb_epoch=nb_epoch)
            yield model

    def run_test(self, model, test_text, diversity):
        """
            model: 用于预测的模型
            test_text: 待预测字符串
            diversity: 调整抽样概率
        """
        print('test text length: ', len(test_text))
        print('>>>>> diversity: ', diversity)
        half_window = self.maxlen / 2
        # padding with '\01'
        test_text = u'\01' * half_window + test_text + u'\01' * half_window
        # 初始化预测结果
        next_tags = ''
        end_pos = len(test_text) - half_window
        # 每个字都要预测，故此显然不要设置 step
        for i in range(half_window, end_pos):
            if (i + 1) % 1000 == 0:
                print("{} words predicted".format(i + 1))
            window = test_text[i - half_window: i + half_window + 1]
            x = np.zeros((1, self.maxlen, len(self.chars)))
            for t, char in enumerate(window):
                if char in self.char_indices:
                    x[0, t, self.char_indices[char]] = 1
            preds = model.predict(x, verbose=0)[0]
            next_index = self.sample(preds, diversity)
            next_tags += self.tags[next_index]
        return next_tags

    def get_lstm_model(self, hidden_nodes, input_shape, output_nodes):
        print('Build model...')
        model = Sequential()
        model.add(LSTM(hidden_nodes, input_shape=input_shape))
        # 输出 len(tags) 维度
        model.add(Dense(output_nodes))
        model.add(Activation('softmax'))
        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def sample(self, preds, temperature=1.0):
        """
            给定一个多维预测结果，在其中抽样一个，并取出该抽样的 index
            temperature 用于调整输入的 preds
            temperature 为 1，则没有调整
            temparature < 1 会加大差距，比如 array([ 0.1 ,  0.15,  0.5 ,  0.25]) ---0.5---> array([ 0.02898551,  0.06521739,  0.72463768,  0.18115942])
            反之 temparature > 1 会均匀化，比如同样的数组，经过 1.3 变为  array([ 0.12757797,  0.17427316,  0.43999216,  0.2581567 ])
        """
        preds = np.asarray(preds).astype('float64')
        # return np.argmax(preds)
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        # 抽样，从所有预测维度中，只抽取一个作为结果
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def print_iteration_sign(self, i):
        print()
        print('-' * 50)
        print('Iteration: ', i)
