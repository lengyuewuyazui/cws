# encoding: utf-8

import codecs
import pandas as pd
import numpy as np
from gensim.models import word2vec
import cPickle
from itertools import chain
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import Sequential,Model
# from keras.models import Sequential, Graph
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation
# from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM
from sklearn.cross_validation import train_test_split
import nltk
from nltk.probability import FreqDist


def load_training_file(filename):
    # lines like ['a b c', 'd e f']
    # words like [['a', 'b', 'c'], ['d', 'e', 'f']]
    # tags like 'BMESSBES'
    lines = []
    words = []
    tags = ''
    with codecs.open(filename, 'r', 'utf-8') as fp:
        for line in fp:
            line = line.strip('\n').strip('\r')
            lines.append(line)
            ws = line.split()
            words.append(ws)
            for w in ws:
                if len(w) == 1:
                    tags += 'S'
                else:
                    tags += 'B' + 'M' * (len(w) - 2) + 'E'
    return lines, words, tags


def word_freq(lines):
    """ 返回 DataFrame，按词频倒序排列
        这个是词频统计，其实没有使用，用的是下面的字符频率统计函数
    """
    # default filter is base_filter(), which is '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    # 这样的话，比如 a-b-c 不会被当作一个词，而会被当作 a b c 三个词看待
    # 另外注意，不设置上限 nb_words
    token = Tokenizer(filters='')
    # token 只能接受 str 不能接受 unicode
    token.fit_on_texts(map(lambda x: x.encode('utf-8'), lines))
    wc = token.word_counts
    df = pd.DataFrame({'word': map(lambda x: x.decode('utf-8'), wc.keys()), 'freq': wc.values()})
    df.sort('freq', ascending=False, inplace=True)
    df['idx'] = np.arange(len(wc))
    return df


def char_freq(lines):
    """ 返回 DataFrame，按字符频率倒序排列 """
    corpus = nltk.Text(chain.from_iterable(lines))  # 需要一个长字符串，而不是字符串列表
    wc = FreqDist(corpus)
    df = pd.DataFrame({'word': wc.keys(), 'freq': wc.values()})
    df.sort('freq', ascending=False, inplace=True)
    df['idx'] = np.arange(len(wc.values()))
    return df


def word2vec_train(corpus, epochs=20, size=100, sg=1, min_count=1, num_workers=4, window=6, sample=1e-5, negative=5):
    """
    其实并未真正使用
    word-embedding 维度 size
    至少出现 min_count 次才被统计，由于要和 Tokenizer 统计词频中的词一一对应，故此这里 min_count 必须为 1
    context 窗口长度 window
    sg=0 by default using CBOW； sg=1 using skip-gram
    negative > 0, negative sampling will be used
    """
    w2v = word2vec.Word2Vec(workers=num_workers, sample=sample, size=size, min_count=min_count, window=window, sg=sg, negative=negative)
    np.random.shuffle(corpus)
    w2v.build_vocab(corpus)

    for epoch in range(epochs):
        print 'epoch: ', epoch
        np.random.shuffle(corpus)
        w2v.train(corpus)
        w2v.alpha *= 0.9    # learning rate
        w2v.min_alpha = w2v.alpha
    print 'word2vec done'
    word2vec.Word2Vec.save(w2v, 'w2v_model')
    return w2v


def word2vec_order(w2v, idx2word):
    """
    其实并未真正使用
    按 word index 顺序保存 word2vec 的 embedding 矩阵，而不是用 w2v.syn0
    """
    ordered_w2v = []
    for i in xrange(len(idx2word)):
        ordered_w2v.append(w2v[idx2word[i]])
    return ordered_w2v


def sent2veclist(sentence, word2idx, context=7):
    """
    本函数把一个文档转为数值形式，并处理未登录词和 padding
    然后把文档中的每个字取 context 窗口，然该词在窗口中间
    sentence 为词的列表，注意不是字符串；context 即窗口长度，设为奇数
    注意：转化方法为逐字，而不是逐词，这样才能和字的 tag 一一对应上
    """
    numlist = []
    for word in sentence:
        for c in word:
            numlist.append(word2idx[c if c in word2idx else u'U'])
    pad = context / 2
    numlist = [word2idx[u'P']] * pad + numlist + [word2idx[u'P']] * pad

    veclist = []
    # 文档中的第一个字 (注意不是词) ，idx=0，恰好窗口为 numlist[0:7]
    for i in xrange(len(numlist) - pad * 2):    # 注意不是 len(sentence)，因为 sentence 是 word 的集合，而不是 char 的集合
        veclist.append(numlist[i: i + context])
    return veclist


def prepare_train_data(filename):
    lines, words, tags = load_training_file(filename)
    freqdf = char_freq(lines)
    max_features = freqdf.shape[0]   # 词个数
    print "Number of words: ", max_features
    word2idx = dict((c, i) for c, i in zip(freqdf.word, freqdf.idx))
    idx2word = dict((i, c) for c, i in zip(freqdf.word, freqdf.idx))

    """
    if load_w2v_file:
        w2v = word2vec.Word2Vec.load('w2v_model')
    else:
        w2v = word2vec_train(words)
    print "Shape of word2vec model: ", w2v.syn0.shape
    ordered_w2v = word2vec_order(w2v, idx2word)
    """

    # 定义'U'为未登陆新字, 'P'为两头padding用途，并增加两个相应的向量表示
    char_num = len(idx2word)
    idx2word[char_num] = u'U'
    word2idx[u'U'] = char_num
    idx2word[char_num + 1] = u'P'
    word2idx[u'P'] = char_num + 1
    # ordered_w2v.append(np.random.randn(100, ))   # for u'U'
    # ordered_w2v.append(np.zeros(100, ))          # for u'P'

    # 生成训练 X/y 变量
    windows = list(chain.from_iterable(map(lambda x: sent2veclist(x, word2idx, context=7), words)))
    print "Length of train X: ", len(windows)
    print "Length of train y: ", len(tags)
    cPickle.dump(windows, open('training_chars.pickle', 'wb'))
    cPickle.dump(tags, open('training_tags.pickle', 'wb'))
    cPickle.dump(word2idx, open('training_w2idx.pickle', 'wb'))
    return windows, tags, word2idx


def run(windows, tags, word2idx, test_file, batch_size=128):
    label_dict = dict(zip(['B', 'M', 'E', 'S'], range(4)))
    num_dict = {n: l for l, n in label_dict.iteritems()}
    # tags 转化为数字
    train_label = [label_dict[y] for y in tags]

    train_X, test_X, train_y, test_y = train_test_split(np.array(windows), train_label, train_size=0.8, random_state=1)
    # label num -> one-hot vector
    Y_train = np_utils.to_categorical(train_y, 4)
    Y_test = np_utils.to_categorical(test_y, 4)
    # 词典大小
    max_features = len(word2idx)
    word_dim = 100
    maxlen = 7  # 即 context
    hidden_units = 100

    model = build_lstm_model(max_features, word_dim, maxlen, hidden_units)
    print('train model ...')
    model.fit(train_X, Y_train, batch_size=batch_size, nb_epoch=20, validation_data=(test_X, Y_test))
    save_model(model)

    """
    graph training
    graph = build_lstm_graph(max_features, word_dim, maxlen, hidden_units)
    print('train graph ...')
    graph.fit({'input': train_X, 'output': Y_train, batch_size=batch_size, nb_epoch=20, validation_data=({'input': test_X, 'output': Y_test}))
    """

    """
    # model structure understanding, but seems not working with my keras version
    import theano
    print "LSTM model weights:"
    for w in model.get_weights():
        print(w.shape)
    layer = theano.function([model.layers[0].input], model.layers[3].get_output(train=False), allow_input_downcast=True)
    layer_out = layer(test_X[:10])
    print "example output shape of top 10 test_X: ", layer_out.shape   # 前 10 个窗口的第 0 层输出，经过了 relu 计算, should be (10, 4)
    """

    temp_txt = u'国家食药监总局发布通知称，酮康唑口服制剂因存在严重肝毒性不良反应，即日起停止生产销售使用。'
    temp_txt_list = list(temp_txt)
    temp_vec = sent2veclist(temp_txt_list, word2idx, context=7)
    print " ==> ", predict_sentence(temp_vec, temp_txt, model, label_dict, num_dict)
    segment_file(test_file, test_file + '.out', word2idx, model, label_dict, num_dict)


def build_lstm_model(max_features, word_dim, maxlen, hidden_units):
    print('stacking LSTM ...')
    model = Sequential()
    model.add(Embedding(max_features, word_dim, input_length=maxlen))
    model.add(Bidirectional(LSTM(output_dim=hidden_units)))
    # model.add(LSTM(output_dim=hidden_units, return_sequences=True))   # 中间层 lstm
    # model.add(LSTM(output_dim=hidden_units, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(4))    # 输出 4 个结果对应 BMES
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def build_lstm_graph(max_features, word_dim, maxlen, hidden_units):
    graph = Model()
    graph.add_input(name='input', input_shape=(maxlen,), dtype=int)
    graph.add_node(Embedding(max_features, word_dim, input_length=maxlen), name='embedding', input='input')
    graph.add_node(LSTM(output_dim=hidden_units), name='fwd', input='embedding')
    graph.add_node(LSTM(output_dim=hidden_units, go_backwards=True), name='backwd', input='embedding')
    graph.add_node(Dropout(0.5), name='dropout', input=['fwd', 'backwd'])
    graph.add_node(Dense(4, activation='softmax'), name='softmax', input='dropout')
    graph.add_output(name='output', input='softmax')
    graph.compile(loss={'output': 'categorical_crossentropy'}, optimizer='adam')
    return graph


def save_model(model):
    print "save model to json file"
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5 format
    model.save_weights("model.h5")


def load_model():
    from keras.models import model_from_json
    with open('model.json', 'r') as json_file:
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights("model.h5")
        return model
    return None


def predict_sentence(input_vec, input_txt, model, label_dict, num_dict):
    """
    给句子分词，然后根据逻辑调整不正常的分词结果，然后输出结果
    """
    input_vec = np.array(input_vec)
    predict_prob = model.predict_proba(input_vec, verbose=False)    # 得到每个 label 的概率
    predict_label = model.predict_classes(input_vec, verbose=False)   # 得到最可能的 label 值
    # fix 不正常的分词 label 错误，原则是根据前面的 label 纠正后面一个 label 的值，故此只取到最后一个结果 label 之前
    for i in xrange(len(predict_label) - 1):
        label = predict_label[i]
        # 首字不可为 E/M
        if i == 0:
            predict_prob[i, label_dict[u'E']] = 0
            predict_prob[i, label_dict[u'M']] = 0
            # 更新本 label
            predict_label[i] = predict_prob[i].argmax()
            label = predict_label[i]
        # 前字为B，后字不可为B,S
        if label == label_dict[u'B']:
            predict_prob[i + 1, label_dict[u'B']] = 0
            predict_prob[i + 1, label_dict[u'S']] = 0
        # 前字为E，后字不可为M,E
        if label == label_dict[u'E']:
            predict_prob[i + 1, label_dict[u'M']] = 0
            predict_prob[i + 1, label_dict[u'E']] = 0
        # 前字为M，后字不可为B,S
        if label == label_dict[u'M']:
            predict_prob[i + 1, label_dict[u'B']] = 0
            predict_prob[i + 1, label_dict[u'S']] = 0
        # 前字为S，后字不可为M,E
        if label == label_dict[u'S']:
            predict_prob[i + 1, label_dict[u'M']] = 0
            predict_prob[i + 1, label_dict[u'E']] = 0
        # 纠正之后，重新取下一个标签
        predict_label[i + 1] = predict_prob[i + 1].argmax()
    # 由标签数字来分词,空格分割
    return gen_segmented_sentence(predict_label, num_dict, input_txt)


def gen_segmented_sentence(predict_label, num_dict, input_txt):
    segs = []
    bpos = 0
    for i in xrange(len(predict_label)):
        if num_dict[predict_label[i]] in ['S', 'E']:
            segs.append(input_txt[bpos: i + 1])
            bpos = i + 1
    if bpos < len(predict_label):
        segs.append(input_txt[bpos:])
    return u' '.join(segs)


def segment_file(filename, fileout, word2idx, model, label_dict, num_dict):
    with codecs.open(filename, 'r', 'utf-8') as fp:
        with codecs.open(fileout, 'w', 'utf-8') as fout:
            for line in fp:
                line = line.strip()
                if line == "":
                    continue
                line_list = list(line)
                line_vec = sent2veclist(line_list, word2idx, context=7)
                seg_result = predict_sentence(line_vec, line, model, label_dict, num_dict)
                fout.write(seg_result + u'\n')


if __name__ == '__main__':
    import sys
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    windows, tags, word2idx = prepare_train_data(train_file)
    run(windows, tags, word2idx, test_file, batch_size=128)
