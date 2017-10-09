# encoding: utf-8

import codecs
import numpy as np
from gensim.models import word2vec
import cPickle
from itertools import chain
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from sklearn.cross_validation import train_test_split
from lstm_w2v_segment import char_freq, sent2veclist, save_model, gen_segmented_sentence


def load_training_file(filename, label_dict):
    # lines like ['a b c', 'd e f']
    # words like [['a', 'b', 'c'], ['d', 'e', 'f']]
    # tags like 'BMESSBES'
    #
    lines = []
    words = []
    tags = ''
    tagcnt = np.zeros((4))
    tagtranscnt = np.zeros((4, 4))
    with codecs.open(filename, 'r', 'utf-8') as fp:
        for line in fp:
            line = line.strip('\n').strip('\r')
            lines.append(line)
            ws = line.split()
            words.append(ws)
            line_tags = ''
            for w in ws:
                if len(w) == 1:
                    tags += 'S'
                    line_tags += 'S'
                    tagcnt[label_dict['S']] += 1
                else:
                    tags += 'B' + 'M' * (len(w) - 2) + 'E'
                    line_tags += 'B' + 'M' * (len(w) - 2) + 'E'
                    tagcnt[label_dict['B']] += 1
                    tagcnt[label_dict['E']] += 1
                    tagcnt[label_dict['M']] += len(w) - 2
            for c, l in zip(line_tags, line_tags[1:]):
                tagtranscnt[label_dict[c]][label_dict[l]] += 1
    return lines, words, tags, tagcnt, tagtranscnt


def make_word2vec_corpus(words):
    """
    words 参数是通过上面 load_training_file 函数得到的，是一个列表，列表中每个元素也是个列表，为训练文件中的每一行对应的空格分割的词
    例如：words: [ ['早上', '好'], ['天气', '不错'] ]
    word2vec 目的是为了生成 char 级别的 vector，故此需要把词打散
    故此，这里返回一个 generator，每一次返回每一行的 char 列表，如 ['早', '上', '好']
    """
    corpus = []
    for words_per_line in words:
        corpus.append(list(''.join(words_per_line)))
    return corpus


def word2vec_train(corpus, epochs=20, size=100, sg=1, min_count=1, num_workers=4, window=6, sample=1e-5, negative=5):
    """
    利用 word2vec 学习一个 char 级别的 vertors，用作 Embedding 层的初始 weight
    word-embedding 维度 size
    至少出现 min_count 次才被统计，由于要和所有字符一一对应，故此这里 min_count 必须为 1
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
    word2vec.Word2Vec.save(w2v, 'w2v_model.chars')
    return w2v


def load_or_train_w2vmodel(words):
    try:
        return word2vec.Word2Vec.load('w2v_model.chars')
    except:
        return word2vec_train(make_word2vec_corpus(words))


def prepare_train_data(lines, words, tags, tagcnt, tagtranscnt):
    freqdf = char_freq(lines)
    max_features = freqdf.shape[0]   # 词个数
    print "Number of words: ", max_features
    word2idx = dict((c, i) for c, i in zip(freqdf.word, freqdf.idx))
    idx2word = dict((i, c) for c, i in zip(freqdf.word, freqdf.idx))

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
    cPickle.dump(tagcnt, open('training_tagcnt.pickle', 'wb'))
    cPickle.dump(tagtranscnt, open('training_tagtranscnt.pickle', 'wb'))
    cPickle.dump(words, open('training_words.pickle', 'wb'))
    return windows, word2idx


def cal_probs(tagcnt, tagtranscnt, tags):
    total_tags = len(tags)
    initprob = tagcnt / total_tags
    transprob = tagtranscnt / sum(sum(tagtranscnt))
    return initprob, transprob


def cal_embedding_params(w2v_model, word2idx):
    # 使用 char 级别的 word2vec 模型为初始参数，那么就要让 Embedding 层的维度和 word2vec 模型维度一致
    word_dim = w2v_model.vector_size
    vec_unknown = [0] * word_dim
    weights = np.zeros((len(word2idx), word_dim))
    for word, idx in word2idx.iteritems():
        weights[idx, :] = w2v_model[word] if word in w2v_model else vec_unknown
    return word_dim, weights


def run(word_dim, weights, label_dict, num_dict, windows, tags, word2idx, test_file, batch_size=128):
    # tags 转化为数字
    train_label = [label_dict[y] for y in tags]

    train_X, test_X, train_y, test_y = train_test_split(np.array(windows), train_label, train_size=0.8, random_state=1)
    # label num -> one-hot vector
    Y_train = np_utils.to_categorical(train_y, 4)
    Y_test = np_utils.to_categorical(test_y, 4)
    # 词典大小
    max_features = len(word2idx)
    maxlen = 7  # 即 context
    hidden_units = 100

    model = build_lstm_model(word_dim, weights, max_features, maxlen, hidden_units)
    print('train model ...')
    model.fit(train_X, Y_train, batch_size=batch_size, nb_epoch=20, validation_data=(test_X, Y_test))
    save_model(model)

    temp_txt = u'国家食药监总局发布通知称，酮康唑口服制剂因存在严重肝毒性不良反应，即日起停止生产销售使用。'
    temp_txt_list = list(temp_txt)
    temp_vec = sent2veclist(temp_txt_list, word2idx, context=7)
    print " ==> ", predict_sentence(temp_vec, temp_txt, model, num_dict, initprob, transprob)
    segment_file(test_file, test_file + '.out', word2idx, model, num_dict, initprob, transprob)


def build_lstm_model(word_dim, weights, max_features, maxlen, hidden_units):
    print('stacking LSTM ...')
    model = Sequential()
    model.add(Embedding(max_features, word_dim, input_length=maxlen, mask_zero=True, weights=[weights]))
    model.add(LSTM(output_dim=hidden_units, return_sequences=True))   # 中间层 lstm
    model.add(LSTM(output_dim=hidden_units, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(4))    # 输出 4 个结果对应 BMES
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def viterbi(obs, states, initprob, transprob, emitprob):
    """ 参数为观察状态、隐藏状态、概率三元组(初始概率、转移概率、观察概率) """
    lenobs = len(obs)
    lenstates = len(states)
    V = np.zeros((lenobs, lenstates))
    path = np.zeros((lenstates, lenobs))

    # initial t0
    for y in range(lenstates):
        V[0][y] = initprob[y] * emitprob[y][0]
        path[y][0] = y

    for t in range(1, lenobs):
        newpath = np.zeros((lenstates, lenobs))
        for y in range(lenstates):
            prob = -1
            state = 0
            for y0 in range(lenstates):
                nprob = V[t - 1][y0] * transprob[y0][y] * emitprob[y][t]
                if nprob > prob:
                    prob = nprob
                    state = y0
                    V[t][y] = prob
                    newpath[y][:t] = path[state][:t]
                    newpath[y][t] = y
        path = newpath

    prob = -1
    state = 0
    for y in range(lenstates):
        if V[lenobs - 1][y] > prob:
            prob = V[lenobs - 1][y]
            state = y
    return prob, path[state]


def predict_sentence(input_vec, input_txt, model, num_dict, initprob, transprob):
    """
    给句子分词
    """
    input_vec = np.array(input_vec)
    predict_prob = model.predict_proba(input_vec, verbose=False)    # 得到每个 label 的概率
    prob, path = viterbi(input_vec, num_dict, initprob, transprob, predict_prob.transpose())
    return gen_segmented_sentence(path, num_dict, input_txt)


def segment_file(filename, fileout, word2idx, model, num_dict, initprob, transprob):
    with codecs.open(filename, 'r', 'utf-8') as fp:
        with codecs.open(fileout, 'w', 'utf-8') as fout:
            for line in fp:
                line = line.strip()
                if line == "":
                    continue
                line_list = list(line)
                line_vec = sent2veclist(line_list, word2idx, context=7)
                seg_result = predict_sentence(line_vec, line, model, num_dict, initprob, transprob)
                fout.write(seg_result + u'\n')


if __name__ == '__main__':
    import sys
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    label_dict = dict(zip(['B', 'M', 'E', 'S'], range(4)))
    num_dict = {n: l for l, n in label_dict.iteritems()}
    print "prepare data ..."
    try:
        windows = cPickle.load(open('training_chars.pickle', 'rb'))
        tags = cPickle.load(open('training_tags.pickle', 'rb'))
        word2idx = cPickle.load(open('training_w2idx.pickle', 'rb'))
        tagcnt = cPickle.load(open('training_tagcnt.pickle', 'rb'))
        tagtranscnt = cPickle.load(open('training_tagtranscnt.pickle', 'rb'))
        words = cPickle.load(open('training_words.pickle', 'rb'))
    except:
        lines, words, tags, tagcnt, tagtranscnt = load_training_file(train_file, label_dict)
        windows, word2idx = prepare_train_data(lines, words, tags, tagcnt, tagtranscnt)
    print "generate probs ..."
    initprob, transprob = cal_probs(tagcnt, tagtranscnt, tags)
    print "load or train w2v model"
    w2v_model = load_or_train_w2vmodel(words)
    print "generate embeding layer parameters"
    word_dim, weights = cal_embedding_params(w2v_model, word2idx)

    print "running ..."
    # run(word_dim, weights, label_dict, num_dict, windows, tags, word2idx, test_file)
    print "loading model ..."
    from lstm_w2v_segment import load_model
    model = load_model()
    print "doing segmentation ..."
    segment_file(test_file, test_file + '.viterbi.out', word2idx, model, num_dict, initprob, transprob)
