# encoding: utf-8

import codecs
import numpy as np
import cPickle
from itertools import chain
from lstm_w2v_segment import char_freq, sent2veclist, load_model, gen_segmented_sentence


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


def prepare_train_data(filename, label_dict):
    lines, words, tags, tagcnt, tagtranscnt = load_training_file(filename, label_dict)
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
    return windows, tags, word2idx, tagcnt, tagtranscnt


def cal_probs(tagcnt, tagtranscnt, tags):
    total_tags = len(tags)
    initprob = tagcnt / total_tags
    transprob = tagtranscnt / sum(sum(tagtranscnt))
    return initprob, transprob


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
    except:
        windows, tags, word2idx, tagcnt, tagtranscnt = prepare_train_data(train_file, label_dict)
    print "generate probs ..."
    initprob, transprob = cal_probs(tagcnt, tagtranscnt, tags)

    print "loading model ..."
    model = load_model()
    print "doing segmentation ..."
    segment_file(test_file, test_file + '.viterbi.out', word2idx, model, num_dict, initprob, transprob)
