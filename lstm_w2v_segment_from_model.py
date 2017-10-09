# encoding: utf-8

import codecs
import numpy as np
import cPickle


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


def run(test_file, batch_size=128):
    word2idx = cPickle.load(open('training_w2idx.pickle', 'rb'))

    label_dict = dict(zip(['B', 'M', 'E', 'S'], range(4)))
    num_dict = {n: l for l, n in label_dict.iteritems()}

    print('loading model ...')
    model = load_model()

    temp_txt = u'国家食药监总局发布通知称，酮康唑口服制剂因存在严重肝毒性不良反应，即日起停止生产销售使用。'
    temp_txt_list = list(temp_txt)
    temp_vec = sent2veclist(temp_txt_list, word2idx, context=7)
    print " ==> ", predict_sentence(temp_vec, temp_txt, model, label_dict, num_dict)
    segment_file(test_file, test_file + '.out', word2idx, model, label_dict, num_dict)


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
    test_file = sys.argv[1]
    run(test_file, batch_size=128)
