# Author: xinru
# 2018-05-30 20:55

import numpy as np
# 对每一句正则化
def sentence_regularize(sentence):
    new_sentence = ''
    for s in sentence:
        if s.isalpha():
            new_sentence += s.lower()
        elif s in [' ','\t','\n']:
            new_sentence += s
        else:
            continue
    return new_sentence

# 获取每一行的标签和处理后的正文
def get_label_and_data(data_path):
    labels, text = [], []
    with open(data_path) as f:
        for line in f.readlines():
            line = line.strip()
            label, sentence = line.split('\t')
            labels.append(label)
            sentence = sentence_regularize(sentence)
            text.append(sentence)
    labels = np.array(labels)
    text = np.array(text)
    labels[labels=='spam'] = 1
    labels[labels=='ham'] = 0
    return labels, text

# 获取字典
def get_dict(text):
    all_words = []
    for sentence in text:
        words = sentence.split(' ')
        all_words.extend(words)
    all_words = set(all_words)
    word_to_id_table = dict([(word, id) for (id, word) in enumerate(all_words)])
    return word_to_id_table

# 将一句话转化成字典长度的向量
def numerize_sentence(sentence, word_to_id_table):
    vector = [0]*(len(word_to_id_table) + 1)
    words = sentence.split(' ')
    for word in words:
        if word in word_to_id_table:
            id = word_to_id_table[word]
            vector[id] += 1
        else:
            vector[-1] += 1
    return vector


def numerize_text(text, word_to_id_table):
    all_vector = []
    for sentence in text:
        vector = numerize_sentence(sentence, word_to_id_table)
        all_vector.append(vector)
    all_vector = np.array(all_vector)
    return all_vector


