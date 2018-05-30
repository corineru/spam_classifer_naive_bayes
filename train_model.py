from Read_utils import *
import numpy as np


# 找到每个词的条件概率
def calculate_freq(text, labels):
    freq_table = np.zeros((2, len(text[0])))
    freq_table[0,:] = np.sum(text[labels=='1'],axis=0)
    freq_table[0, :] /= len(labels[labels=='1'])
    freq_table[1, :] = np.sum(text[labels=='0'], axis=0)
    freq_table[1, :] /= len(labels[labels=='0'])
    freq_table[freq_table!=0] = np.log(freq_table[freq_table!=0])
    return freq_table


def classify(sentence, freq_table, word_to_id_table):
    regularized_sentence = sentence_regularize(sentence)
    numerized_sentence = np.array(numerize_sentence(regularized_sentence, word_to_id_table))
    p_spam = np.sum(freq_table[0]*numerized_sentence)
    p_ham = np.sum(freq_table[1]*numerized_sentence)
    label = 1 if p_spam>p_ham else 0
    return label






