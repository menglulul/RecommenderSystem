#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:58:29 2019

@author: lulumeng
"""
import operator
import pandas as pd
import numpy as np
import re


def load_tags(file_path):
    data = pd.read_csv(file_path)
    tag_data = data['tag']
    print(tag_data)
    return tag_data

def gen_vocab(tag_train):
    K = 200 # choose top K words
    dict = {}
    for text in tag_train:
        words = re.split(',',text)
        for word in words:
            if word in dict:
                dict[word] += 1
            else:
                dict[word] = 1
    vocab = []
    sorted_dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    cnt=0
    for item in sorted_dict:
        vocab.append(item[0])
        cnt +=1
        if K==cnt:
            break
    print(vocab)
    return vocab

def bagOfWords(tag_raw, vocab):
    tag_mat = np.zeros((len(tag_raw), len(vocab)))
    for i in len(tag_raw):
        words = re.split(',',tag_raw[i])
        for j in range(len(vocab)):
            for word in words:
                if vocab[j]==word:
                    tag_mat[i][j] += 1
    return tag_mat
    

if __name__ == "__main__":
    file_path = "processed_tags.csv"
    tag_data = load_tags(file_path)
    tag_train_raw, tag_vali_raw = train_vali_split(tag_data)
    vocab = gen_vocab(tag_train_raw)
    tag_train = bagOfWords(tag_train_raw, vocab)
    tag_vali = bagOfWords(tag_vali_raw, vocab)