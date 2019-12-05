#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#datamining project
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
import operator
import re

def read_data(file_path):
    rate_data = pd.read_csv(file_path).drop(['userid'], axis =1).to_numpy()

    time_data = ''
    tag_data = ''
    return rate_data, time_data, tag_data

def train_vali_split(data, ratio=0.7):
    n = len(data)
    cut = round(n*ratio)
    train_set = data[:cut]
    vali_set = data[cut:]
    return train_set, vali_set

# def tag2vec(tag_train, tag_vali):
#     return tag_train_vec, tag_vali_vec

def cal_pearson(u1, u2):
    corr, pvalue = pearsonr(u1, u2)
    #print("Pearsonr", corr)
    #print("p-value",pvalue)
    return corr

def cal_rate_sim(rate_train, rate_test):
    len_train = len(rate_train)
    len_test = len(rate_test)
    sim_mat = np.zeros((len_test, len_train))
    for i in range(len_test):
        for j in range(len_train):
            sim_mat[i][j] = cal_pearson(rate_train[j], rate_test[i])   
    return sim_mat

# def cal_eucl_dis(u1, u2):
#     return dis
#
# def cal_combined_sim(rate_sim, time_sim, tag_sim, a, b, c):
#     return combined_sim

# def pred_rating(rate_train, rate_sim):
#     return rate_prediction

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

def pred_rating(rate_train, rate_sim, k):
    return rate_prediction

def evaluation(rate_prediction, rate_test):
    RMSE = sqrt(mean_squared_error(rate_test, rate_prediction))
    print("RMSE",RMSE)

if __name__ == "__main__":

    file_path = "fakerating.csv"
    rate_data, time_data, tag_data = read_data(file_path)
    print(rate_data)
    
    rate_train, rate_vali = train_vali_split(rate_data)
    sim_mat = cal_rate_sim(rate_train, rate_vali)
    
    k=2
    
    
    #predict the rating of each user in vali set
    rate_prediction = pred_rating(rate_train, rate_sim, k)
    
    evaluation(rate_prediction, rate_test)


    file_path = "processed_tags.csv"
    tag_data = load_tags(file_path)
    tag_train_raw, tag_vali_raw = train_vali_split(tag_data)
    vocab = gen_vocab(tag_train_raw)
    tag_train = bagOfWords(tag_train_raw, vocab)
    tag_vali = bagOfWords(tag_vali_raw, vocab)
    # rate_train, rate_vali = train_vali_split(rate_data)
    # time_train, time_vali = train_vali_split(time_data)
    # tag_train, tag_vali = train_vali_split(tag_data)
    # tag_train_vec, tag_vali_vec = tag2vec(tag_train, tag_vali)
    #
    # #find out how similar each user in train set and each user in vali set are
    # rate_sim = cal_rate_sim(rate_train, rate_vali)
    # time_sim = cal_rate_sim(rate_train, rate_vali)
    # tag_sim = cal_rate_sim(tag_train_vec, tag_vali_vec)
    #
    # sim_mat = cal_combined_sim(rate_sim, time_sim, tag_sim, a, b, c)
    #
    # #predict the rating of each user in vali set
    # rate_prediction = pred_rating(rate_train, rate_sim)
    #
    # evaluation(rate_prediction, rate_test)