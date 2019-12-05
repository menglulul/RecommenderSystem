#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#datamining project
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

def read_data(file_path):
    rate_data = pd.read_csv(file_path).drop(['userid'], axis =1).to_numpy()

    time_data = ''
    tag_data = ''
    return rate_data, time_data, tag_data

# def train_vali_split(data):
#     return train_set, vali_set

# def tag2vec(tag_train, tag_vali):
#     return tag_train_vec, tag_vali_vec

def cal_pearson(u1, u2):
    corr, pvalue = pearsonr(u1, u2)
    print("Pearsonr", corr)
    print("p-value",pvalue)
    return corr

# def cal_eucl_dis(u1, u2):
#     return dis
#
# def cal_combined_sim(rate_sim, time_sim, tag_sim, a, b, c):
#     return combined_sim

# def pred_rating(rate_train, rate_sim):
#     return rate_prediction

def evaluation(rate_prediction, rate_test):
    RMSE = sqrt(mean_squared_error(rate_test, rate_prediction))
    print("RMSE",RMSE)

if __name__ == "__main__":

    file_path = "fakerating.csv"
    rate_data, time_data, tag_data = read_data(file_path)
    print(rate_data)

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