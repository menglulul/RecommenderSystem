#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# datamining project
from scipy.stats import pearsonr, tmean
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
import operator
import re
import time
import math


def read_data(file_path):
    raw_data = pd.read_csv(file_path).to_numpy()
    data = raw_data[:, 1:]
    return data


# split the data into 2 parts: traning set, validation set   
# i means return ith fold 
def fold(data, i, nfolds=5):
    n = len(data)
    n_left = round(n * (i / nfolds))
    n_right = round(n * ((i + 1) / nfolds))
    t1 = data[:n_left]
    t2 = data[n_left:n_right]
    t3 = data[n_right:]
    train_set = np.concatenate((t1, t3), axis=0)
    vali_set = t2
    return train_set, vali_set


def train_vali_split(data, ratio=0.7):
    n = len(data)
    cut = round(n * ratio)
    train_set = data[:cut, 1:]
    vali_set = data[cut:, 1:]
    return train_set, vali_set


def cal_pearson(u1, u2):
    corr, pvalue = pearsonr(u1, u2)
    # print("Pearsonr", corr)
    # print("p-value",pvalue)
    return corr


def cal_rate_sim(rate_train, rate_test):
    len_train = len(rate_train)
    len_test = len(rate_test)
    sim_mat = np.zeros((len_test, len_train))
    for i in range(len_test):
        for j in range(len_train):
            sim_mat[i][j] = cal_pearson(rate_train[j], rate_test[i])
    return sim_mat


def cal_tag_sim(tag_train, tag_test):
    len_train = len(tag_train)
    len_test = len(tag_test)
    vocab = gen_vocab(tag_train)
    data_train = bagOfWords(tag_train, vocab)
    data_test = bagOfWords(tag_test, vocab)
    sim_mat = np.zeros((len_test, len_train))
    for i in range(len_test):
        for j in range(len_train):
            sim_mat[i][j] = 1 / (np.linalg.norm(data_train[j] - data_test[i]) + 1)
    return sim_mat


def word_to_vec(text):
    # TODO
    return


def gen_vocab(tag_train):
    K = 200  # choose top K words
    dict = {}
    for text in tag_train:
        words = re.split(',', text)
        for word in words:
            if word in dict:
                dict[word] += 1
            else:
                dict[word] = 1
    vocab = []
    sorted_dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    cnt = 0
    for item in sorted_dict:
        vocab.append(item[0])
        cnt += 1
        if K == cnt:
            break
    print("vocab", vocab)
    return vocab


def bagOfWords(tag_raw, vocab):
    tag_mat = np.zeros((len(tag_raw), len(vocab)))
    for i in range(len(tag_raw)):
        words = re.split(',', tag_raw[i])
        for j in range(len(vocab)):
            for word in words:
                if vocab[j] == word:
                    tag_mat[i][j] += 1
    return tag_mat


def get_average_rating(ratings):
    '''
    get average rating of a movie based on available ratings by similar users
    args:
    @ratings (1darray) ratings of a movie by k similar users
    returns:
    @rating (float) average rating
    '''
    rating = 0.0
    try:
        # find the arithmetic mean of given values, ignoring values outside [0.5, 5]
        rating = tmean(ratings, (0.5, 5))
    except:
        # tmean() raises error if there are no values within given range
        # print('no valid ratings from similar users')
        pass
    return rating


def get_k_highest(arr, arr_target, k):
    '''
    find k largest elements from the target arr
    args:
    @arr (1darray) arr to be compared
    @arr_target (ndarray) arr to get k elements from
    @k (int)
    returns:
    (1darray) k elements in target arr
    '''
    idx = np.argpartition(arr, -k)[-k:]
    return arr_target[idx]


def pred_rating(rate_train, rate_sim, k):
    '''
    get predicted ratings based on ratings from k most similar users
    args:
    @rate_train (2darray) ratings of (col) movies, from (row) users
    @rate_sim (2darray) similarty scores between (row) users known and (col) users to be predicted
    returns:
    prediction (2darray) predicted ratings of (col) movies, for (row) users
    '''
    prediction = np.zeros((len(rate_sim), rate_train.shape[1]))
    for i in range(len(rate_sim)):
        sim = rate_sim[i]  # all user sim scores for i-th user
        sim_users = get_k_highest(sim, rate_train, k)  # ratings by k most similar users
        prediction[i] = np.apply_along_axis(get_average_rating, 0, sim_users)  # averaged ratings for i-th user
    # prediction = np.around((prediction * 2))/2

    return prediction


def weighted_time(u_ts):
    u_wt = np.zeros((u_ts.shape))
    for i in range(len(u_ts)):
        u_t = u_ts[i]
        hl = np.amin(u_t) + (np.amax(u_t) - np.amin(u_t)) / 2  # get mean timestamp
        t_recent = np.amax(u_t)
        func = lambda t: math.exp(-math.log(2) * (t_recent - t) / hl)
        u_wt[i] = np.array([func(x) for x in u_t])
    return u_wt


def evaluation(rate_prediction, rate_test):
    row = rate_test.shape[0]
    col = rate_test.shape[1]
    MSE = 0
    cnt = 0
    for i in range(row):
        for j in range(col):
            if rate_test[i][j] > 0 and rate_prediction[i][j] > 0:

                MSE += math.pow((rate_test[i][j] - rate_prediction[i][j]), 2)
                cnt += 1
    MSE /= cnt
    RMSE = sqrt(MSE)
    # RMSE = sqrt(mean_squared_error(rate_test, rate_prediction))
    print("RMSE", RMSE)
    return RMSE

if __name__ == "__main__":
    r_file_path = "new_processed_ratings.csv"
    ratings = read_data(r_file_path)
    print("ratings data load successfully")
    print("total: ", len(ratings))

    t_file_path = "new_processed_times.csv"
    times = read_data(t_file_path)
    print("time data load successfully")
    print("total: ", len(times))

    tags_file_path = "new_processed_tags.csv"
    tags = read_data(tags_file_path)
    tags = tags[:, 1]
    print("tags data load successfully")
    print("total: ", len(tags))

    # shuffle data
    np.random.seed(5)
    indices = np.arange(ratings.shape[0])
    np.random.shuffle(indices)
    ratings = ratings[indices]
    times = times[indices]
    tags = tags[indices]

    fold_n = 3
    k_list = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]
    # for testing with smaller dataset
    # ratings = ratings[:100]
    # times = times[:100]
    # tags = tags[:100]

    # part 1
    cv_rmse = np.zeros(len(k_list))
    print("base on ratings")
    for i in range(fold_n):
        ts = time.time()
        rating_train, rating_vali = fold(ratings, i, fold_n)
        rating_sim = cal_rate_sim(rating_train, rating_vali)
        for k in k_list:
            print("k", k)
            rating_prediction = pred_rating(rating_train, rating_sim, k)
            rmse = evaluation(rating_prediction, rating_vali)
            cv_rmse[k_list.index(k)] += rmse / fold_n
        print("cost time:", time.time() - ts)
    for k in k_list:
        print("cross validation rmse when k =", k)
        print(cv_rmse[k_list.index(k)])

    # part 2
    cv_rmse = np.zeros(len(k_list))
    print("base on time")
    for i in range(fold_n):
        ts = time.time()
        rating_train, rating_vali = fold(ratings, i, fold_n)
        time_train, time_vali = fold(times, i, fold_n)
        time_sim = cal_rate_sim(rating_train, rating_vali)
        for k in k_list:
            print("k", k)
            time_prediction = pred_rating(rating_train, time_sim, k)
            rmse = evaluation(time_prediction, rating_vali)
            cv_rmse[k_list.index(k)] += rmse / fold_n
        print("cost time:", time.time() - ts)
    for k in k_list:
        print("cross validation rmse when k =", k)
        print(cv_rmse[k_list.index(k)])

    # part 3
    cv_rmse = np.zeros(len(k_list))
    print("base on tags - wordbag")
    for i in range(fold_n):
        ts = time.time()
        rating_train, rating_vali = fold(ratings, i, fold_n)
        tag_train, tag_vali = fold(tags, i, fold_n)
        tag_sim = cal_tag_sim(tag_train, tag_vali)
        for k in k_list:
            print("k", k)
            tag_prediction = pred_rating(rating_train, tag_sim, k)
            rmse = evaluation(tag_prediction, rating_vali)
            cv_rmse[k_list.index(k)] += rmse / fold_n
        print("cost time:", time.time() - ts)
    for k in k_list:
        print("cross validation rmse when k =", k)
        print(cv_rmse[k_list.index(k)])

    # part 4
    cv_rmse = np.zeros(len(k_list))
    print("weighted time")
    for i in range(fold_n):
        ts = time.time()
        rating_train, rating_vali = fold(ratings, i, fold_n)
        time_train, time_vali = fold(times, i, fold_n)
        wtime_train = weighted_time(time_train)
        wtime_vali = weighted_time(time_vali)
        wtime_sim = cal_rate_sim(wtime_train, wtime_vali)
        for k in k_list:
            print("k", k)
            wtime_prediction = pred_rating(rating_train, wtime_sim, k)
            rmse = evaluation(wtime_prediction, rating_vali)
            cv_rmse[k_list.index(k)] += rmse / fold_n
        print("cost time:", time.time() - ts)
    for k in k_list:
        print("cross validation rmse when k =", k)
        print(cv_rmse[k_list.index(k)])

    # part 5
    # no cross-validation
    # k=1
    fold_n = 5
    print("combined similarity")
    ts = time.time()
    rating_train, rating_vali = fold(ratings, 0, fold_n)
    rating_sim = cal_rate_sim(rating_train, rating_vali)
    time_train, time_vali = fold(times, 0, fold_n)
    time_sim = cal_rate_sim(time_train, time_vali)
    tag_train, tag_vali = fold(tags, 0, fold_n)
    tag_sim = cal_tag_sim(tag_train, tag_vali)
    for i in np.arange(0,1.3,0.2):
        for j in np.arange(0,1.3,0.2):
            print("i",i)
            print("j",j)
            combined_sim = i*rating_sim + (1-i)*(j*time_sim+(1-j)*tag_sim)
            rating_prediction = pred_rating(rating_train, combined_sim, 100)
            rmse = evaluation(rating_prediction, rating_vali)
    print("cost time:",time.time()-ts)
